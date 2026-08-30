// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Scoped reader for the `<name>.meta.json` documents written by
// molix.export.Exporter / molix.engine.export_for_lammps.
//
// A meta.json describes an exported model to a C++ host that did not produce
// it: the device the export was built for, plus at most one `{...}`
// sub-document ("block") per consuming engine, each holding that engine's own
// contract fields. This reader stays deliberately ignorant of what any block
// means; it walks keys and values and nothing else.
//
// This header is the *neutral* layer of the meta contract: it is engine
// agnostic and carries no LAMMPS vocabulary (no block name, no unit style, no
// pair-style contract). Those live one layer up, in
// interface/lammps/lammps_contract.h, which is built on top of `block(name)`.
// Keeping the split means the next engine to link `libmolnex_interface` does
// not inherit a pile of concepts that belong to LAMMPS.
//
// Dependencies: pure std. No `torch/`, no `ATen/`, no LAMMPS headers — so the
// reader compiles and is testable on a machine with neither LibTorch nor a
// LAMMPS source tree.
//
// Why a hand-rolled scanner instead of a JSON library: this is a purpose-built
// reader for *our own writer's* output. The documents are a few hundred bytes
// of fixed-schema key/value pairs emitted by molix, they are never
// user-authored, and pulling a JSON dependency into a library that gets linked
// into third-party MD engines costs more than it buys. The scanner is
// correspondingly narrow — it understands string / number / bool / array /
// object values and nothing else (no escape decoding in returned strings, no
// unicode escapes, no comments), and it is documented as such rather than
// pretending to be a JSON parser.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace molnex::interface {

/// One meta.json document, or one `{...}` sub-document scoped out of it.
///
/// Every accessor resolves at the **top depth of this document only**. A key
/// nested inside a sub-object neither shadows nor satisfies a lookup at this
/// depth, and a key of an enclosing document is invisible from a scoped
/// sub-document. That scoping is what replaces the "key names are unique
/// within the file" promise of the retired interface/lammps/lammps_meta.h —
/// a promise which was unenforceable and which nested blocks had already
/// broken (`energy_conv.cutoff` would answer a lookup for `cutoff`).
///
/// A sub-document owns a copy of its own text, so it stays valid after the
/// document it was carved from is destroyed. meta.json is small; the copy is
/// cheaper than the lifetime coupling would be.
///
/// Construction deliberately validates nothing. The bare `Exporter` writes
/// metas holding only `device` and `model_class` — no blocks, no
/// `schema_version` — and `ModelRunner` must keep loading them. Contract
/// validation belongs to the vocabulary layer that owns the contract (see
/// `molnex::lammps::LammpsContract`), not here.
///
/// Example:
/// \code
///   const MetaSchema doc = MetaSchema::from_file(dir + "/model.meta.json");
///   const std::string dev = doc.device();
///   if (doc.has_block("lammps")) {
///     // units are the field's own; this one is a neighbour cutoff in angstrom
///     const double cutoff = doc.block("lammps").get_number("cutoff");
///   }
/// \endcode
class MetaSchema {
 public:
  /// Wrap a JSON document held in memory.
  ///
  /// \param json_text  The whole document, or a `{...}` sub-document. Kept by
  ///                   value: this object owns its text.
  explicit MetaSchema(std::string json_text) : text_(std::move(json_text)) {}

  /// Read a meta.json off disk.
  ///
  /// Alternate constructor with distinct semantics (file I/O), not a `make_*`
  /// alias of the constructor above.
  ///
  /// \param path  Path to the `<name>.meta.json` file.
  /// \returns     The document.
  /// \throws std::runtime_error  If the file cannot be opened.
  static MetaSchema from_file(const std::string& path) {
    std::ifstream is(path);
    if (!is) throw std::runtime_error("cannot open meta.json: " + path);
    std::stringstream ss;
    ss << is.rdbuf();
    return MetaSchema(ss.str());
  }

  /// The `device` field the export was produced for (`"cpu"` / `"cuda"`).
  ///
  /// **Top-level-only field.** Calling this on a sub-document returned by
  /// `block()` throws "missing 'device' field" rather than silently resolving
  /// to the enclosing document's value — scoped resolution guarantees the
  /// enclosing value is not reachable from here. The boundary is recorded
  /// rather than enforced by a separate `MetaBlock` type, which would only
  /// duplicate the whole read surface to remove one method.
  ///
  /// Error texts are inherited verbatim from the `parse_device_field` this
  /// method absorbed (formerly interface/src/model_runner.cpp), so existing
  /// load-failure diagnostics do not change wording.
  ///
  /// \returns The device string.
  /// \throws std::runtime_error  If the field is absent, is not a string, or
  ///                             is unterminated.
  std::string device() const {
    const std::size_t v = find_value("device");
    if (v == std::string::npos) {
      throw std::runtime_error("meta.json: missing 'device' field");
    }
    if (text_[v] != '"') {
      throw std::runtime_error("meta.json: 'device' value not a string");
    }
    const std::size_t end = scan_string(text_, v);
    if (end == std::string::npos) {
      throw std::runtime_error("meta.json: 'device' value not terminated");
    }
    return text_.substr(v + 1, end - v - 1);
  }

  /// Whether this document has a top-depth `name` key whose value is an object.
  ///
  /// The key is matched whole and quoted, so `"lammps_notes"` answers neither
  /// `has_block("lammps")` (prefix) nor `has_block("lammps_notes")` (its value
  /// is a string, not a block).
  ///
  /// \param name  Block key to look for.
  /// \returns     True only for a `{...}`-valued key at this depth.
  bool has_block(const std::string& name) const {
    const std::size_t v = find_value(name);
    return v != std::string::npos && text_[v] == '{';
  }

  /// Scope into the `{...}` value of a top-depth key.
  ///
  /// The result is an independent document: its own accessors resolve at *its*
  /// top depth, and it can be scoped again (`block("lammps").block(
  /// "energy_conv")`).
  ///
  /// \param name  Block key.
  /// \returns     The sub-document, owning a copy of its text.
  /// \throws std::runtime_error  If the key is absent, is not an object, or
  ///                             its braces are unbalanced.
  MetaSchema block(const std::string& name) const {
    const std::size_t v = find_value(name);
    if (v == std::string::npos) {
      throw std::runtime_error("meta.json: missing block '" + name + "'");
    }
    if (text_[v] != '{') {
      throw std::runtime_error("meta.json: '" + name + "' is not a block");
    }
    const std::size_t close = match_brace(text_, v);
    if (close == std::string::npos) {
      throw std::runtime_error("meta.json: block '" + name + "' not terminated");
    }
    return MetaSchema(text_.substr(v, close - v + 1));
  }

  /// Read a string-valued key at this document's top depth.
  ///
  /// The value is returned as it appears between the quotes; escape sequences
  /// are not decoded (none of the fields molix writes contain any).
  ///
  /// \param key  Key to read.
  /// \returns    The string value.
  /// \throws std::runtime_error  If the key is absent, is not a string, or is
  ///                             unterminated.
  std::string get_string(const std::string& key) const {
    const std::size_t v = require_value(key);
    if (text_[v] != '"') {
      throw std::runtime_error("meta.json: '" + key + "' is not a string");
    }
    const std::size_t end = scan_string(text_, v);
    if (end == std::string::npos) {
      throw std::runtime_error("meta.json: '" + key + "' string not terminated");
    }
    return text_.substr(v + 1, end - v - 1);
  }

  /// Read an array-of-strings key at this document's top depth.
  ///
  /// Returns **every** element. Reading only the first (what `get_string` would
  /// do on an array) is the defect this accessor exists to remove — the
  /// vocabulary layer's tensor-name lists are the motivating example.
  /// Non-string elements, and strings nested deeper inside the array, are
  /// skipped.
  ///
  /// \param key  Key to read.
  /// \returns    The array's string elements, in document order.
  /// \throws std::runtime_error  If the key is absent, is not an array, or the
  ///                             array is unterminated.
  std::vector<std::string> get_string_array(const std::string& key) const {
    const std::size_t v = require_value(key);
    if (text_[v] != '[') {
      throw std::runtime_error("meta.json: '" + key + "' is not an array");
    }
    std::vector<std::string> out;
    std::size_t i = v + 1;
    int depth = 1;
    while (i < text_.size() && depth > 0) {
      const char c = text_[i];
      if (c == '"') {
        const std::size_t end = scan_string(text_, i);
        if (end == std::string::npos) {
          throw std::runtime_error("meta.json: '" + key + "' array not terminated");
        }
        if (depth == 1) out.push_back(text_.substr(i + 1, end - i - 1));
        i = end + 1;
      } else if (c == '[' || c == '{') {
        ++depth;
        ++i;
      } else if (c == ']' || c == '}') {
        --depth;
        ++i;
      } else {
        ++i;
      }
    }
    if (depth != 0) {
      throw std::runtime_error("meta.json: '" + key + "' array not terminated");
    }
    return out;
  }

  /// Read a numeric key at this document's top depth.
  ///
  /// The number carries whatever unit its own field is defined in (a length is
  /// in angstrom, a count is dimensionless); this layer neither knows which is
  /// which nor converts between them.
  ///
  /// \param key  Key to read.
  /// \returns    The value as a double. A double represents every integer up to
  ///             2^53 exactly, so integral fields such as an atom count or an
  ///             edge capacity survive the round trip well past any system size
  ///             molecular dynamics will ask for.
  /// \throws std::runtime_error  If the key is absent or its value does not
  ///                             start a number.
  double get_number(const std::string& key) const {
    const std::size_t v = require_value(key);
    const char* begin = text_.c_str() + v;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) {
      throw std::runtime_error("meta.json: '" + key + "' is not a number");
    }
    return value;
  }

  /// Read a boolean key at this document's top depth, tolerating absence.
  ///
  /// Optional capability flags read this way: an older export that predates the
  /// flag means "not supported", not "malformed". A present-but-unparseable
  /// value also yields the fallback rather than throwing, matching the reader
  /// this method replaces.
  ///
  /// \param key       Key to read.
  /// \param fallback  Returned when the key is absent or not `true`/`false`.
  /// \returns         The flag.
  bool get_bool(const std::string& key, bool fallback) const {
    const std::size_t v = find_value(key);
    if (v == std::string::npos) return fallback;
    if (text_.compare(v, 4, "true") == 0) return true;
    if (text_.compare(v, 5, "false") == 0) return false;
    return fallback;
  }

 private:
  /// Index of the closing quote of the string literal opening at `open`.
  ///
  /// \param s     Text being scanned.
  /// \param open  Index of the opening `"`.
  /// \returns     Index of the closing `"`, or npos if unterminated. `\"` and
  ///              any other backslash escape do not close the literal.
  static std::size_t scan_string(const std::string& s, std::size_t open) {
    for (std::size_t i = open + 1; i < s.size(); ++i) {
      if (s[i] == '\\') {
        ++i;  // the escaped character cannot close the literal
        continue;
      }
      if (s[i] == '"') return i;
    }
    return std::string::npos;
  }

  /// Index of the `}` closing the object that opens at `open`.
  ///
  /// String literals are skipped, so braces inside string values do not
  /// unbalance the count.
  ///
  /// \returns Index of the matching `}`, or npos if unbalanced.
  static std::size_t match_brace(const std::string& s, std::size_t open) {
    int depth = 0;
    std::size_t i = open;
    while (i < s.size()) {
      const char c = s[i];
      if (c == '"') {
        const std::size_t end = scan_string(s, i);
        if (end == std::string::npos) return std::string::npos;
        i = end + 1;
      } else if (c == '{') {
        ++depth;
        ++i;
      } else if (c == '}') {
        --depth;
        if (depth == 0) return i;
        ++i;
      } else {
        ++i;
      }
    }
    return std::string::npos;
  }

  /// First index at or after `i` that is not JSON whitespace.
  static std::size_t skip_ws(const std::string& s, std::size_t i) {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
      ++i;
    }
    return i;
  }

  /// Index of the first character of `key`'s value, at this document's top
  /// depth only.
  ///
  /// The scanner walks the text tracking brace/bracket depth and skipping
  /// string literals, so braces, colons and commas inside string values do not
  /// confuse it. A string literal is treated as a key only when it sits at
  /// depth 1 (this document's top level) *and* is followed by a colon, which
  /// makes the match both prefix-safe (`"lammps_notes"` is not `"lammps"`) and
  /// value-safe (a string *value* is never mistaken for a key).
  ///
  /// \param key  Key to locate.
  /// \returns    Index of the value's first non-whitespace character, or npos
  ///             when this document has no such key at its top depth.
  /// \throws std::runtime_error  On an unterminated string literal, rather than
  ///                             reporting a key that follows it as absent.
  std::size_t find_value(const std::string& key) const {
    const std::string& s = text_;
    const std::size_t n = s.size();
    std::size_t i = 0;
    int depth = 0;
    while (i < n) {
      const char c = s[i];
      if (c == '"') {
        const std::size_t end = scan_string(s, i);
        if (end == std::string::npos) {
          throw std::runtime_error("meta.json: unterminated string literal");
        }
        const std::size_t after = skip_ws(s, end + 1);
        if (depth == 1 && after < n && s[after] == ':') {
          if (s.compare(i + 1, end - i - 1, key) == 0) {
            return skip_ws(s, after + 1);
          }
          i = after + 1;  // step past the colon; the value is scanned normally
          continue;
        }
        i = end + 1;
      } else if (c == '{' || c == '[') {
        ++depth;
        ++i;
      } else if (c == '}' || c == ']') {
        --depth;
        ++i;
      } else {
        ++i;
      }
    }
    return std::string::npos;
  }

  /// `find_value`, with absence promoted to the shared "missing field" error.
  std::size_t require_value(const std::string& key) const {
    const std::size_t v = find_value(key);
    if (v == std::string::npos) {
      throw std::runtime_error("meta.json: missing field '" + key + "'");
    }
    return v;
  }

  std::string text_;
};

}  // namespace molnex::interface
