// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See metadata.h.
//
// Why a hand-rolled scanner instead of a JSON library: the documents are a few
// hundred bytes of fixed-schema key/value pairs written by molix, never
// user-authored, and a JSON dependency in a library linked into third-party
// engines costs more than it buys. The scanner is correspondingly narrow — it
// understands string / number / array / object values at one object's top
// depth and nothing else (no escape decoding, no unicode escapes, no
// comments).

#include "molnex/runtime/metadata.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace molnex::runtime {
namespace {

/// One `{...}` object of the document. Every accessor resolves at this
/// object's top depth only, so a key nested in a sub-object neither shadows nor
/// satisfies a lookup here.
class JsonObject {
 public:
  explicit JsonObject(std::string text) : text_(std::move(text)) {}

  std::string get_string(const std::string& key) const {
    const std::size_t v = require_value(key);
    if (text_[v] != '"') fail("'" + key + "' is not a string");
    const std::size_t end = scan_string(text_, v);
    if (end == std::string::npos) fail("'" + key + "' string not terminated");
    return text_.substr(v + 1, end - v - 1);
  }

  double get_number(const std::string& key) const {
    const std::size_t v = require_value(key);
    const char* begin = text_.c_str() + v;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) fail("'" + key + "' is not a number");
    return value;
  }

  /// Every element must be a whole decimal integer that fits `int`: a dropped
  /// element of an identity list is indistinguishable from one never declared.
  std::vector<int> get_int_array(const std::string& key) const {
    const std::size_t v = require_value(key);
    if (text_[v] != '[') fail("'" + key + "' is not an array");
    std::vector<int> out;
    std::size_t i = skip_ws(text_, v + 1);
    if (i < text_.size() && text_[i] == ']') return out;
    while (i < text_.size()) {
      std::size_t end = i;
      while (end < text_.size() && text_[end] != ',' && text_[end] != ']')
        ++end;
      if (end == text_.size()) break;
      std::size_t stop = end;
      while (stop > i && is_space(text_[stop - 1])) --stop;
      out.push_back(parse_int(text_.substr(i, stop - i), key));
      if (text_[end] == ']') return out;
      i = skip_ws(text_, end + 1);
    }
    fail("'" + key + "' array not terminated");
  }

  JsonObject block(const std::string& key) const {
    const std::size_t v = require_value(key);
    if (text_[v] != '{') fail("'" + key + "' is not an object");
    const std::size_t close = match_brace(text_, v);
    if (close == std::string::npos) fail("'" + key + "' object not terminated");
    return JsonObject(text_.substr(v, close - v + 1));
  }

  /// Every top-depth `"key": "value"` pair of this object, in document order.
  std::vector<std::pair<std::string, std::string>> string_entries() const {
    std::vector<std::pair<std::string, std::string>> out;
    std::size_t i = 0;
    int depth = 0;
    while (i < text_.size()) {
      const char c = text_[i];
      if (c == '"') {
        const std::size_t end = scan_string(text_, i);
        if (end == std::string::npos) fail("unterminated string literal");
        const std::size_t after = skip_ws(text_, end + 1);
        if (depth == 1 && after < text_.size() && text_[after] == ':') {
          std::string key = text_.substr(i + 1, end - i - 1);
          const std::size_t v = skip_ws(text_, after + 1);
          if (v >= text_.size() || text_[v] != '"') {
            fail("'" + key + "' is not a string");
          }
          const std::size_t value_end = scan_string(text_, v);
          if (value_end == std::string::npos)
            fail("unterminated string literal");
          out.emplace_back(std::move(key),
                           text_.substr(v + 1, value_end - v - 1));
          i = value_end + 1;
          continue;
        }
        i = end + 1;
      } else {
        if (c == '{' || c == '[') ++depth;
        if (c == '}' || c == ']') --depth;
        ++i;
      }
    }
    return out;
  }

 private:
  [[noreturn]] static void fail(const std::string& what) {
    throw std::runtime_error("metadata: " + what);
  }

  static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  static std::size_t skip_ws(const std::string& s, std::size_t i) {
    while (i < s.size() && is_space(s[i])) ++i;
    return i;
  }

  /// Index of the closing quote of the literal opening at `open`, or npos.
  static std::size_t scan_string(const std::string& s, std::size_t open) {
    for (std::size_t i = open + 1; i < s.size(); ++i) {
      if (s[i] == '\\') {
        ++i;
        continue;
      }
      if (s[i] == '"') return i;
    }
    return std::string::npos;
  }

  /// Index of the `}` closing the object at `open`, or npos; strings skipped.
  static std::size_t match_brace(const std::string& s, std::size_t open) {
    int depth = 0;
    for (std::size_t i = open; i < s.size(); ++i) {
      if (s[i] == '"') {
        i = scan_string(s, i);
        if (i == std::string::npos) return i;
      } else if (s[i] == '{') {
        ++depth;
      } else if (s[i] == '}' && --depth == 0) {
        return i;
      }
    }
    return std::string::npos;
  }

  static int parse_int(const std::string& token, const std::string& key) {
    const char* begin = token.c_str();
    char* end = nullptr;
    const long long value = std::strtoll(begin, &end, 10);
    if (token.empty() || end != begin + token.size()) {
      fail("'" + key + "' element '" + token + "' is not an integer");
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      fail("'" + key + "' element '" + token + "' does not fit in an int");
    }
    return static_cast<int>(value);
  }

  /// Index of `key`'s value at this object's top depth, or npos. A literal is
  /// a key only at depth 1 and followed by a colon, so a string value is never
  /// mistaken for a key and `"format_x"` never answers `"format"`.
  std::size_t find_value(const std::string& key) const {
    std::size_t i = 0;
    int depth = 0;
    while (i < text_.size()) {
      const char c = text_[i];
      if (c == '"') {
        const std::size_t end = scan_string(text_, i);
        if (end == std::string::npos) fail("unterminated string literal");
        const std::size_t after = skip_ws(text_, end + 1);
        if (depth == 1 && after < text_.size() && text_[after] == ':') {
          if (text_.compare(i + 1, end - i - 1, key) == 0) {
            return skip_ws(text_, after + 1);
          }
          i = after + 1;
          continue;
        }
        i = end + 1;
      } else {
        if (c == '{' || c == '[') ++depth;
        if (c == '}' || c == ']') --depth;
        ++i;
      }
    }
    return std::string::npos;
  }

  std::size_t require_value(const std::string& key) const {
    const std::size_t v = find_value(key);
    if (v == std::string::npos || v >= text_.size()) {
      fail("missing field '" + key + "'");
    }
    return v;
  }

  std::string text_;
};

at::ScalarType dtype_named(const std::string& name) {
  if (name == "float32") return at::kFloat;
  if (name == "float64") return at::kDouble;
  throw std::runtime_error("metadata: 'dtype' is '" + name +
                           "'; it must be float32 or float64");
}

}  // namespace

Metadata Metadata::read(const std::string& path) {
  std::ifstream is(path);
  if (!is) throw std::runtime_error("metadata: cannot open " + path);
  std::stringstream ss;
  ss << is.rdbuf();
  return parse(ss.str());
}

Metadata Metadata::parse(const std::string& json_text) {
  const JsonObject doc(json_text);

  Metadata meta;
  meta.format = doc.get_string("format");
  meta.cutoff = doc.get_number("cutoff");
  if (!(meta.cutoff > 0.0)) {
    throw std::runtime_error("metadata: 'cutoff' must be positive, got " +
                             std::to_string(meta.cutoff));
  }

  meta.species = doc.get_int_array("species");
  if (meta.species.empty()) {
    throw std::runtime_error(
        "metadata: 'species' is empty; the model declares no element");
  }
  for (const int z : meta.species) {
    if (z <= 0) {
      throw std::runtime_error("metadata: 'species' contains " +
                               std::to_string(z) +
                               ", which is not an atomic number");
    }
  }
  std::sort(meta.species.begin(), meta.species.end());
  meta.species.erase(std::unique(meta.species.begin(), meta.species.end()),
                     meta.species.end());

  meta.dtype = dtype_named(doc.get_string("dtype"));

  // No default: guessing 1 would let a deeper model pass an engine's ghost
  // shell check against a threshold several times too small.
  const double layers = doc.get_number("receptive_field_layers");
  meta.receptive_field_layers = static_cast<int>(layers);
  if (static_cast<double>(meta.receptive_field_layers) != layers ||
      meta.receptive_field_layers < 1) {
    throw std::runtime_error(
        "metadata: 'receptive_field_layers' is " + std::to_string(layers) +
        "; it must be a whole number of message-passing layers, at least 1");
  }

  meta.energy_unit = doc.get_string("energy_unit");
  meta.length_unit = doc.get_string("length_unit");
  for (auto& [library, requirement] :
       doc.block("op_libraries").string_entries()) {
    meta.op_libraries.emplace(std::move(library), std::move(requirement));
  }
  return meta;
}

}  // namespace molnex::runtime
