// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The engine-neutral description of a deployed model: `{name}.json`, written
// by `molix.deploy.DeployMetadata` beside the packaged model. It states the
// model's own units, cutoff and receptive field, the package format, and the
// native operator libraries the package calls. An engine converts into its own
// unit system and sizes its own ghost shell; nothing here knows any engine.
//
// The fields and their validation mirror the Python writer one for one; a
// key this reader does not know is ignored.

#pragma once

#include <map>
#include <string>
#include <vector>

#include <c10/core/ScalarType.h>

namespace molnex::runtime {

/// One deployed model's `{name}.json`.
///
/// An aggregate, so a caller that has the facts already (a test, an engine
/// that embeds them) builds one directly; `read` / `parse` are the paths that
/// validate a document.
///
/// Example:
/// \code
///   const Metadata meta = Metadata::read(dir + "/model.json");
///   const double cutoff = meta.cutoff;  // in meta.length_unit
/// \endcode
struct Metadata {
  /// Package format, e.g. `"aoti"` (the package is `{name}.pt2`).
  std::string format;
  /// Neighbour cutoff, in `length_unit`.
  double cutoff = 0.0;
  /// Sorted atomic numbers the model accepts; never empty, every entry > 0.
  std::vector<int> species;
  /// The model's input dtype: `at::kFloat` or `at::kDouble`.
  at::ScalarType dtype = at::kDouble;
  /// Message-passing depth T: an atom's energy depends on every atom within
  /// `T * cutoff`.
  int receptive_field_layers = 1;
  /// The model's energy unit, e.g. `"eV"`.
  std::string energy_unit;
  /// The model's length unit, e.g. `"angstrom"`. Forces are
  /// `energy_unit / length_unit`.
  std::string length_unit;
  /// `{library: requirement}`: native libraries that register the non-ATen
  /// operators in the package, each with the upstream version its binding was
  /// built against. Empty for a package of ATen operators only.
  std::map<std::string, std::string> op_libraries;

  /// Read and validate `{name}.json` off disk.
  ///
  /// \throws std::runtime_error  If the file cannot be opened, or as `parse`.
  static Metadata read(const std::string& path);

  /// Parse and validate a document held in memory.
  ///
  /// \throws std::runtime_error  On a missing or mistyped field, an empty or
  ///                             non-positive `species`, a non-positive
  ///                             `cutoff`, a `receptive_field_layers` below
  ///                             1, or a `dtype` other than float32 /
  ///                             float64. The message names the field.
  static Metadata parse(const std::string& json_text);
};

}  // namespace molnex::runtime
