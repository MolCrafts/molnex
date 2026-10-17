// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Native operator libraries a package needs before it can be loaded.
//
// A package that calls a non-ATen operator (`cuequivariance::uniform_1d`, ...)
// resolves it through the torch dispatcher at load time. Python registers such
// operators when their package is imported; an engine process has no Python,
// so the registration has to come from a native library that does it in C++
// (`TORCH_LIBRARY_FRAGMENT`). The metadata names those libraries
// (`Metadata::op_libraries`); this type finds and loads them.
//
// A library is `lib<name>.so` or `lib<name>.<machine>.so` (molnex's own
// per-arch builds, e.g. `libmolnex_opLib.aarch64.so`), looked up in each directory of
// `MOLNEX_OP_LIBRARY_PATH` (colon-separated) in order, or by the dynamic
// loader's own search when that variable is unset. It must export
//
//     extern "C" const char* molnex_op_library_requirement();
//
// returning the upstream requirement it was built against, e.g.
// `"cuequivariance_ops_torch==0.12.0"`. A binding to another project's
// internal ABI is only valid for the version it was written for, so a
// mismatch with the metadata is refused rather than run.

#pragma once

#include <string>
#include <vector>

namespace molnex::runtime {

/// Finds and loads native operator libraries.
///
/// Loaded libraries are never closed: their operators are registered into the
/// process-wide torch dispatcher, and unloading code the dispatcher still
/// points at would crash at the next call or at exit.
class OpLibraries {
 public:
  /// \param search_path  Directories searched in order. Empty defers to the
  ///                     dynamic loader's own search (`LD_LIBRARY_PATH`, ...).
  explicit OpLibraries(std::vector<std::string> search_path);

  /// The search path from `MOLNEX_OP_LIBRARY_PATH`.
  static OpLibraries from_environment();

  /// Load `lib<library>.so` and check the requirement it was built against.
  ///
  /// \param library      Library name without prefix or suffix.
  /// \param requirement  What the package needs, e.g.
  ///                     `"cuequivariance_ops_torch==0.12.0"`.
  /// \throws std::runtime_error  If the library is not found (the message
  ///                             lists where it looked), does not export
  ///                             `molnex_op_library_requirement`, or was built
  ///                             against another requirement.
  void load(const std::string& library, const std::string& requirement) const;

 private:
  std::vector<std::string> search_path_;
};

}  // namespace molnex::runtime
