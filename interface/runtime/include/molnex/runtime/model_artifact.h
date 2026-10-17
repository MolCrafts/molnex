// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// A loaded model package, whatever its format.
//
// The format is the metadata's `format` field; each format is one subclass
// (`AotiArtifact` for `"aoti"`). Callers above this seam — the site-energy
// execution, an engine adapter — hold a `ModelArtifact&` and never name a
// format.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>

#include "molnex/runtime/metadata.h"

namespace molnex::runtime {

/// One loaded model package.
class ModelArtifact {
 public:
  virtual ~ModelArtifact();

  ModelArtifact(const ModelArtifact&) = delete;
  ModelArtifact& operator=(const ModelArtifact&) = delete;

  /// Run the packaged program. Inputs must already live on `device()`.
  virtual std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs) = 0;

  /// The device the package was built for.
  virtual at::Device device() const = 0;

  /// Load `{dir}/{name}.<ext>` in the format `metadata.format` names, after
  /// loading the native operator libraries `metadata.op_libraries` lists
  /// (searched as `OpLibraries::from_environment`).
  ///
  /// \throws std::runtime_error  If an op library cannot be loaded, if the
  ///                             format has no loader in this build, or as the
  ///                             format's own loader.
  static std::unique_ptr<ModelArtifact> open(const std::string& dir,
                                             const std::string& name,
                                             const Metadata& metadata);

 protected:
  ModelArtifact() = default;
};

}  // namespace molnex::runtime
