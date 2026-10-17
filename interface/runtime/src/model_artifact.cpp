// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See model_artifact.h.

#include "molnex/runtime/model_artifact.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "molnex/runtime/aoti_artifact.h"
#include "molnex/runtime/op_libraries.h"

namespace molnex::runtime {

// Out of line so the vtable is emitted once.
ModelArtifact::~ModelArtifact() = default;

std::unique_ptr<ModelArtifact> ModelArtifact::open(const std::string& dir,
                                                   const std::string& name,
                                                   const Metadata& metadata) {
  // Before the package: its loader resolves every operator it calls.
  const OpLibraries libraries = OpLibraries::from_environment();
  for (const auto& [library, requirement] : metadata.op_libraries) {
    libraries.load(library, requirement);
  }

  if (metadata.format == "aoti") {
    return std::make_unique<AotiArtifact>(
        (std::filesystem::path(dir) / (name + ".pt2")).string());
  }
  throw std::runtime_error("model format '" + metadata.format +
                           "' has no loader in this build; known: aoti");
}

}  // namespace molnex::runtime
