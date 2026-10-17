// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The `"aoti"` format: an AOTInductor `.pt2` package, written by
// `molix.export.aoti.package` (`torch.export.export` then
// `torch._inductor.aoti_compile_and_package`).
//
// Wraps torch::inductor::AOTIModelPackageLoader. Inference is loader->run;
// weight reload is loader->load_constants on the active buffer. The device is
// the package's own AOTI_DEVICE_KEY.

#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "molnex/runtime/model_artifact.h"

namespace molnex::runtime {

/// A loaded `.pt2` package.
///
/// `num_models > 1` requires `run_single_threaded = false`: torch >= 2.12
/// rejects the other combination inside the loader. Weight reload rewrites
/// the active constant buffer under `reload_mutex_`.
class AotiArtifact : public ModelArtifact {
 public:
  /// \param pt2_path   The package file.
  /// \param num_models Number of AOTI runners the loader allocates, >= 1.
  /// \param run_single_threaded  Forwarded to the loader. Must be false when
  ///                   `num_models > 1`.
  /// \throws std::runtime_error  If the file is missing, the arguments are
  ///                             inconsistent, or the package is a CUDA one
  ///                             and this build has no CUDA toolkit.
  explicit AotiArtifact(const std::string& pt2_path, int num_models = 1,
                        bool run_single_threaded = true);

  ~AotiArtifact() override;

  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) override;

  at::Device device() const override { return device_; }

  /// Reload constants from a `.pt` state_dict pickle (`{param_fqn: tensor}`).
  ///
  /// Concurrent `run()` calls block for the rewrite. The package must have
  /// been exported with `aot_inductor.use_runtime_constant_folding` so its
  /// parameters remain reloadable constants.
  void update_weights(const std::string& weight_path);

  /// Reload constants from an in-memory map (parameter FQN -> tensor).
  void update_weights(
      const std::unordered_map<std::string, at::Tensor>& params);

  /// Fully-qualified constant names the package exposes.
  std::vector<std::string> constant_names() const;

 private:
  at::Device device_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
  // run() holds this shared; update_weights() holds it exclusive across the
  // rewrite of the ACTIVE buffer. Without it a run already inside the model
  // can read constants while load_constants rewrites them — an intermittent
  // SIGSEGV under concurrent run-and-reload.
  std::shared_mutex reload_mutex_;
};

}  // namespace molnex::runtime
