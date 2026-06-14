// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// C++ runtime for AOT-Inductor exported MolNex models.
//
// Wraps torch::inductor::AOTIModelContainerRunner{Cpu,Cuda} with a
// thin facade that hides the runner subclass selection (driven by the
// export's meta.json `device` field) and exposes double-buffered
// zero-stop weight reload via update_weights().
//
// This header is part of `libmolnex_interface`, a pure C++ library
// intended to be linked into external runtimes (e.g. LAMMPS plugins).
// It deliberately depends only on LibTorch — never on Python — and
// is not packaged in any wheel.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace molnex::interface {

/// Thin facade over a torch::inductor::AOTIModelContainerRunner.
///
/// Construction selects the concrete runner (CPU or CUDA) by reading
/// the export directory's `<name>.meta.json`. After construction the
/// runner is ready to serve `run()` calls. Pass `num_models >= 2` (the
/// default) to enable double-buffered weight hot-reload via
/// `update_weights()`; the active buffer keeps serving traffic while
/// the inactive buffer is rewritten, then a single atomic swap flips
/// them — no inference pause, no caller-visible lock.
class ModelRunner {
 public:
  /// Load a model from an export directory produced by
  /// `molix.export.export_model(...)`.
  ///
  /// \param model_dir  Path to the export directory. Must contain
  ///                   `<name>.so` and `<name>.meta.json`.
  /// \param num_models Number of constant buffers the underlying runner
  ///                   allocates. Default 2 enables double-buffered
  ///                   weight reload; pass 1 only when reload is
  ///                   guaranteed unused.
  /// \param name       Artifact basename inside `model_dir` (matches
  ///                   the `name=` argument passed to `export_model`).
  ModelRunner(const std::string& model_dir,
              int num_models = 2,
              const std::string& name = "model");

  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ModelRunner(ModelRunner&&) = delete;
  ModelRunner& operator=(ModelRunner&&) = delete;

  /// Synchronous inference. Inputs must already live on the runner's
  /// device.
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

  /// Asynchronous CUDA inference. Throws std::runtime_error if the
  /// runner was loaded for a CPU model.
  std::vector<at::Tensor> run_async(const std::vector<at::Tensor>& inputs,
                                    at::cuda::CUDAStream stream);

  /// Reload constants from a `.pt` state_dict file (the artifact
  /// `<model_dir>/<name>.pt` follows this format). Performs a
  /// double-buffered swap: the inactive buffer is rewritten, then
  /// swapped atomically with the active one. The active buffer keeps
  /// serving `run()` calls until the swap point.
  ///
  /// Requires the runner to have been constructed with `num_models >= 2`.
  void update_weights(const std::string& weight_path);

  /// Reload constants from an in-memory tensor map (parameter FQN →
  /// tensor). Semantics identical to the file-path overload.
  void update_weights(const std::unordered_map<std::string, at::Tensor>& params);

  /// (name, scalar-type-string) pairs for every constant the model
  /// container knows about. Useful for sanity-checking a reload map
  /// against the model's expected schema.
  std::vector<std::pair<std::string, std::string>> parameter_info() const;

  /// `"cuda"` or `"cpu"`, as resolved at load time.
  const std::string& device() const { return device_; }

 private:
  std::string device_;
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner_;
  // Serializes update_weights() callers so two reloads can't race on
  // the inactive buffer. run() is *not* serialized — the underlying
  // runner exposes the active buffer lock-free.
  std::mutex reload_mutex_;
};

}  // namespace molnex::interface
