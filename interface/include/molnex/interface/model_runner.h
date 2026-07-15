// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// C++ runtime for AOT-Inductor exported MolNex models.
//
// Wraps torch::inductor::AOTIModelPackageLoader with a thin facade that
// loads a `.pt2` package (produced by molix.export.Exporter), borrows
// the loader's AOTIModelContainerRunner, and exposes double-buffered
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
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#ifdef MOLNEX_INTERFACE_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace molnex::interface {

/// Thin facade over a torch::inductor::AOTIModelPackageLoader.
///
/// Construction loads the export directory's `<name>.pt2` package; the
/// loader builds the right CPU/CUDA runner for the package's device
/// (we cross-check it against the `<name>.meta.json` `device` field).
/// After construction the runner is ready to serve `run()` calls. Pass
/// `num_models >= 2` (the default) to enable double-buffered weight
/// hot-reload via `update_weights()`; the active buffer keeps serving
/// traffic while the inactive buffer is rewritten, then a single atomic
/// swap flips them — no inference pause, no caller-visible lock.
class ModelRunner {
 public:
  /// Load a model from an export directory produced by
  /// `molix.export.Exporter` / `molix.engine.export_for_lammps`.
  ///
  /// \param model_dir  Path to the export directory. Must contain
  ///                   `<name>.pt2` and `<name>.meta.json`.
  /// \param num_models Number of constant buffers the underlying runner
  ///                   allocates. Default 2 enables double-buffered
  ///                   weight reload; pass 1 only when reload is
  ///                   guaranteed unused.
  /// \param name       Artifact basename inside `model_dir` (matches
  ///                   the `.pt2` stem written at export time).
  /// \param run_single_threaded  Load the AOTI runner in single-threaded
  ///                   mode. Required for CUDA-graph capture (the default
  ///                   multi-threaded run does a capture-illegal alloc/sync —
  ///                   PyTorch #158834, fixed torch 2.8); harmless otherwise.
  ModelRunner(const std::string& model_dir,
              int num_models = 2,
              const std::string& name = "model",
              bool run_single_threaded = true);

  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ModelRunner(ModelRunner&&) = delete;
  ModelRunner& operator=(ModelRunner&&) = delete;

  /// Synchronous inference. Inputs must already live on the runner's
  /// device.
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

#ifdef MOLNEX_INTERFACE_CUDA
  /// Asynchronous CUDA inference. Throws std::runtime_error if the
  /// runner was loaded for a CPU model. Only available when
  /// libmolnex_interface was built with a CUDA toolkit present.
  std::vector<at::Tensor> run_async(const std::vector<at::Tensor>& inputs,
                                    at::cuda::CUDAStream stream);

  /// CUDA-graph inference for FIXED-shape models. The first call warms up
  /// and captures `runner_->run(inputs)` into a CUDA graph; every later call
  /// just replays it. `inputs` MUST be the same persistent tensors each call
  /// (the caller updates their contents in place) — the graph reads/writes
  /// fixed device addresses. Returns the captured output tensors (also at
  /// fixed addresses; valid until the next replay). Requires the runner to
  /// have been loaded `run_single_threaded`. CUDA only.
  std::vector<at::Tensor> run_graphed(const std::vector<at::Tensor>& inputs);
#endif

  /// Reload constants from a `.pt` state_dict file (a plain
  /// ``{param_fqn: tensor}`` pickle, e.g. one saved with
  /// ``torch.save(model.state_dict(), ...)``). Performs a
  /// double-buffered swap: the inactive buffer is rewritten, then
  /// swapped atomically with the active one. The active buffer keeps
  /// serving `run()` calls until the swap point.
  ///
  /// The model must have been exported with
  /// ``inductor_configs={"aot_inductor.use_runtime_constant_folding":
  /// True}`` so its parameters remain reloadable constant buffers, and
  /// the runner constructed with `num_models >= 2`.
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
  // Owns the unpacked `.pt2` and the runner it builds. `runner_` is a
  // non-owning pointer borrowed from `loader_` (valid for loader_'s
  // lifetime); keep `loader_` declared first so it outlives every use.
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
  torch::inductor::AOTIModelContainerRunner* runner_ = nullptr;
#ifdef MOLNEX_INTERFACE_CUDA
  // CUDA-graph state (run_graphed): captured once, replayed thereafter.
  std::unique_ptr<at::cuda::CUDAGraph> graph_;
  std::vector<at::Tensor> graph_outputs_;
  bool graph_captured_ = false;
#endif
  // Serializes update_weights() callers so two reloads can't race on
  // the inactive buffer. run() is *not* serialized — the underlying
  // runner exposes the active buffer lock-free.
  std::mutex reload_mutex_;
};

}  // namespace molnex::interface
