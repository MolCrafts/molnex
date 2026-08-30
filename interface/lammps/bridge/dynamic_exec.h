// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The per-step execution strategy: whatever shapes this timestep produced go
// straight to the model.
//
// This is the universal path. It places no requirement on the export beyond the
// three-input cutoff-graph contract, works on CPU and CUDA alike, and is what
// runs whenever the meta does not ask for `execution = cuda-graph`. `GraphedExec`
// is the specialisation; this is the baseline it has to agree with.

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "bridge/exec.h"
#include "molnex/interface/model_runner.h"

namespace molnex::lammps_bridge {

/// Runs the model on this step's shapes, via `ModelRunner::run`.
///
/// Pure delegation: the only work done here is the device/dtype policy the
/// `Exec` interface keeps private (`Z` and `edge_index` to the model's device,
/// `pos` cast to the model's dtype and then moved) and the validated output
/// conversion, which is `read_model_outputs`' job and not duplicated.
class DynamicExec : public Exec {
 public:
  /// \param runner          Loaded model runner. **Borrowed, not owned** — the
  ///                        pair style outlives its `Exec` and keeps the runner
  ///                        alive for its whole lifetime.
  /// \param device          Device the runner's package was built for.
  /// \param model_dtype     Dtype the model expects `pos` in, from the export
  ///                        meta's `model_dtype`.
  /// \param supports_virial Whether the export declared a virial output.
  /// \throws std::runtime_error  If `runner` is null, or `model_dtype` is
  ///                             neither float32 nor float64.
  DynamicExec(molnex::interface::ModelRunner* runner, at::Device device,
              at::ScalarType model_dtype, bool supports_virial);

  /// Evaluate the model on the staged tensors as they stand.
  ///
  /// See `Exec::run` for the parameter and ownership contracts. Any shape the
  /// export's dynamic dimensions accept is accepted here; nothing is padded and
  /// no mask is passed, because there is no fixed shape to pad to.
  ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
                   const at::Tensor& edge_index) override;

 private:
  // The runner alone: the device, the model dtype and the virial flag are the
  // base's, because they are the same three facts for every strategy.
  molnex::interface::ModelRunner* runner_;
};

}  // namespace molnex::lammps_bridge
