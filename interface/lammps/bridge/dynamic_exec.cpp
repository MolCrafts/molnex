// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `dynamic_exec.h` for the design notes.

#include "bridge/dynamic_exec.h"

#include <vector>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

DynamicExec::DynamicExec(molnex::interface::ModelRunner* runner,
                         at::Device device, at::ScalarType model_dtype,
                         bool supports_virial)
    : Exec(device, model_dtype, supports_virial), runner_(runner) {
  // Borrowed, never owned — but "borrowed" still means "present". The refusal
  // is the base's, so both strategies fail in the same words.
  require_runner(runner_, "DynamicExec");
}

ModelOutputs DynamicExec::run(const at::Tensor& Z, const at::Tensor& pos_f64,
                              const at::Tensor& pair_index,
                              const at::Tensor& local_mask) {
  // The dtype cast the staging layer deliberately did not do: `stage_nodes`
  // keeps `pos` in fp64 so the coordinates that reach here are exactly LAMMPS'
  // own, and the single narrowing to the model's dtype happens at the one place
  // that knows what the model wants. `local_mask` is NOT cast: it is a bool in
  // the export's signature, and the multiplication against it happens inside
  // the traced graph.
  const std::vector<at::Tensor> inputs = {
      Z.to(device_),
      pos_f64.to(model_dtype_).to(device_),
      pair_index.to(device_),
      local_mask.to(device_),
  };
  // The atom count the per-atom energy is checked against is the one that was
  // just staged — read from `Z` rather than carried as a member, so it cannot
  // describe a previous step.
  return read_model_outputs(runner_->run(inputs),
                            OutputExpectations{supports_virial_, Z.size(0)});
}

}  // namespace molnex::lammps_bridge
