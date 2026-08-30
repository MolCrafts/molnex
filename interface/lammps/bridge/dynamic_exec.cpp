// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `dynamic_exec.h` for the design notes.

#include "bridge/dynamic_exec.h"

#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

DynamicExec::DynamicExec(molnex::interface::ModelRunner* runner,
                         at::Device device, at::ScalarType model_dtype,
                         bool supports_virial)
    : Exec(model_dtype),
      runner_(runner),
      device_(device),
      model_dtype_(model_dtype),
      supports_virial_(supports_virial) {
  // Borrowed, never owned — but "borrowed" still means "present". Without this
  // the null would only surface as a segfault inside the first timestep, far
  // from the `pair_style` line that failed to supply a runner.
  if (runner_ == nullptr) {
    throw std::runtime_error(
        "DynamicExec: model runner is null; the pair style must load the model "
        "before it constructs an execution strategy");
  }
}

ModelOutputs DynamicExec::run(const at::Tensor& Z, const at::Tensor& pos_f64,
                              const at::Tensor& edge_index) {
  // The dtype cast the staging layer deliberately did not do: `stage_nodes`
  // keeps `pos` in fp64 so the coordinates that reach here are exactly LAMMPS'
  // own, and the single narrowing to the model's dtype happens at the one place
  // that knows what the model wants.
  const std::vector<at::Tensor> inputs = {
      Z.to(device_),
      pos_f64.to(model_dtype_).to(device_),
      edge_index.to(device_),
  };
  return read_model_outputs(runner_->run(inputs), supports_virial_);
}

}  // namespace molnex::lammps_bridge
