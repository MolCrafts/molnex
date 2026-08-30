// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `graphed_exec.h` for the design notes. The preprocessor appears in this
// file and nowhere else on the execution axis: `pad_edges` is defined outside
// every guard (it is CPU arithmetic and must stay unit-testable on a machine
// with no CUDA toolkit), the constructor refuses a CPU-only build, and only
// `run`'s body is conditional.

#include "bridge/graphed_exec.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

PaddedEdges pad_edges(const at::Tensor& edge_index, int64_t e_max) {
  const int64_t ne = edge_index.size(0);
  if (ne > e_max) {
    throw std::runtime_error(
        "pad_edges: " + std::to_string(ne) +
        " edges exceed the fixed capacity e_max " + std::to_string(e_max) +
        "; a fixed-shape export cannot grow, and truncating would drop "
        "interactions and change the energy with no diagnostic (re-export with "
        "a larger e_max)");
  }

  // Zero-filled, so the padding rows are `(0, 0)` without a second pass. They
  // are a self-loop on atom 0 and are inert only because the mask marks them
  // false — which is why the mask is produced here rather than left to the
  // caller to remember.
  const auto opts = at::TensorOptions().device(at::kCPU);
  PaddedEdges out{at::zeros({e_max, 2}, opts.dtype(at::kLong)),
                  at::zeros({e_max}, opts.dtype(at::kBool))};
  if (ne > 0) {
    out.edges.narrow(0, 0, ne).copy_(edge_index);
    out.mask.narrow(0, 0, ne).fill_(true);
  }
  return out;
}

GraphedExec::GraphedExec(molnex::interface::ModelRunner* runner,
                         at::Device device, at::ScalarType model_dtype,
                         bool supports_virial, int64_t n_fixed, int64_t e_max)
    : Exec(model_dtype),
      runner_(runner),
      device_(device),
      model_dtype_(model_dtype),
      supports_virial_(supports_virial),
      n_fixed_(n_fixed),
      e_max_(e_max) {
#ifndef MOLNEX_INTERFACE_CUDA
  // Same dialect as ModelRunner's refusal of a device=cuda package on a
  // CPU-only build: the object would be constructible but its only method
  // unreachable, so it fails at the line that asked for it instead.
  throw std::runtime_error(
      "GraphedExec: the export asks for execution=cuda-graph, but "
      "libmolnex_interface was built CPU-only (no CUDA toolkit at build time); "
      "rebuild where CUDA is available, or re-export with cuda_graph=False");
#else
  if (runner_ == nullptr) {
    throw std::runtime_error(
        "GraphedExec: model runner is null; the pair style must load the model "
        "before it constructs an execution strategy");
  }
  if (n_fixed_ < 0 || e_max_ < 0) {
    throw std::runtime_error(
        "GraphedExec: n_fixed " + std::to_string(n_fixed_) + " and e_max " +
        std::to_string(e_max_) +
        " come from the export meta and must both be non-negative");
  }

  // Allocated once and never reassigned: the capture binds to these exact
  // addresses, so every step updates their CONTENTS in place. Reassigning any
  // of them would leave the captured graph reading freed memory.
  const auto opts = at::TensorOptions().device(device_);
  Zg_ = at::zeros({n_fixed_}, opts.dtype(at::kLong));
  posg_ = at::zeros({n_fixed_, 3}, opts.dtype(model_dtype_));
  edgeg_ = at::zeros({e_max_, 2}, opts.dtype(at::kLong));
  maskg_ = at::zeros({e_max_}, opts.dtype(at::kBool));
#endif
}

ModelOutputs GraphedExec::run(const at::Tensor& Z, const at::Tensor& pos_f64,
                              const at::Tensor& edge_index) {
#ifndef MOLNEX_INTERFACE_CUDA
  // Unreachable in practice — the constructor already refused — but the method
  // still has to exist so a CPU-only build links the class its tests name.
  (void)Z;
  (void)pos_f64;
  (void)edge_index;
  throw std::runtime_error(
      "GraphedExec::run: libmolnex_interface was built CPU-only (no CUDA "
      "toolkit at build time); there is no CUDA graph to replay");
#else
  // Both messages keep the migrated wording (the part a LAMMPS user greps for)
  // under this component's own prefix, as every other bridge component does.
  if (Z.size(0) != n_fixed_) {
    throw std::runtime_error(
        "GraphedExec: nlocal=" + std::to_string(Z.size(0)) +
        " != fixed N=" + std::to_string(n_fixed_) +
        " (export n_atoms must match the system)");
  }
  if (edge_index.size(0) > e_max_) {
    throw std::runtime_error(
        "GraphedExec: " + std::to_string(edge_index.size(0)) +
        " edges > e_max " + std::to_string(e_max_) +
        " (re-export with a larger e_max)");
  }

  const PaddedEdges padded = pad_edges(edge_index, e_max_);

  // In-place content updates, then replay. `copy_` handles both the host->device
  // move and any dtype narrowing; `pos` is cast first so the conversion happens
  // once on the CPU rather than as a device-side reinterpretation.
  Zg_.copy_(Z);
  posg_.copy_(pos_f64.to(model_dtype_));
  edgeg_.copy_(padded.edges);
  maskg_.copy_(padded.mask);

  const std::vector<at::Tensor> inputs = {Zg_, posg_, edgeg_, maskg_};
  // `run_graphed`'s outputs live at the captured graph's fixed addresses and
  // the next replay overwrites them in place; `read_model_outputs` moves them
  // to the CPU, which is what makes the returned tensors the caller's own.
  return read_model_outputs(runner_->run_graphed(inputs), supports_virial_);
#endif
}

}  // namespace molnex::lammps_bridge
