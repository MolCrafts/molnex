// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The CUDA-graph execution strategy: fixed atom count, edges padded to a fixed
// capacity, captured once and replayed every step thereafter.
//
// CAPTURE STATE STAYS IN `ModelRunner`. The `at::cuda::CUDAGraph`, the
// "already captured" flag and the captured output vector all live in
// `interface/src/model_runner.cpp`; this class holds only the persistent INPUT
// buffers whose contents it updates in place, and calls `run_graphed`. A second
// copy of the capture bookkeeping here would be a second thing that could
// believe it had captured when it had not.
//
// THE CLASS IS DECLARED UNCONDITIONALLY — no `#ifdef` around it. A CPU-only
// build still compiles, links and can construct-and-fail this type, which is
// what lets the test suite name it on a machine with no CUDA toolkit; the
// constructor refuses (in the dialect of `model_runner.cpp`'s CPU-only
// diagnostic) and only the body of `run` is preprocessor-guarded, inside the
// `.cpp`. The migrated code spread that `#ifdef` across `PairMolnex::compute`
// instead, which is exactly the shape this arrangement exists to retire.

#pragma once

#include <cstdint>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "bridge/exec.h"
#include "molnex/interface/model_runner.h"

namespace molnex::lammps_bridge {

/// A fixed-capacity edge tensor and the mask saying which of its rows are real.
///
/// Attribute-only aggregate, same shape rule as `NodeTensors` / `ModelOutputs`:
/// two tensors that travel together, no methods, no invariant beyond the shared
/// leading dimension their producer established.
struct PaddedEdges {
  /// Edges `(e_max, 2)` int64 on the CPU: the real rows first, then `(0, 0)`.
  at::Tensor edges;
  /// Validity `(e_max,)` bool on the CPU: true exactly on the real rows.
  at::Tensor mask;
};

/// Pad an edge tensor out to a fixed row count, with a validity mask.
///
/// A free function, and outside every CUDA guard, on purpose. It is a
/// zero-state CPU tensor computation, and `GraphedExec`'s end-to-end path needs
/// both a GPU and a fixed-shape export — so this is the piece that would
/// otherwise be untestable on the machines that run the suite. Extracting a
/// single-call-site helper is the exception CLAUDE.md names explicitly: a unit
/// test must be able to target this unit.
///
/// The padding rows are `(0, 0)`, i.e. a self-loop on atom 0. They are inert
/// only because `mask` marks them false and the model is expected to honour it;
/// the two outputs are meaningless apart, which is why they come back together.
///
/// \param edge_index  Real edges `(ne, 2)` int64 on the CPU, as
///                    `GraphBridge::build` produced them.
/// \param e_max       Fixed row count the export was compiled for.
/// \returns `PaddedEdges` owning freshly allocated `(e_max, 2)` int64 edges and
///          an `(e_max,)` bool mask; `edge_index` is neither aliased nor
///          modified.
/// \throws std::runtime_error  If `ne > e_max`. Truncating instead would drop
///                             interactions and change the energy with no
///                             diagnostic — the bound is the reason a
///                             fixed-shape export carries `e_max` at all.
PaddedEdges pad_edges(const at::Tensor& edge_index, int64_t e_max);

/// Runs the model at fixed shapes through a captured CUDA graph.
///
/// Requires the export to declare `execution = cuda-graph`, which fixes the
/// atom count `n_fixed` and the padded edge capacity `e_max`, and to take the
/// four inputs `(Z, pos, edge_index, mask)`.
class GraphedExec : public Exec {
 public:
  /// Allocate the persistent device input buffers the capture will bind to.
  ///
  /// \param runner          Loaded model runner. **Borrowed, not owned.**
  /// \param device          Device the runner's package was built for; must be
  ///                        CUDA for a graph-captured export.
  /// \param model_dtype     Dtype the model expects `pos` in.
  /// \param supports_virial Whether the export declared a virial output.
  /// \param n_fixed         Atom count the export was compiled for.
  /// \param e_max           Padded edge capacity the export was compiled for.
  /// \throws std::runtime_error  If the library was built without a CUDA
  ///                             toolkit (there is no replay path to reach), if
  ///                             `runner` is null, or if `n_fixed` / `e_max` is
  ///                             negative.
  GraphedExec(molnex::interface::ModelRunner* runner, at::Device device,
              at::ScalarType model_dtype, bool supports_virial, int64_t n_fixed,
              int64_t e_max);

  /// Copy this step's tensors into the persistent buffers and replay.
  ///
  /// See `Exec::run` for the parameter and ownership contracts. The ownership
  /// clause is at its sharpest here: `run_graphed` returns tensors at the fixed
  /// addresses the captured graph writes, so what comes back from this method
  /// is a CPU copy of them and never the buffers themselves.
  ///
  /// \throws std::runtime_error  If `Z` does not hold exactly `n_fixed` atoms,
  ///                             if `edge_index` holds more than `e_max` edges,
  ///                             or if the library was built CPU-only.
  ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
                   const at::Tensor& edge_index) override;

 private:
  // The runner and the fixed shapes: the device, the model dtype and the virial
  // flag are the base's, because they are the same three facts for every
  // strategy.
  molnex::interface::ModelRunner* runner_;
  int64_t n_fixed_;
  int64_t e_max_;
  // Persistent device inputs. The captured graph reads these exact addresses
  // every replay, so they are updated in place and never reassigned.
  at::Tensor Zg_, posg_, edgeg_, maskg_;
};

}  // namespace molnex::lammps_bridge
