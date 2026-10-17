// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Node-tensor staging: the LAMMPS atom arrays as the model's `Z` / `pos`.
//
// A free function, not a class, because it has no state and no configuration:
// a zero-state single-method class is a function in a wrapper. It also has no
// natural owning type — `LammpsView` is a borrowed isolation layer and hanging
// a tensor-building method on it would drag LibTorch into that layer, while
// `SpeciesMap` does not own positions.
//
// Two deliberate non-changes carried over from the inlined loop it replaces:
// tensors are allocated per call (no cross-step buffer reuse — that would be an
// unmeasured performance change smuggled into a structural migration), and
// `pos` stays fp64, with the model-dtype cast left at the execution site.

#pragma once

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"
#include "bridge/species_map.h"

namespace molnex::lammps_bridge {

/// The per-node model inputs staged out of a `LammpsView`.
///
/// Attribute-only aggregate, mirroring `molix.md.types.ForceOutput`: two named
/// tensors travelling together, with no invariant tying them beyond the shared
/// leading dimension the producer already established.
struct NodeTensors {
  /// Atomic numbers, `(n_graph,)` int64 on the CPU.
  at::Tensor Z;
  /// Positions, `(n_graph, 3)` float64 on the CPU, in angstrom.
  at::Tensor pos;
};

/// Stage the first `n_graph` atoms of `view` as model node inputs.
///
/// Values are copied, never computed: `Z[i]` is `species.z_of(view.type[i])`
/// and `pos[i][k]` is `view.x[i][k]`. Both tensors stay on the CPU in the
/// dtypes above; moving them to the model's device and dtype is the call
/// site's job.
///
/// \param view     Borrowed LAMMPS atom arrays.
/// \param species  Validated type -> Z map covering every type in `view`.
/// \param n_graph  Number of leading atoms to stage. The pair style passes
///                 `nall()` (locals and ghosts).
/// \returns Freshly allocated `Z` `(n_graph,)` int64 and `pos` `(n_graph, 3)`
///          float64, both on the CPU.
/// \throws std::runtime_error  If `n_graph` is outside `[0, view.nall()]` —
///                             `view.x` belongs to LAMMPS, so an unchecked
///                             overrun reads another object's memory — or if
///                             any atom carries a type the map rejects.
NodeTensors stage_nodes(const LammpsView& view, const SpeciesMap& species,
                        int n_graph);

}  // namespace molnex::lammps_bridge
