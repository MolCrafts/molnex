// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The LAMMPS neighbour list as a molnex `edge_index`.
//
// A type rather than a free function because it has state with a lifetime of
// its own: `cutsq_` is fixed once from the model's cutoff at `pair_style`
// settings, and the two host buffers are reused across steps so a 100k-atom
// system does not reallocate its edge storage every timestep.
//
// The semantics mirror `src/molix/batch/neighbor.py` — the holder of the repo's
// Edge Convention — and its **recompute-every-step** contract: each `build`
// call produces a freshly sized `(ne, 2)` tensor. They deliberately do NOT
// mirror `src/molix/md/neighbors.py`, whose fixed-capacity live-range layout
// (dead padding past `num_edges`) is a different contract that no LAMMPS
// consumer asks for here.
//
// EDGE ORDER IS NOT PROMISED unconditionally. The default is the raw neighbour-
// list order; `(source, target)` order is produced only for an export that
// declares `requires_sorted_edges`. Anything downstream that needs an order
// must obtain it through that declaration rather than by assuming one.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"

namespace molnex::lammps_bridge {

/// Builds the model's `edge_index` from a LAMMPS neighbour list, each step.
class GraphBridge {
 public:
  /// \param cutoff                Model cutoff in angstrom, from the export
  ///                              meta. Stored squared; every candidate pair is
  ///                              refiltered against it.
  /// \param requires_sorted_edges The export's declaration of whether it needs
  ///                              rows in `(source, target)` order. Fixed at
  ///                              construction because it is a property of the
  ///                              model, not of a step; see `build`.
  GraphBridge(double cutoff, bool requires_sorted_edges);

  /// Collect and refilter the cutoff graph over the first `n_graph` atoms.
  ///
  /// Walks `view.ilist` over `inum + gnum` centres, and for each centre `i`
  /// every masked neighbour `view.neighbor(i, jj)`; drops neighbours at or past
  /// `n_graph`; keeps the rest only when `dx^2 + dy^2 + dz^2 < cutoff^2`; sorts
  /// the survivors lexicographically **only if** the contract asked for it.
  ///
  /// Edge Convention: column 0 is the **source**, the centre atom out of
  /// `ilist`; column 1 is the **target**, its neighbour — the same orientation
  /// as `src/molix/batch/neighbor.py`. No `edge_diff` is computed here: models
  /// recompute geometry from `pos`, so this layer only owes the column order.
  ///
  /// THE SOURCE DOMAIN IS `local u ghost`, not local alone, whenever the
  /// neighbour request carried `REQ_GHOST`. That is what makes the resulting
  /// edge set symmetrically closed over the ghost region, and symmetric closure
  /// is why ONE builder serves both aggregation directions: a
  /// target-aggregating model (MACE) needs every ghost->local edge at its first
  /// layer, a source-aggregating one (PiNet) needs every ghost->ghost edge by
  /// its second, and walking the ghost centres supplies both. No per-model
  /// branch exists here, or is needed.
  ///
  /// \param view     Borrowed LAMMPS atom arrays and neighbour spans.
  /// \param n_graph  Number of leading atoms that are graph nodes: `nlocal` for
  ///                 a `supports_pbc=false` export (ghosts excluded), `nall()`
  ///                 otherwise. Both the neighbour filter threshold and the
  ///                 upper bound of every `view.x` read.
  /// \returns Freshly allocated `(ne, 2)` int64 CPU tensor owning its data — it
  ///          never aliases the reused host buffers, so a caller may hold it
  ///          across the next `build`. `(0, 2)` when no pair survives.
  /// \throws std::runtime_error  If `n_graph` is outside `[0, view.nall()]`;
  ///                             `view.x` belongs to LAMMPS, so an unchecked
  ///                             overrun reads another object's memory.
  at::Tensor build(const LammpsView& view, int n_graph);

 private:
  double cutsq_;
  /// Whether `build` sorts. Read from the export's `requires_sorted_edges`;
  /// see the sort site in `build` for the reasoning.
  bool requires_sorted_edges_;
  // Reused across steps, exactly as the inlined loop this replaces did:
  // dropping the reuse would be an unmeasured performance regression smuggled
  // into a structural migration.
  std::vector<std::pair<int, int>> edge_pairs_;
  std::vector<int64_t> edge_flat_;
};

}  // namespace molnex::lammps_bridge
