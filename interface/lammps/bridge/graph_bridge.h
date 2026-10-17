// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The LAMMPS neighbour list as a molnex `pair_index`.
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
// (dead padding past `num_pairs`) is a different contract that no LAMMPS
// consumer asks for here.
//
// EDGE ORDER IS THE NEIGHBOUR-LIST ORDER. A model that needs rows sorted by
// (source, target) sorts inside its own forward, which is part of the
// exported package. This builder does not sort.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"

namespace molnex::lammps_bridge {

/// A cutoff graph over the owned atoms only, periodic images folded onto them.
struct FoldedGraph {
  /// Source-target pairs `(E, 2)` int64, both columns in `[0, nlocal)`.
  at::Tensor pair_index;
  /// Lattice translation of each pair's target image `(E, 3)` float64, in
  /// angstrom: `x[image] - x[owner]`, so that
  /// `x[target] - x[source] + shift` is the displacement LAMMPS measured.
  at::Tensor shifts;
};

/// Builds the model's `pair_index` from a LAMMPS neighbour list, each step.
class GraphBridge {
 public:
  /// \param cutoff  Model cutoff in angstrom. Stored squared; every candidate
  ///                pair is refiltered against it.
  explicit GraphBridge(double cutoff);

  /// Collect and refilter the cutoff graph over the first `n_graph` atoms.
  ///
  /// Walks `view.ilist` over `inum + gnum` centres, and for each centre `i`
  /// every masked neighbour `view.neighbor(i, jj)`; drops neighbours at or past
  /// `n_graph`; keeps the rest only when `dx^2 + dy^2 + dz^2 < cutoff^2`.
  /// Survivors stay in neighbour-list order.
  ///
  /// Edge Convention: column 0 is the **source**, the centre atom out of
  /// `ilist`; column 1 is the **target**, its neighbour — the same orientation
  /// as `src/molix/batch/neighbor.py`. No `pair_diff` is computed here: models
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
  /// \param n_graph  Number of leading atoms that are graph nodes. The pair
  ///                 style passes `nall()` (locals and ghosts). Both the
  ///                 neighbour filter threshold and the upper bound of every
  ///                 `view.x` read.
  /// \returns Freshly allocated `(ne, 2)` int64 CPU tensor owning its data — it
  ///          never aliases the reused host buffers, so a caller may hold it
  ///          across the next `build`. `(0, 2)` when no pair survives.
  /// \throws std::runtime_error  If `n_graph` is outside `[0, view.nall()]`;
  ///                             `view.x` belongs to LAMMPS, so an unchecked
  ///                             overrun reads another object's memory.
  at::Tensor build(const LammpsView& view, int n_graph);

  /// Collect the cutoff graph over the owned atoms, folding ghosts.
  ///
  /// Walks the `inum` LOCAL centres only, and for each centre `i` every masked
  /// neighbour `j` within the cutoff, local or ghost. The pair becomes
  /// `(i, view.owner[j])` with shift `x[j] - x[owner[j]]` — for a local `j`
  /// the owner is `j` itself and the shift is zero. Survivors stay in
  /// neighbour-list order.
  ///
  /// Every pair a model needs is here exactly once: the full list holds each
  /// directed pair from both of its owned endpoints, and an image of `i` that
  /// is its own neighbour (a box under twice the cutoff) is the pair `(i, i)`
  /// with a non-zero shift. The shift is the difference of two LAMMPS
  /// coordinates, so it is whatever lattice translation LAMMPS applied when it
  /// made the image — orthogonal or triclinic alike, with nothing reconstructed
  /// from the box here.
  ///
  /// \param view  Borrowed LAMMPS atom arrays and neighbour spans; `owner`
  ///              must be set whenever there are ghosts.
  /// \returns Fresh tensors owning their data (`(0, 2)` / `(0, 3)` when no
  ///          pair survives).
  /// \throws std::runtime_error  If a ghost neighbour is met with no `owner`
  ///                             table, or an owner is not an owned atom.
  ///                             Both mean this rank does not own every atom,
  ///                             which is the folded mode's precondition.
  FoldedGraph build_folded(const LammpsView& view);

 private:
  double cutsq_;
  // Reused across steps, exactly as the inlined loop this replaces did:
  // dropping the reuse would be an unmeasured performance regression smuggled
  // into a structural migration.
  std::vector<std::pair<int, int>> edge_pairs_;
  std::vector<int64_t> edge_flat_;
  std::vector<double> shift_flat_;
};

}  // namespace molnex::lammps_bridge
