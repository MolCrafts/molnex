// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `graph_bridge.h` for the design notes.

#include "bridge/graph_bridge.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

GraphBridge::GraphBridge(double cutoff, bool requires_sorted_edges)
    : cutsq_(cutoff * cutoff), requires_sorted_edges_(requires_sorted_edges) {}

at::Tensor GraphBridge::build(const LammpsView& view, int n_graph) {
  if (n_graph < 0 || n_graph > view.nall()) {
    throw std::runtime_error(
        "GraphBridge: n_graph " + std::to_string(n_graph) +
        " is outside [0, nall] with nall = " + std::to_string(view.nall()) +
        " (nlocal " + std::to_string(view.nlocal) + " + nghost " +
        std::to_string(view.nghost) +
        "); n_graph is both the neighbour filter "
        "threshold and the upper bound of every position read, so a value past "
        "nall would read past the end of the LAMMPS atom arrays");
  }

  edge_pairs_.clear();
  const int centres = view.inum + view.gnum;
  edge_pairs_.reserve(static_cast<std::size_t>(centres) * 16);
  for (int ii = 0; ii < centres; ++ii) {
    // THE SOURCE DOMAIN IS local u ghost. LAMMPS lays ilist out as `inum` local
    // centres followed by `gnum` ghost ones (the latter present only when the
    // request carried REQ_GHOST, so `gnum` is 0 on the local-only path and this
    // loop reduces to the original one). Walking both segments is what closes
    // the edge set symmetrically over the ghost region, which is the condition
    // that makes this builder correct for both aggregation directions at once —
    // ghost->local edges for a target-aggregating model's first layer,
    // ghost->ghost for a source-aggregating model's second.
    //
    // The centre index itself is trusted to be in [0, nall): it is the
    // neighbour-list builder's invariant, and re-verifying it in the hot loop
    // would only buy psychological comfort. Every position read below IS
    // bounded — by the n_graph check above for targets, and by that same
    // invariant for centres. This comment exists so the next reader knows the
    // invariant is depended upon rather than overlooked.
    const int i = view.ilist[ii];
    // n_graph bounds BOTH endpoints, not just the target. A ghost centre cannot
    // occur on the local-only path (no REQ_GHOST means gnum is 0), but if it
    // ever did, an edge whose SOURCE is past n_graph would index past the
    // staged node tensors inside the model — so the threshold is applied where
    // it is claimed to apply.
    if (i >= n_graph) continue;
    const double xi = view.x[i][0];
    const double yi = view.x[i][1];
    const double zi = view.x[i][2];
    const int jn = view.numneigh[i];
    for (int jj = 0; jj < jn; ++jj) {
      const int j = view.neighbor(i, jj);
      if (j >= n_graph) continue;
      const double dx = view.x[j][0] - xi;
      const double dy = view.x[j][1] - yi;
      const double dz = view.x[j][2] - zi;
      // The cutoff refilter is REQUIRED, not redundant: LAMMPS builds its
      // neighbour list at cutoff + skin, so the list always carries pairs
      // beyond the model cutoff. Dropping this test silently changes the
      // energy, with no diagnostic anywhere.
      if (dx * dx + dy * dy + dz * dz < cutsq_) edge_pairs_.emplace_back(i, j);
    }
  }

  // SORTING IS A CONTRACT DECLARATION, NOT A UNIVERSAL PROMISE.
  //
  // Every aggregation in this tree is scatter-class (`index_add_` /
  // `scatter_add_`), and those are permutation-invariant in exact arithmetic;
  // in floating point the reassociation error is about (n-1)u, and CUDA's own
  // `index_add_` is non-deterministic anyway, so sorting buys no
  // reproducibility either. Paying for a std::sort every step to satisfy nobody
  // is the cost this flag removes: the default path builds in raw
  // neighbour-list order, exactly as pair_nequip_allegro does.
  //
  // The code stays because the requirement is real for a different class of
  // model: segmented / CSR aggregation (cuEquivariance's indexed operators)
  // needs rows grouped by index, and is simply WRONG on unordered edges — with
  // the arithmetic still completing. Such a model exports
  // `requires_sorted_edges = true` and gets the order it needs; a model that
  // does not declare it can never be fed sorted-order-dependent code by
  // accident, because there is no path here that produces one without the
  // other.
  //
  // std::pair's lexicographic ordering IS the (source, target) sort.
  if (requires_sorted_edges_) {
    std::sort(edge_pairs_.begin(), edge_pairs_.end());
  }

  const auto opt_l = at::TensorOptions().dtype(at::kLong);
  const int64_t ne = static_cast<int64_t>(edge_pairs_.size());
  if (ne == 0) {
    // An empty std::vector may hold a null data pointer, and handing that to
    // at::from_blob is undefined even for a zero-element shape. Allocating the
    // empty tensor directly is the whole fix: same (0, 2) int64 result, no null
    // blob.
    return at::empty({0, 2}, opt_l);
  }

  edge_flat_.resize(static_cast<std::size_t>(2 * ne));
  for (int64_t k = 0; k < ne; ++k) {
    const auto e = edge_pairs_[static_cast<std::size_t>(k)];
    edge_flat_[static_cast<std::size_t>(2 * k)] = e.first;
    edge_flat_[static_cast<std::size_t>(2 * k + 1)] = e.second;
  }
  // clone(): the returned tensor owns its data, so the caller may keep it
  // across the next build, which overwrites edge_flat_ in place.
  return at::from_blob(edge_flat_.data(), {ne, 2}, opt_l).clone();
}

}  // namespace molnex::lammps_bridge
