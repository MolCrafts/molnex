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

GraphBridge::GraphBridge(double cutoff) : cutsq_(cutoff * cutoff) {}

at::Tensor GraphBridge::build(const LammpsView& view, int n_graph) {
  if (n_graph < 0 || n_graph > view.nall()) {
    throw std::runtime_error(
        "GraphBridge: n_graph " + std::to_string(n_graph) +
        " is outside [0, nall] with nall = " + std::to_string(view.nall()) +
        " (nlocal " + std::to_string(view.nlocal) + " + nghost " +
        std::to_string(view.nghost) + "); n_graph is both the neighbour filter "
        "threshold and the upper bound of every position read, so a value past "
        "nall would read past the end of the LAMMPS atom arrays");
  }

  edge_pairs_.clear();
  edge_pairs_.reserve(static_cast<std::size_t>(view.inum) * 16);
  for (int ii = 0; ii < view.inum; ++ii) {
    // The centre index is trusted to be local: under REQ_FULL, LAMMPS
    // guarantees ilist only lists atoms in [0, nlocal), so edge_index[:,0]
    // satisfies source in local by construction. No runtime check here — it is
    // the neighbour list builder's invariant, and re-verifying it in the hot
    // loop would only buy psychological comfort. This comment exists so the
    // next reader knows the invariant is depended upon rather than overlooked.
    // (lammps-iface-05-pbc rewrites this sentence when ghost traversal widens
    // the source domain to local u ghost, as its symmetric closure requires.)
    const int i = view.ilist[ii];
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

  // std::pair's lexicographic ordering IS the (source, target) sort molnex
  // aggregation requires — the same order the reference data pipeline's
  // nonzero() yields, and one the raw neighbour-list order does not have.
  // lammps-iface-03-graph keeps the sort unconditionally; lammps-iface-05-pbc
  // demotes it to the requires_sorted_edges contract flag (default off, set by
  // models that need segmented / CSR aggregation), at which point this code
  // stays but stops running every step. Do not anticipate that here.
  std::sort(edge_pairs_.begin(), edge_pairs_.end());

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
