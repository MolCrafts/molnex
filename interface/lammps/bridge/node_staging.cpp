// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `node_staging.h` for the design notes.

#include "bridge/node_staging.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

NodeTensors stage_nodes(const LammpsView& view, const SpeciesMap& species,
                        int n_graph) {
  if (n_graph < 0 || n_graph > view.nall()) {
    throw std::runtime_error(
        "stage_nodes: n_graph " + std::to_string(n_graph) +
        " is outside [0, nall] with nall = " + std::to_string(view.nall()) +
        " (nlocal " + std::to_string(view.nlocal) + " + nghost " +
        std::to_string(view.nghost) +
        "); staging beyond nall would read past "
        "the end of the LAMMPS atom arrays");
  }

  NodeTensors nodes{
      at::empty({n_graph}, at::TensorOptions().dtype(at::kLong)),
      at::empty({n_graph, 3}, at::TensorOptions().dtype(at::kDouble)),
  };
  auto Za = nodes.Z.accessor<int64_t, 1>();
  auto pa = nodes.pos.accessor<double, 2>();
  for (int i = 0; i < n_graph; ++i) {
    Za[i] = species.z_of(view.type[i]);
    pa[i][0] = view.x[i][0];
    pa[i][1] = view.x[i][1];
    pa[i][2] = view.x[i][2];
  }
  return nodes;
}

}  // namespace molnex::lammps_bridge
