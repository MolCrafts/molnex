// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `force_scatter.h` for the design notes.

#include "bridge/force_scatter.h"

#include <stdexcept>
#include <string>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {

ForceScatter::ForceScatter(double force_conv) : force_conv_(force_conv) {}

void ForceScatter::scatter(const LammpsView& view, const at::Tensor& forces,
                           int n_graph) const {
  if (n_graph < 0 || n_graph > view.nall()) {
    throw std::runtime_error(
        "ForceScatter: n_graph " + std::to_string(n_graph) +
        " is outside [0, nall] with nall = " + std::to_string(view.nall()) +
        " (nlocal " + std::to_string(view.nlocal) + " + nghost " +
        std::to_string(view.nghost) +
        "); scattering beyond nall would write "
        "past the end of the LAMMPS force array");
  }
  if (forces.size(0) != n_graph) {
    throw std::runtime_error("ForceScatter: model returned forces for " +
                             std::to_string(forces.size(0)) +
                             " atoms, expected " + std::to_string(n_graph));
  }

  auto fa = forces.accessor<double, 2>();
  for (int i = 0; i < n_graph; ++i) {
    view.f[i][0] += force_conv_ * fa[i][0];
    view.f[i][1] += force_conv_ * fa[i][1];
    view.f[i][2] += force_conv_ * fa[i][2];
  }
}

}  // namespace molnex::lammps_bridge
