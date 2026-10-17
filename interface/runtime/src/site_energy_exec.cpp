// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See site_energy_exec.h.

#include "molnex/runtime/site_energy_exec.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

namespace molnex::runtime {

SiteEnergyExec::SiteEnergyExec(ModelArtifact& artifact, at::ScalarType dtype)
    : artifact_(&artifact), dtype_(dtype) {}

SiteEnergyOutputs SiteEnergyExec::run(const at::Tensor& Z,
                                      const at::Tensor& pos_f64,
                                      const at::Tensor& pair_index,
                                      const at::Tensor& shifts_f64,
                                      const at::Tensor& local_mask) {
  // Positions and shifts arrive in fp64 so the coordinates are exactly the
  // engine's; the one narrowing to the model dtype is here. `local_mask`
  // stays bool: the multiplication against it is inside the packaged graph.
  const at::Device device = artifact_->device();
  const std::vector<at::Tensor> raw = artifact_->run({
      Z.to(device),
      pos_f64.to(dtype_).to(device),
      pair_index.to(device),
      shifts_f64.to(dtype_).to(device),
      local_mask.to(device),
  });

  if (raw.size() != 4) {
    throw std::runtime_error(
        "SiteEnergyExec: the package returned " + std::to_string(raw.size()) +
        " output(s); exactly 4 are required (energy, forces, site_energy, "
        "virial) — a package exported before metadata schema 2 returns 3");
  }
  // Each output is freshly allocated per call; moving it to CPU float64 is the
  // copy the caller owns.
  SiteEnergyOutputs out;
  out.energy = raw[0].to(at::kCPU, at::kDouble);
  out.forces = raw[1].to(at::kCPU, at::kDouble).contiguous();
  out.site_energy = raw[2].to(at::kCPU, at::kDouble).contiguous();
  out.virial = raw[3].to(at::kCPU, at::kDouble).contiguous();

  // One entry per staged atom: an engine sums a slice of these and scatters
  // the forces by row, so a length that differs from what was staged would
  // sum the wrong sites or write past the atoms it owns.
  const int64_t n = Z.size(0);
  if (out.site_energy.dim() != 1 || out.site_energy.size(0) != n) {
    throw std::runtime_error(
        "SiteEnergyExec: site_energy has " +
        std::to_string(out.site_energy.numel()) + " entries for " +
        std::to_string(n) +
        " staged atom(s); it must hold exactly one per atom");
  }
  if (out.forces.dim() != 2 || out.forces.size(0) != n ||
      out.forces.size(1) != 3) {
    throw std::runtime_error(
        "SiteEnergyExec: forces have " + std::to_string(out.forces.numel()) +
        " entries; they must be (" + std::to_string(n) + ", 3)");
  }
  if (out.virial.dim() != 2 || out.virial.size(0) != 3 ||
      out.virial.size(1) != 3) {
    throw std::runtime_error("SiteEnergyExec: virial has " +
                             std::to_string(out.virial.numel()) +
                             " entries; it must be (3, 3)");
  }
  return out;
}

}  // namespace molnex::runtime
