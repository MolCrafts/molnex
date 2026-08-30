// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `species_map.h` for the design notes.

#include "bridge/species_map.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace molnex::lammps_bridge {
namespace {

/// Render every valid atomic number for a "valid species: ..." message.
///
/// Listing the whole legal set is what turns "this Z is wrong" into an
/// actionable diagnostic — the operator sees at once whether the fix is the
/// `pair_coeff` line or a re-export.
///
/// \param species  Atomic numbers the model was exported for.
/// \returns Comma-separated list, e.g. `"1, 6, 8"`.
std::string join(const std::vector<int>& species) {
  std::string joined;
  for (const int z : species) {
    if (!joined.empty()) joined += ", ";
    joined += std::to_string(z);
  }
  return joined;
}

}  // namespace

SpeciesMap::SpeciesMap(std::vector<int> z_by_type,
                       const std::vector<int>& meta_species)
    : z_by_type_(std::move(z_by_type)) {
  if (z_by_type_.empty()) {
    throw std::runtime_error(
        "SpeciesMap: the type table is empty; a pair style with no atom types "
        "has nothing to map (pass one atomic number per LAMMPS type, in type "
        "order)");
  }
  for (std::size_t i = 0; i < z_by_type_.size(); ++i) {
    const int z = z_by_type_[i];
    const std::string type = std::to_string(i + 1);
    if (z <= 0) {
      throw std::runtime_error(
          "SpeciesMap: LAMMPS type " + type + " maps to Z " +
          std::to_string(z) +
          ", which is not an atomic number (every Z must be > 0); valid "
          "species: " +
          join(meta_species));
    }
    if (std::find(meta_species.begin(), meta_species.end(), z) ==
        meta_species.end()) {
      throw std::runtime_error(
          "SpeciesMap: LAMMPS type " + type + " maps to Z " +
          std::to_string(z) +
          ", which the model was not exported for; valid species: " +
          join(meta_species));
    }
  }
}

int SpeciesMap::z_of(int lammps_type) const {
  if (lammps_type < 1 || lammps_type > ntypes()) {
    throw std::runtime_error(
        "SpeciesMap: LAMMPS type " + std::to_string(lammps_type) +
        " is out of range; types are 1-based and this map covers 1.." +
        std::to_string(ntypes()));
  }
  return z_by_type_[static_cast<std::size_t>(lammps_type) - 1];
}

int SpeciesMap::ntypes() const { return static_cast<int>(z_by_type_.size()); }

}  // namespace molnex::lammps_bridge
