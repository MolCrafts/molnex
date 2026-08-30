// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// LAMMPS type -> atomic number mapping, validated at construction.
//
// The table this replaces (`PairMolnex::type2z`) was filled straight from
// `pair_coeff` with no validation whatsoever: an element the model was never
// exported for reached the network as an embedding index it has no row for,
// and the run produced a confident, wrong energy with no diagnostic. Validating
// in the constructor is what closes that path — there is no half-constructed
// SpeciesMap to feed a network from.
//
// Deliberately schema-blind: the set of legal atomic numbers arrives as a plain
// `std::vector<int>`, not as a `LammpsContract`. This component does identity
// mapping; binding it to the meta.json schema would make it move every time the
// export format grows a field. No file under `bridge/` includes meta_schema.h
// or lammps_contract.h.

#pragma once

#include <vector>

namespace molnex::lammps_bridge {

/// Validated mapping from a 1-based LAMMPS atom type to an atomic number.
///
/// The table is **0-based and dense**: `z_by_type[t - 1]` is the atomic number
/// of LAMMPS type `t`, and its length *is* `ntypes()`. There is no placeholder
/// slot at index 0 (which is what made the old table's length ambiguous to
/// read) and no second `ntypes` argument that could disagree with it.
class SpeciesMap {
 public:
  /// Build the map and validate every entry against the model's species.
  ///
  /// \param z_by_type    0-based dense table of atomic numbers, one per LAMMPS
  ///                     type, length `ntypes`. Taken by value and moved.
  /// \param meta_species Every atomic number the model was exported for, e.g.
  ///                     from `LammpsContract::species()`.
  /// \throws std::runtime_error  If `z_by_type` is empty (a pair style with no
  ///                             atom types has nothing to map), if any entry
  ///                             is not a positive atomic number, or if any
  ///                             entry is absent from `meta_species`. The
  ///                             message names the offending LAMMPS type, its
  ///                             Z, and every valid species.
  SpeciesMap(std::vector<int> z_by_type, const std::vector<int>& meta_species);

  /// Atomic number of a LAMMPS atom type.
  ///
  /// \param lammps_type  1-based LAMMPS type, as found in `atom->type`.
  /// \returns The atomic number registered for that type.
  /// \throws std::runtime_error  If `lammps_type` is outside `[1, ntypes()]`.
  ///                             LAMMPS types are 1-based, so 0 is not "the
  ///                             first type" — reading it would walk off the
  ///                             front of the table.
  int z_of(int lammps_type) const;

  /// Number of LAMMPS atom types the map covers.
  ///
  /// \returns The table length, which is the largest valid `lammps_type`.
  int ntypes() const;

 private:
  std::vector<int> z_by_type_;
};

}  // namespace molnex::lammps_bridge
