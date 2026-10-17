// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The cutoff-graph contract: rebuild the neighbour graph from a cutoff every
// step, evaluate the model over it, scatter the forces back.
//
// This is the composition point of the bridge components — `SpeciesMap`,
// `stage_nodes`, `GraphBridge`, `ForceScatter`, `pack_reverse_forces` /
// `unpack_reverse_forces` — and the runtime's `AtomEnergyExec`. It adds no
// numerics of its own: what lives here is the ORDER those components run in
// and the handful of decisions that order needs (how many atoms are graph
// nodes, whether the pair style owes a reverse comm, which unit factor
// applies).
//
// One implementation, every model. Any package that follows the atom-energy
// convention runs through this class; nothing here is named after a network
// or a package format.
//
// NO LAMMPS HEADER, and the execution arrives by constructor injection. Those
// two facts let ctest drive this class over a stub `ModelArtifact` and a
// synthetic `LammpsView`. The pair style opens the artifact; this class does
// not.

#ifndef MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H
#define MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H

#include <optional>
#include <string>
#include <vector>

#include "bridge/force_scatter.h"
#include "bridge/graph_bridge.h"
#include "bridge/lammps_view.h"
#include "bridge/species_map.h"
#include "impls/contract_impl.h"
#include "molnex/runtime/metadata.h"
#include "molnex/runtime/atom_energy_exec.h"

namespace molnex::lammps {

/// How periodic neighbours reach the model.
///
/// * `kGhost` — every ghost within `T * cutoff + skin` is a graph node, the
///   mask selects the owned atoms, shifts are zero, ghost forces go home
///   through the integrator's reverse comm, the virial is LAMMPS' `fdotr`.
///   Works on any number of ranks; costs the whole ghost shell on every step.
/// * `kFold` — only the `nlocal` owned atoms are graph nodes; a ghost
///   neighbour is folded onto its owner with the lattice translation as the
///   pair's shift, forces land on owned atoms only, the virial is the model's.
///   Needs every ghost to be an image of an owned atom: one rank.
enum class GraphMode { kGhost, kFold };

/// The `cutoff-graph` contract, over an injected execution strategy.
///
/// Lifecycle, mirroring the `pair_style` / `pair_coeff` / `init` / `run` order
/// LAMMPS itself imposes: construct (capabilities, cutoff, unit factor and the
/// execution are fixed), `coeff` (the type -> Z map arrives and is validated
/// against the export's species), `init` (this run's configuration is checked),
/// then `compute` per step.
///
/// Example:
/// \code
///   CutoffGraphImpl impl(metadata, energy_factor,
///                        AtomEnergyExec(*artifact, metadata.dtype),
///                        GraphMode::kFold);
///   impl.coeff({"1", "6", "8"}, atom->ntypes);
///   impl.init(LammpsRunEnv{...});
///   const EvalResult r = impl.compute(view);
/// \endcode
class CutoffGraphImpl : public ContractImpl {
 public:
  /// Fix everything this contract can know before the input script continues.
  ///
  /// \param metadata       The model's metadata; its cutoff, depth and species
  ///                       are copied out here, so it need not outlive this.
  /// \param energy_factor  One model energy unit in the active LAMMPS unit
  ///                       style. The facade computes it (it is the one layer
  ///                       that may call LAMMPS' own conversion table). Forces
  ///                       share it because the model's length unit is
  ///                       checked to be angstrom, LAMMPS' length in both
  ///                       `real` and `metal`.
  /// \param exec           How the model runs; its artifact must outlive this.
  /// \param mode           Ghost staging or folded images (`GraphMode`); the
  ///                       facade picks it at `pair_style`, and it fixes the
  ///                       neighbour request, so it cannot change afterwards.
  /// \throws std::runtime_error  If `metadata.length_unit` is not angstrom.
  CutoffGraphImpl(const molnex::runtime::Metadata& metadata,
                  double energy_factor, molnex::runtime::AtomEnergyExec exec,
                  GraphMode mode);

  /// The neighbour request this contract needs LAMMPS to make.
  ///
  /// \returns `{full = true, ghost, cutoff, comm_reverse = 3,
  ///           comm_reverse_off = 3}`. In the ghost mode ghost centres are
  ///           requested so the edge set closes over the ghost region; that
  ///           request is not the reverse-comm decision —
  ///           `needs_pair_reverse_comm` stays false, and the integrator
  ///           reduces ghost forces. The folded mode walks owned centres only
  ///           and asks for none.
  NeighborNeeds needs() const override;

  /// Read one atomic number per LAMMPS atom type and build the species map.
  ///
  /// Parsing is FULL-CONSUMPTION: a token is accepted only when
  /// `std::from_chars` both reports no error AND stops at the token's end.
  /// `std::stoi` would accept `"6abc"` as 6 and turn a typo into a different
  /// element, evaluated with no diagnostic anywhere.
  ///
  /// \param tokens  One token per atom type, in type order (type 1 first).
  /// \param ntypes  `atom->ntypes`; `tokens.size()` must equal it exactly.
  /// \throws std::runtime_error  If the token count disagrees with `ntypes`, if
  ///                             any token is not wholly an integer (a trailing
  ///                             character, an empty token and a value outside
  ///                             `int` are all refused, quoting the token), or
  ///                             if any Z is one the export was not built for.
  ///                             Nothing is stored on any failing path.
  void coeff(const std::vector<std::string>& tokens, int ntypes) override;

  /// Check this run against the export, and snapshot what `compute` will need.
  ///
  /// All predicates fail-closed: `special_bonds` on a molecular system (both
  /// modes); in the ghost mode `newton on` and the fat-ghost receptive-field
  /// condition; in the folded mode a single rank. None is downgraded to a
  /// warning.
  ///
  /// \param env  Snapshot of the LAMMPS globals.
  /// \throws std::runtime_error  If a molecular system's `special_bonds`
  ///                             coefficients are not all 1.0; in the ghost
  ///                             mode if `newton` is off (ghost forces would be
  ///                             left unreduced) or the ghost cutoff is below
  ///                             `T * cutoff + skin` (the message names
  ///                             `comm_modify cutoff` and the value to give
  ///                             it); in the folded mode if more than one rank
  ///                             runs (a ghost would then be another rank's
  ///                             atom, with no owner here to fold onto).
  void init(const LammpsRunEnv& env) override;

  /// Stage, build the graph, run the model, scatter the forces.
  ///
  /// \param view  Borrowed LAMMPS atom arrays and neighbour spans.
  /// \returns The partition energy and its per-atom decomposition (length
  ///          `nlocal`), the reverse-comm decision (always false), and in the
  ///          folded mode the model's virial. Forces are already in `view.f`:
  ///          for local and ghost alike in the ghost mode, for owned atoms
  ///          only in the folded mode.
  /// \throws std::runtime_error  If `pair_coeff` has not run, if `compute`
  ///                             is reached without `init` having confirmed
  ///                             newton, or if any staged component refuses.
  EvalResult compute(const molnex::lammps_bridge::LammpsView& view) override;

  /// Forwards to `molnex::lammps_bridge::pack_reverse_forces`.
  int pack_reverse(const molnex::lammps_bridge::LammpsView& view, int n,
                   int first, double* buf) override;

  /// Forwards to `molnex::lammps_bridge::unpack_reverse_forces`.
  void unpack_reverse(const molnex::lammps_bridge::LammpsView& view, int n,
                      const int* list, const double* buf) override;

 private:
  /// `compute` in the folded mode: owned atoms, shifted pairs, model virial.
  EvalResult compute_folded(const molnex::lammps_bridge::LammpsView& view);

  /// Model cutoff in angstrom; reported through `needs()` and squared inside
  /// `graph_bridge_`.
  double cutoff_;
  /// T, the model's message-passing depth, from the metadata. The fat-ghost
  /// threshold is `T * cutoff_ + skin`; read once at construction because it is
  /// a property of the model, not of a run.
  int receptive_field_layers_;
  /// Fixed at construction; see `GraphMode`.
  GraphMode mode_;
  /// `force->newton_pair` AS SEEN BY `init`, or empty when `init` has not run.
  ///
  /// Optional rather than a plain int with a default, because "init never ran"
  /// and "init ran and saw newton off" are different faults and only the
  /// optional can tell them apart from a default value that happens to look
  /// safe. `compute` re-reads it to catch the first; `init` refuses the second
  /// outright.
  ///
  /// The snapshot catches an internal ordering fault: `compute` reached before
  /// `init` confirmed newton. It does not catch LAMMPS changing newton after
  /// `init`; `init_style` re-runs at every `run` setup and re-enters `init`.
  std::optional<int> newton_pair_;
  /// One model energy unit expressed in the active LAMMPS unit style.
  double energy_conv_;
  /// Atomic numbers the export was built for; every `pair_coeff` Z is checked
  /// against this list.
  std::vector<int> species_;
  /// Empty until `coeff` runs — a `pair_style` without its `pair_coeff` has no
  /// type -> Z table, and there is no half-built one to evaluate from.
  std::optional<molnex::lammps_bridge::SpeciesMap> species_map_;
  /// Owned by value: fixed at construction from the cutoff, unsorted, reused
  /// every step.
  molnex::lammps_bridge::GraphBridge graph_bridge_;
  /// Owned by value: fixed at construction from the energy factor.
  molnex::lammps_bridge::ForceScatter force_scatter_;
  /// The injected execution; never selected here.
  molnex::runtime::AtomEnergyExec exec_;
};

}  // namespace molnex::lammps

#endif  // MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H
