// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The cutoff-graph contract: rebuild the neighbour graph from a cutoff every
// step, evaluate the model over it, scatter the forces back.
//
// This is the composition point of every component lammps-iface-02-view and
// -03-graph produced — `SpeciesMap`, `stage_nodes`, `GraphBridge`, `Exec`,
// `ForceScatter`, `pack_reverse_forces` / `unpack_reverse_forces` — and it adds
// no numerics of its own: what lives here is the ORDER those components run in
// and the handful of decisions that order needs (how many atoms are graph
// nodes, whether the pair style owes a reverse comm, which unit factor
// applies).
//
// It is a CONTRACT-FAMILY implementation, not a model implementation. PiNet,
// MACE and Allegro all export `contract = "cutoff-graph"` and all run through
// this one class; "no per-model C++" is not at risk, and would be the moment a
// class here were named after a network.
//
// NO LAMMPS HEADER, and the `Exec` arrives by CONSTRUCTOR INJECTION. Those two
// facts together are what let the entire layer-2 + layer-3 stack be exercised
// by ctest with no LAMMPS build, no CUDA and no `.pt2` package — the tests hand
// in a stub `Exec` and a synthetic `LammpsView`. Selecting the execution
// strategy internally would put a semantic branch back into the class and take
// that testability with it; the `Execution` -> `Exec` mapping is the dispatch's
// job (`registry.cpp`).

#ifndef MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H
#define MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "bridge/exec.h"
#include "bridge/force_scatter.h"
#include "bridge/graph_bridge.h"
#include "bridge/lammps_view.h"
#include "bridge/species_map.h"
#include "impls/contract_impl.h"
#include "lammps_contract.h"

namespace molnex::lammps {

/// The `cutoff-graph` contract, over an injected execution strategy.
///
/// Lifecycle, mirroring the `pair_style` / `pair_coeff` / `init` / `run` order
/// LAMMPS itself imposes: construct (capabilities, cutoff, unit factors and the
/// strategy are fixed), `coeff` (the type -> Z map arrives and is validated
/// against the export's species), `init` (this run's configuration is checked),
/// then `compute` per step.
///
/// Example:
/// \code
///   CutoffGraphImpl impl(contract, update->unit_style,
///                        std::make_unique<DynamicExec>(...));
///   impl.coeff({"1", "6", "8"}, atom->ntypes);
///   impl.init(LammpsRunEnv{...});
///   const EvalResult r = impl.compute(view);
/// \endcode
class CutoffGraphImpl : public ContractImpl {
 public:
  /// Fix everything this contract can know before the input script continues.
  ///
  /// \param contract           Validated export contract; its cutoff, species,
  ///                           `supports_pbc` / `supports_virial` flags and
  ///                           unit table are copied out here, so it need not
  ///                           outlive this object.
  /// \param lammps_unit_style  `update->unit_style`, the unit style the running
  ///                           simulation asked for. The one input the meta
  ///                           cannot supply: `units_for` needs it to turn
  ///                           model energies and forces into LAMMPS' own.
  /// \param exec               The execution strategy, taken by ownership.
  ///                           Injected rather than chosen internally — see the
  ///                           file comment.
  /// \throws std::runtime_error  If `exec` is null, or if the contract cannot
  ///                             serve `lammps_unit_style` (the message lists
  ///                             the styles it does serve), or if its
  ///                             `species` list is malformed.
  CutoffGraphImpl(const LammpsContract& contract,
                  const std::string& lammps_unit_style,
                  std::unique_ptr<molnex::lammps_bridge::Exec> exec);

  /// The neighbour request this contract needs LAMMPS to make.
  ///
  /// \returns `{full = true, ghost = <supports_pbc>, cutoff = <meta cutoff>,
  ///           comm_reverse = 3, comm_reverse_off = 3}`. `ghost` follows the
  ///           export's `supports_pbc`: a periodic contract needs ghost centres
  ///           in `ilist` to close its edge set, a local-only one has no ghosts
  ///           at all. See `NeighborNeeds::ghost` for why that is not the same
  ///           statement as the reverse-comm decision — the two answers now
  ///           point in opposite directions.
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
  /// Four predicates, all fail-closed: the single-domain rule, `special_bonds`,
  /// newton-pair for a periodic contract, and the fat-ghost receptive-field
  /// condition. None is ever downgraded to a warning — each of them, unmet,
  /// produces a run that completes and reports wrong physics.
  ///
  /// \param env  Snapshot of the LAMMPS globals.
  /// \throws std::runtime_error  If a non-periodic export is being decomposed
  ///                             over several ranks with `newton_pair` on; if a
  ///                             molecular system's `special_bonds`
  ///                             coefficients are not all 1.0 — LAMMPS deletes
  ///                             the scaled 1-2 / 1-3 / 1-4 pairs from the
  ///                             neighbour list, and a many-body model would
  ///                             then be evaluated on a graph missing its
  ///                             nearest neighbours; if a `supports_pbc` export
  ///                             is run with `newton_pair` off, which would
  ///                             leave every ghost force unreduced; or if the
  ///                             ghost cutoff is below `T * cutoff + skin`, in
  ///                             which case the message names
  ///                             `comm_modify cutoff` and the value to give it.
  void init(const LammpsRunEnv& env) override;

  /// Stage, build the graph, run the model, scatter the forces.
  ///
  /// \param view  Borrowed LAMMPS atom arrays and neighbour spans.
  /// \returns The partition energy and its per-atom decomposition (length
  ///          `nlocal`), the converted virial when this evaluation produced
  ///          one, and the reverse-comm decision. Forces are already in
  ///          `view.f`, for local and ghost alike; see `EvalResult`.
  /// \throws std::runtime_error  If `pair_coeff` has not run, if a
  ///                             `supports_pbc = false` export is handed ghost
  ///                             atoms, if a `supports_pbc` export reaches
  ///                             here without `init` having confirmed newton,
  ///                             or if any staged component refuses.
  EvalResult compute(const molnex::lammps_bridge::LammpsView& view) override;

  /// Forwards to `molnex::lammps_bridge::pack_reverse_forces`.
  int pack_reverse(const molnex::lammps_bridge::LammpsView& view, int n,
                   int first, double* buf) override;

  /// Forwards to `molnex::lammps_bridge::unpack_reverse_forces`.
  void unpack_reverse(const molnex::lammps_bridge::LammpsView& view, int n,
                      const int* list, const double* buf) override;

 private:
  /// Delegated-to constructor, so the unit factors are looked up exactly once
  /// and `energy_conv_` and `force_scatter_` cannot come from two lookups that
  /// could disagree.
  CutoffGraphImpl(const LammpsContract& contract, UnitConversion conv,
                  std::unique_ptr<molnex::lammps_bridge::Exec> exec);

  /// Model cutoff in angstrom; reported through `needs()` and squared inside
  /// `graph_bridge_`.
  double cutoff_;
  /// Whether ghosts are graph nodes: decides `n_graph`, the ghost neighbour
  /// request, and which run-configuration predicates apply.
  bool supports_pbc_;
  /// Whether the export declares a virial output at all.
  bool supports_virial_;
  /// T, the model's message-passing depth, from the export meta. The fat-ghost
  /// threshold is `T * cutoff_ + skin`; read once at construction because it is
  /// a property of the model, not of a run.
  int receptive_field_layers_;
  /// `force->newton_pair` AS SEEN BY `init`, or empty when `init` has not run.
  ///
  /// Optional rather than a plain int with a default, because "init never ran"
  /// and "init ran and saw newton off" are different faults and only the
  /// optional can tell them apart from a default value that happens to look
  /// safe. `compute` re-reads it to catch the first; `init` refuses the second
  /// outright.
  ///
  /// EFFICACY BOUNDARY, stated so the next reader knows what this does NOT
  /// buy: it is a snapshot taken by this object, so it catches an internal
  /// ordering fault — `compute` reached on a periodic contract whose newton
  /// precondition was never validated. It CANNOT catch LAMMPS changing newton
  /// after `init`; that case is covered by `init_style` re-running at every
  /// `run` setup, which re-enters `init` and re-checks.
  std::optional<int> newton_pair_;
  /// One model energy unit expressed in the active LAMMPS unit style.
  double energy_conv_;
  /// Atomic numbers the export was built for; every `pair_coeff` Z is checked
  /// against this list.
  std::vector<int> species_;
  /// Empty until `coeff` runs — a `pair_style` without its `pair_coeff` has no
  /// type -> Z table, and there is no half-built one to evaluate from.
  std::optional<molnex::lammps_bridge::SpeciesMap> species_map_;
  /// Owned by value: fixed at construction from the cutoff, reused every step.
  molnex::lammps_bridge::GraphBridge graph_bridge_;
  /// Owned by value: fixed at construction from the force factor.
  molnex::lammps_bridge::ForceScatter force_scatter_;
  /// The injected strategy; never selected here.
  std::unique_ptr<molnex::lammps_bridge::Exec> exec_;
};

}  // namespace molnex::lammps

#endif  // MOLNEX_LAMMPS_IMPLS_CUTOFF_GRAPH_IMPL_H
