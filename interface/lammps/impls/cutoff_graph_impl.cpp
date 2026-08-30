// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `cutoff_graph_impl.h` for the design notes (composition point, contract
// family rather than model, injected Exec).
//
// Every diagnostic below keeps the wording the pre-refactor pair_molnex.cpp
// used for the same condition — that text is what a LAMMPS user greps for — and
// only the prefix changes, exactly as the bridge components did when they were
// migrated. The one message with no ancestor is the special_bonds refusal,
// which is a defect this spec fixes rather than relocates.

#include "impls/cutoff_graph_impl.h"

#include <charconv>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/comm_reverse.h"
#include "bridge/node_staging.h"

namespace molnex::lammps {
namespace {

using molnex::lammps_bridge::LammpsView;

/// Doubles the reverse direction carries per atom: fx, fy, fz.
constexpr int kReverseCommSize = 3;

/// The `special_bonds` coefficient every 1-2 / 1-3 / 1-4 pair must carry for
/// the neighbour list to remain complete.
constexpr double kSpecialKeepsPair = 1.0;

/// Index range of the special-bond coefficients: `[1, 3]`, entry 0 being unused
/// in LAMMPS' own arrays.
constexpr int kFirstSpecial = 1;
constexpr int kLastSpecial = 3;

/// Parse one `pair_coeff` token as an atomic number, consuming all of it.
///
/// The strictness is the point, and it is why this is not `std::stoi`: that
/// function reads `"6abc"` as 6 and hands back an element the user never wrote,
/// which then reaches the model as a valid-looking embedding index. Acceptance
/// requires BOTH a clean `errc` and an end pointer at the token's end, so a
/// trailing character, an empty token and a value outside `int` are all
/// refused.
///
/// \param token        The token as the input script wrote it.
/// \param lammps_type  1-based LAMMPS type it was given for, for the message.
/// \returns The parsed atomic number (its plausibility as a Z is `SpeciesMap`'s
///          judgement, not this function's).
/// \throws std::runtime_error  If the whole token is not an integer; the
///                             message quotes the original token.
int parse_atomic_number(const std::string& token, int lammps_type) {
  int value = 0;
  const char* begin = token.data();
  const char* end = begin + token.size();
  const std::from_chars_result parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc() || parsed.ptr != end) {
    throw std::runtime_error(
        "pair_coeff for molnex: '" + token + "' (atom type " +
        std::to_string(lammps_type) +
        ") is not a whole atomic number; every token after '* *' must parse "
        "completely as an integer — a trailing character, an empty token or a "
        "value outside int is refused rather than silently read as some other "
        "element");
  }
  return value;
}

}  // namespace

CutoffGraphImpl::CutoffGraphImpl(
    const LammpsContract& contract, const std::string& lammps_unit_style,
    std::unique_ptr<molnex::lammps_bridge::Exec> exec)
    // units_for() is called exactly once, here, and its result is handed to the
    // delegated constructor: energy and force must not come from two separate
    // lookups that a later edit could let diverge.
    : CutoffGraphImpl(contract, contract.units_for(lammps_unit_style),
                      std::move(exec)) {}

CutoffGraphImpl::CutoffGraphImpl(
    const LammpsContract& contract, UnitConversion conv,
    std::unique_ptr<molnex::lammps_bridge::Exec> exec)
    : cutoff_(contract.cutoff()),
      supports_pbc_(contract.supports_pbc()),
      supports_virial_(contract.supports_virial()),
      // Read at CONSTRUCTION, so a meta that never declared T is refused while
      // the `pair_style` line is still on screen rather than at the first run.
      receptive_field_layers_(contract.receptive_field_layers()),
      energy_conv_(conv.energy),
      // Read whole and kept: `pair_coeff` runs after `pair_style` and needs the
      // export's species list to reject a Z the model was never built for.
      species_(contract.species()),
      // The sort is the export's declaration, fixed here for the same reason:
      // it is a property of the model, and re-reading it per step would invite
      // a per-step branch on a constant.
      graph_bridge_(cutoff_, contract.requires_sorted_edges()),
      // The force factor lives here and nowhere else. It equals `conv.energy`
      // only because `units_for` asserted the model measures length in
      // angstrom; that assertion is the contract's, not restated here.
      force_scatter_(conv.force),
      exec_(std::move(exec)) {
  if (exec_ == nullptr) {
    throw std::runtime_error(
        "CutoffGraphImpl: execution strategy is null; the strategy is injected "
        "at construction and there is no fallback to select internally");
  }
}

NeighborNeeds CutoffGraphImpl::needs() const {
  // REQ_FULL for many-body aggregation, the model cutoff, fx/fy/fz in either
  // newton mode — and REQ_GHOST exactly when the export accepts ghosts.
  //
  // The ghost request is what puts ghost centres into `ilist` (as
  // `LammpsView::gnum` entries after the local ones) so `GraphBridge` can walk
  // them. Without it the neighbour list carries no ghost->local edge at all,
  // and a target-aggregating model is wrong at its very first layer. The
  // facade's translation of this field — `if (needs.ghost) flags |=
  // NeighConst::REQ_GHOST;` — is unchanged: flipping the answer here was the
  // whole edit.
  return NeighborNeeds{/*full=*/true,
                       /*ghost=*/supports_pbc_,
                       /*cutoff=*/cutoff_,
                       /*comm_reverse=*/kReverseCommSize,
                       /*comm_reverse_off=*/kReverseCommSize};
}

void CutoffGraphImpl::coeff(const std::vector<std::string>& tokens,
                            int ntypes) {
  if (static_cast<int>(tokens.size()) != ntypes) {
    throw std::runtime_error(
        "pair_coeff for molnex needs: * * <Z1> ... <Zntypes> (one Z per atom "
        "type); got " +
        std::to_string(tokens.size()) + " token(s) for " +
        std::to_string(ntypes) + " atom type(s)");
  }

  // 0-based and dense: entry t-1 is the Z of LAMMPS type t, and the length is
  // ntypes — the layout SpeciesMap documents. Built whole before anything is
  // stored, so a rejected token leaves no half-filled table behind.
  std::vector<int> z_by_type;
  z_by_type.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    z_by_type.push_back(
        parse_atomic_number(tokens[i], static_cast<int>(i) + 1));
  }

  // SpeciesMap's constructor IS the validation: a Z the export was never built
  // for used to reach the network as an embedding index it has no row for and
  // produce a confident, wrong energy.
  species_map_.emplace(std::move(z_by_type), species_);
}

void CutoffGraphImpl::init(const LammpsRunEnv& env) {
  // (1) Single domain, early diagnostic. The hard guard is compute()'s
  // zero-ghost refusal; this catches the common misconfiguration (multi-rank
  // run of a non-periodic export) at setup, where the message can still name
  // the remedy.
  if (!supports_pbc_ && env.newton_pair == 1 && env.nprocs > 1) {
    throw std::runtime_error(
        "pair_molnex without supports_pbc assumes a single domain; run on 1 "
        "MPI rank or re-export with supports_pbc=true");
  }

  // (2) special_bonds. LAMMPS deletes a 1-2 / 1-3 / 1-4 pair from the neighbour
  // list outright when its coefficient is 0.0 — and `0.0 0.0 0.0` is the
  // DEFAULT. GraphBridge would then never see those pairs, and a many-body
  // model would be evaluated on a graph missing exactly its nearest
  // neighbours: wrong energies, wrong forces, no diagnostic. A coefficient
  // other than 1.0 means LAMMPS believes a scaling applies, and molnex applies
  // none, so anything but 1.0 is refused rather than approximated.
  //
  // Only reachable with a bond topology present: `atom_style atomic` has no
  // special pairs, so an atomic system keeps running whatever its (unused)
  // coefficients say.
  if (env.molecular != 0) {
    for (int k = kFirstSpecial; k <= kLastSpecial; ++k) {
      if (env.special_lj[k] != kSpecialKeepsPair ||
          env.special_coul[k] != kSpecialKeepsPair) {
        throw std::runtime_error(
            "pair_molnex on a molecular system requires 'special_bonds lj/coul "
            "1.0 1.0 1.0'; LAMMPS removes 1-2 / 1-3 / 1-4 pairs from the "
            "neighbour list wherever the coefficient is 0.0 (its default) and "
            "scales the rest, so this pair style would be handed a cutoff "
            "graph "
            "missing its nearest neighbours — molnex applies no special "
            "scaling "
            "of its own, and a coefficient other than 1.0 says LAMMPS expects "
            "one");
      }
    }
  }

  // (3) Newton pair, for a periodic contract. THIS IS THE PRECONDITION OF THE
  // CONSTANT `needs_pair_reverse_comm = false` in compute(): ghost forces are
  // left in atom->f for Verlet::run's global reverse comm to reduce, and that
  // reduction happens only `if (force->newton)`. With newton off nothing
  // reduces them, and every ghost's contribution — the cross-term of a local
  // site energy with respect to a periodic image's coordinates — is silently
  // discarded. The two decisions are one and must be changed together. The
  // non-periodic condition above is independent: newton off on several ranks
  // still passes there.
  if (supports_pbc_ && env.newton_pair != 1) {
    throw std::runtime_error(
        "pair_molnex with supports_pbc requires newton on: ghost forces are "
        "accumulated into atom->f and reduced onto their owning atoms by the "
        "integrator's global reverse communication, which LAMMPS performs only "
        "when newton is on — with it off, every ghost contribution is silently "
        "discarded. Add 'newton on' to the input script (it is the default), "
        "or "
        "re-export without supports_pbc for a single-domain run");
  }

  // (4) D3.2, the fat-ghost condition. A T-layer message-passing network's site
  // energy depends on every atom within T * cutoff, so a single-shot evaluation
  // needs this rank to hold a copy of all of them. LAMMPS' default ghost cutoff
  // is cutoff + skin, which suffices only at T == 1; at T >= 2 the outer sites
  // of the graph are missing neighbours and the model returns confident, wrong
  // forces with no diagnostic anywhere.
  //
  // SCOPED TO `supports_pbc` CONTRACTS, because that is the scope of the
  // problem: the condition is about whether GHOSTS reach far enough, and a
  // non-periodic export has none. It requests no ghost neighbours (`needs()`)
  // and compute() refuses outright when `view.nghost != 0`, so every atom of
  // the system is a local graph node and the receptive field is complete at any
  // T, whatever the ghost cutoff happens to be. That zero-ghost guard IS the
  // non-periodic path's completeness enforcement — applying the ghost threshold
  // on top of it would refuse a correct single-molecule run of any T >= 2 model
  // (MACE, PiNet: most of the zoo) over a `comm_modify` line that could not
  // change one number of its result.
  //
  // THE ZERO-MEANS-UNSET RULE LIVES HERE AND NOWHERE ELSE. The facade snapshots
  // comm->cutghostuser raw, and LAMMPS holds 0.0 when no `comm_modify cutoff`
  // ran; turning that into "the effective cutoff is cutoff + skin" is this
  // layer's single responsibility. Interpreting it in the facade too would give
  // one rule two implementations that could drift apart.
  //
  // FAIL CLOSED — never a warning. A warning on a condition that silently
  // corrupts every force of a multi-day run is not a diagnostic, it is a line
  // in a log nobody reads.
  const double effective_ghost_cutoff =
      env.ghost_cutoff > 0.0 ? env.ghost_cutoff : cutoff_ + env.skin;
  const double required_ghost_cutoff =
      receptive_field_layers_ * cutoff_ + env.skin;
  if (supports_pbc_ && effective_ghost_cutoff < required_ghost_cutoff) {
    throw std::runtime_error(
        "pair_molnex: this model has " +
        std::to_string(receptive_field_layers_) +
        " message-passing layer(s), so each atom's energy depends on every "
        "atom "
        "within " +
        std::to_string(receptive_field_layers_) +
        " * cutoff = " + std::to_string(receptive_field_layers_ * cutoff_) +
        " A, and this rank must hold a copy of all of them. The ghost cutoff "
        "is "
        "only " +
        std::to_string(effective_ghost_cutoff) + " A but must be at least " +
        std::to_string(required_ghost_cutoff) + " A (" +
        std::to_string(receptive_field_layers_) + " * " +
        std::to_string(cutoff_) + " + skin " + std::to_string(env.skin) +
        "). Add 'comm_modify cutoff " + std::to_string(required_ghost_cutoff) +
        "' to the input script; running as configured would evaluate the model "
        "on a graph whose outer atoms are missing neighbours, with no error "
        "and "
        "no warning");
  }

  // Snapshotted LAST, so it is recorded only on a run this method accepted:
  // compute() reads it to re-assert the newton invariant, and a value stored
  // before the checks could describe a configuration that was then refused.
  newton_pair_ = env.newton_pair;
}

EvalResult CutoffGraphImpl::compute(const LammpsView& view) {
  // One guard, then everything below dereferences directly. Empty means the
  // input script reached a run without a complete pair_style / pair_coeff pair.
  if (!species_map_) {
    throw std::runtime_error(
        "pair_style molnex: pair_style/pair_coeff not initialized (the type -> "
        "Z table arrives with pair_coeff)");
  }

  // Ahead of everything else: a local-only export handed ghosts would evaluate
  // a system silently missing its periodic images, so nothing is staged and no
  // force lands until this has passed.
  if (!supports_pbc_ && view.nghost != 0) {
    throw std::runtime_error(
        "pair_molnex (supports_pbc=false) requires no ghost atoms (single "
        "molecule, 1 rank, non-periodic). nghost=" +
        std::to_string(view.nghost) +
        ". Re-export with supports_pbc=true for periodic / multi-rank MD.");
  }

  // The cheap re-assertion of the newton invariant. See `newton_pair_` for what
  // it does and does not buy: it reads this object's own init-time snapshot, so
  // it catches compute() being reached on a periodic contract whose newton
  // precondition was never validated — an internal ordering fault — and not a
  // LAMMPS-side change after init, which init_style re-running at every run
  // setup already covers. An empty optional means init never completed, which
  // is exactly as unsafe as newton being off.
  if (supports_pbc_ && newton_pair_.value_or(0) != 1) {
    throw std::runtime_error(
        "pair_molnex with supports_pbc reached compute() without a validated "
        "newton setting: ghost forces are reduced only by the integrator's "
        "global reverse communication, which runs only with newton on, so "
        "CutoffGraphImpl::init must have confirmed it first. Ensure 'newton "
        "on' "
        "is set (it is the default); if it is, this is an ordering fault "
        "inside "
        "the pair style rather than a configuration problem");
  }

  // Graph atom count: local-only when supports_pbc is false, local + ghost
  // otherwise. The guard above is what makes the two agree on the local-only
  // path — there, nall() IS nlocal.
  const int n_graph = supports_pbc_ ? view.nall() : view.nlocal;

  // WHICH SITES THIS RANK OWNS, as the model's `local_mask` input. LAMMPS lays
  // atoms out locals-first, so ownership is the prefix `[0, nlocal)` — the one
  // fact that turns a graph over local + ghost into a PARTITION of the total
  // energy. An all-ones mask here would make the model return
  // sum_{L u G} E_i and every rank would count its ghosts' site energies as its
  // own; a mask built from anything but the LAMMPS layout would silently select
  // the wrong sites.
  const at::Tensor local_mask =
      at::zeros({n_graph}, at::TensorOptions().dtype(at::kBool));
  if (view.nlocal > 0) local_mask.narrow(0, 0, view.nlocal).fill_(true);

  // Four calls, one pipeline: stage_nodes copies Z / pos (fp64; the model-dtype
  // cast is the Exec's), GraphBridge collects and refilters against the cutoff
  // (sorting only if the export declared it needs sorted edges), the Exec runs
  // the model and hands back CPU float64 outputs this method owns, ForceScatter
  // accumulates them onto view.f in LAMMPS units.
  const molnex::lammps_bridge::NodeTensors nodes =
      molnex::lammps_bridge::stage_nodes(view, *species_map_, n_graph);
  const at::Tensor edge_index = graph_bridge_.build(view, n_graph);
  const molnex::lammps_bridge::ModelOutputs out =
      exec_->run(nodes.Z, nodes.pos, edge_index, local_mask);
  // Forces over ALL n_graph rows, ghosts included. A ghost's force is not a
  // by-product to be discarded: it is the derivative of LOCAL site energies
  // with respect to that ghost's coordinates, and it belongs to the atom the
  // ghost is a copy of. Leaving it in atom->f is precisely what lets the
  // integrator's global reverse comm carry it to its owner.
  force_scatter_.scatter(view, out.forces, n_graph);

  EvalResult result;
  // D1, the partition energy. `atomic_energy` comes back for all n_graph sites;
  // the first `nlocal` of them are this rank's, and their sum is its share of
  // the total. Slice, convert, then sum — so `e_total` is by construction the
  // sum of the `e_atom` the facade tallies, rather than a second number that
  // could disagree with it (LAMMPS' own thermo compares eng_vdwl against the
  // eatom sum, and two independent computations would eventually differ).
  //
  // The model's own scalar `out.energy` is NOT read here. Under v3 it is the
  // same quantity — the mask above is what makes it so — but deriving both
  // reported values from one tensor is what removes the possibility of a
  // mismatch entirely.
  result.e_atom = out.atomic_energy.slice(0, 0, view.nlocal) * energy_conv_;
  result.e_total = result.e_atom.sum().item<double>();
  // `defined()` is the whole presence test — read_model_outputs leaves the
  // virial undefined unless it holds exactly six numbers, so a defined one is
  // readable in full. The `supports_virial_` conjunction is the contract's
  // side of the same statement: an export that declares no virial must not have
  // one tallied for it, whatever its execution strategy happens to return.
  if (supports_virial_ && out.virial.defined()) {
    result.virial = out.virial * energy_conv_;
  }
  // D2: THE REDUCTION BELONGS TO THE INTEGRATOR, so this is a CONSTANT false
  // and not an expression. Verlet::run issues one global comm->reverse_comm()
  // whenever force->newton is set, and it reduces every ghost contribution any
  // style wrote into atom->f — including the ones ForceScatter just wrote. A
  // pair-level comm->reverse_comm(this) on top of that would send each ghost
  // force home TWICE, which is the bug this spec exists to fix.
  //
  // ITS SAFETY RESTS ENTIRELY ON init()'s NEWTON ENFORCEMENT above: without
  // newton there is no global reduction, and this constant would drop every
  // ghost force instead of double-counting it. Any change that relaxes that
  // enforcement must revisit this line in the same edit.
  //
  // Permanently false for the cutoff-graph contract. A future contract would
  // fill it true only if it scatters ghost forces OUTSIDE newton-reduced
  // integration — and doing so also re-activates ContractImpl::pack_reverse /
  // unpack_reverse, which LAMMPS can reach through no other route than the
  // facade's existing `if (r.needs_pair_reverse_comm)` call site.
  result.needs_pair_reverse_comm = false;
  return result;
}

int CutoffGraphImpl::pack_reverse(const LammpsView& view, int n, int first,
                                  double* buf) {
  return molnex::lammps_bridge::pack_reverse_forces(view, n, first, buf);
}

void CutoffGraphImpl::unpack_reverse(const LammpsView& view, int n,
                                     const int* list, const double* buf) {
  molnex::lammps_bridge::unpack_reverse_forces(view, n, list, buf);
}

}  // namespace molnex::lammps
