// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `cutoff_graph_impl.h` for the design notes (composition point, contract
// family rather than model, injected execution).
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

CutoffGraphImpl::CutoffGraphImpl(const molnex::runtime::Metadata& metadata,
                                 double energy_factor,
                                 molnex::runtime::SiteEnergyExec exec,
                                 GraphMode mode)
    : cutoff_(metadata.cutoff),
      receptive_field_layers_(metadata.receptive_field_layers),
      mode_(mode),
      energy_conv_(energy_factor),
      // Kept whole: `pair_coeff` runs after `pair_style` and needs the model's
      // species to reject a Z it was never built for.
      species_(metadata.species),
      graph_bridge_(cutoff_),
      // The force factor equals the energy factor only because the length unit
      // is checked below to be LAMMPS' own.
      force_scatter_(energy_factor),
      exec_(exec) {
  // [F] = [E]/[L]: one factor serves both only while the model measures length
  // in angstrom, which LAMMPS `real` and `metal` both do. A Bohr export must
  // be refused here rather than run with forces mis-scaled.
  if (metadata.length_unit != "angstrom") {
    throw std::runtime_error(
        "pair_molnex: the model's length unit is '" + metadata.length_unit +
        "', not 'angstrom'; LAMMPS 'real' and 'metal' measure length in "
        "angstrom, and this pair style converts energy only");
  }
}

NeighborNeeds CutoffGraphImpl::needs() const {
  // REQ_FULL for many-body aggregation, the model cutoff, fx/fy/fz in either
  // newton mode. REQ_GHOST in the ghost mode only: ghost centres in `ilist`
  // are what lets GraphBridge close the edge set over the ghost shell. The
  // folded mode reaches every pair from an owned centre and needs none.
  return NeighborNeeds{/*full=*/true,
                       /*ghost=*/mode_ == GraphMode::kGhost,
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
  // (1) special_bonds. LAMMPS deletes a 1-2 / 1-3 / 1-4 pair from the neighbour
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

  // The folded mode writes no ghost force and stages no ghost shell, so (2)
  // and (3) below are ghost-mode conditions. What it needs instead is that
  // every ghost is an image of an atom THIS rank owns, which on more than one
  // rank is false for the ghosts other ranks own.
  if (mode_ == GraphMode::kFold) {
    if (env.nprocs != 1) {
      throw std::runtime_error(
          "pair_molnex: 'mode fold' folds every periodic image onto the atom "
          "it copies, which this rank must own; with " +
          std::to_string(env.nprocs) +
          " ranks a ghost may be another rank's atom. Use 'mode ghost' (the "
          "default on more than one rank)");
    }
    newton_pair_ = env.newton_pair;
    return;
  }

  // (2) Newton pair. Ghost forces are left in atom->f for the integrator's
  // global reverse comm, and that reduction happens only when newton is on.
  // With newton off every ghost contribution is discarded. The constant
  // `needs_pair_reverse_comm = false` in compute() depends on this check:
  // a pair-level reverse comm on top of the integrator's would double-count.
  if (env.newton_pair != 1) {
    throw std::runtime_error(
        "pair_molnex requires newton on: ghost forces are accumulated into "
        "atom->f and reduced onto their owning atoms by the integrator's "
        "global reverse communication, which LAMMPS performs only when newton "
        "is on — with it off, every ghost contribution is silently discarded. "
        "Add 'newton on' to the input script (it is the default)");
  }

  // (3) Fat ghost. A T-layer network's site energy depends on every atom
  // within T * cutoff, so this rank must hold a copy of all of them. LAMMPS'
  // default ghost cutoff is cutoff + skin, which suffices only at T == 1.
  //
  // THE ZERO-MEANS-UNSET RULE LIVES HERE AND NOWHERE ELSE. The facade snapshots
  // comm->cutghostuser raw, and LAMMPS holds 0.0 when no `comm_modify cutoff`
  // ran; the effective cutoff is then cutoff + skin.
  const double effective_ghost_cutoff =
      env.ghost_cutoff > 0.0 ? env.ghost_cutoff : cutoff_ + env.skin;
  const double required_ghost_cutoff =
      receptive_field_layers_ * cutoff_ + env.skin;
  if (effective_ghost_cutoff < required_ghost_cutoff) {
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

  if (mode_ == GraphMode::kFold) return compute_folded(view);

  // Re-assert the newton snapshot. An empty optional means init never
  // completed, which drops ghost forces the same way newton off does.
  if (newton_pair_.value_or(0) != 1) {
    throw std::runtime_error(
        "pair_molnex reached compute() without a validated newton setting: "
        "ghost forces are reduced only by the integrator's global reverse "
        "communication, which runs only with newton on, so "
        "CutoffGraphImpl::init must have confirmed it first. Ensure 'newton "
        "on' is set (it is the default); if it is, this is an ordering fault "
        "inside the pair style rather than a configuration problem");
  }

  // Locals and ghosts. With nghost == 0 this is nlocal, and local_mask is
  // all true.
  const int n_graph = view.nall();

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

  // stage_nodes copies Z / pos (fp64; the model-dtype cast is the
  // execution's), GraphBridge collects and refilters against the cutoff in
  // neighbour-list order, SiteEnergyExec runs the model, ForceScatter
  // accumulates onto view.f.
  const molnex::lammps_bridge::NodeTensors nodes =
      molnex::lammps_bridge::stage_nodes(view, *species_map_, n_graph);
  const at::Tensor pair_index = graph_bridge_.build(view, n_graph);
  // Ghost copies are staged at their image coordinates: no shift.
  const at::Tensor shifts = at::zeros({pair_index.size(0), 3},
                                      at::TensorOptions().dtype(at::kDouble));
  const molnex::runtime::SiteEnergyOutputs out =
      exec_.run(nodes.Z, nodes.pos, pair_index, shifts, local_mask);
  // Forces over ALL n_graph rows, ghosts included. A ghost's force is not a
  // by-product to be discarded: it is the derivative of LOCAL site energies
  // with respect to that ghost's coordinates, and it belongs to the atom the
  // ghost is a copy of. Leaving it in atom->f is precisely what lets the
  // integrator's global reverse comm carry it to its owner.
  force_scatter_.scatter(view, out.forces, n_graph);

  EvalResult result;
  // D1, the partition energy. `site_energy` comes back for all n_graph sites;
  // the first `nlocal` of them are this rank's, and their sum is its share of
  // the total. Slice, convert, then sum — so `e_total` is by construction the
  // sum of the `e_atom` the facade tallies, rather than a second number that
  // could disagree with it (LAMMPS' own thermo compares eng_vdwl against the
  // eatom sum, and two independent computations would eventually differ).
  //
  // The model's own scalar `out.energy` is not read here. Deriving both
  // reported values from `site_energy` keeps them one number and its parts.
  result.e_atom = out.site_energy.slice(0, 0, view.nlocal) * energy_conv_;
  result.e_total = result.e_atom.sum().item<double>();
  // THE REDUCTION BELONGS TO THE INTEGRATOR, so this is a constant false.
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
  // LAMMPS' fdotr over local + ghost forces is the ghost mode's virial.
  result.virial_from_model = false;
  result.virial = {};
  return result;
}

EvalResult CutoffGraphImpl::compute_folded(const LammpsView& view) {
  if (!newton_pair_) {
    throw std::runtime_error(
        "pair_molnex reached compute() before CutoffGraphImpl::init accepted "
        "the run; this is an ordering fault inside the pair style");
  }
  // Owned atoms only: every site is local, every image a shifted pair.
  const int n = view.nlocal;
  const at::Tensor local_mask =
      at::ones({n}, at::TensorOptions().dtype(at::kBool));
  const molnex::lammps_bridge::NodeTensors nodes =
      molnex::lammps_bridge::stage_nodes(view, *species_map_, n);
  const molnex::lammps_bridge::FoldedGraph graph =
      graph_bridge_.build_folded(view);
  const molnex::runtime::SiteEnergyOutputs out =
      exec_.run(nodes.Z, nodes.pos, graph.pair_index, graph.shifts, local_mask);
  // Forces on owned atoms only. A ghost's share is already in its owner's row:
  // the image enters the model as the owner's position plus a constant shift.
  force_scatter_.scatter(view, out.forces, n);

  EvalResult result;
  result.e_atom = out.site_energy * energy_conv_;
  result.e_total = result.e_atom.sum().item<double>();
  result.needs_pair_reverse_comm = false;
  // The model's +dE/deps, as LAMMPS' -dE/deps in xx yy zz xy xz yz. fdotr
  // cannot stand in: it sums r_i . f_i over the atoms present, and with the
  // images folded away it would miss every periodic contribution.
  const auto w = out.virial.accessor<double, 2>();
  result.virial_from_model = true;
  result.virial = {-energy_conv_ * w[0][0], -energy_conv_ * w[1][1],
                   -energy_conv_ * w[2][2], -energy_conv_ * w[0][1],
                   -energy_conv_ * w[0][2], -energy_conv_ * w[1][2]};
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

// Out of line so the base vtable is emitted once, in this translation unit.
ContractImpl::~ContractImpl() = default;

}  // namespace molnex::lammps
