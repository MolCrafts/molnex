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

CutoffGraphImpl::CutoffGraphImpl(const LammpsContract& contract,
                                 const std::string& lammps_unit_style,
                                 std::unique_ptr<molnex::lammps_bridge::Exec> exec)
    // units_for() is called exactly once, here, and its result is handed to the
    // delegated constructor: energy and force must not come from two separate
    // lookups that a later edit could let diverge.
    : CutoffGraphImpl(contract, contract.units_for(lammps_unit_style),
                      std::move(exec)) {}

CutoffGraphImpl::CutoffGraphImpl(const LammpsContract& contract,
                                 UnitConversion conv,
                                 std::unique_ptr<molnex::lammps_bridge::Exec> exec)
    : cutoff_(contract.cutoff()),
      supports_pbc_(contract.supports_pbc()),
      supports_virial_(contract.supports_virial()),
      energy_conv_(conv.energy),
      // Read whole and kept: `pair_coeff` runs after `pair_style` and needs the
      // export's species list to reject a Z the model was never built for.
      species_(contract.species()),
      graph_bridge_(cutoff_),
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
  // Value for value the request the facade used to make inline: REQ_FULL for
  // many-body aggregation, no ghost neighbours (that is lammps-iface-05-pbc's
  // change), the model cutoff, and fx/fy/fz in either newton mode.
  return NeighborNeeds{/*full=*/true,
                       /*ghost=*/false,
                       /*cutoff=*/cutoff_,
                       /*comm_reverse=*/kReverseCommSize,
                       /*comm_reverse_off=*/kReverseCommSize};
}

void CutoffGraphImpl::coeff(const std::vector<std::string>& tokens, int ntypes) {
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
  // (1) Single domain. Carried over verbatim from pair_molnex.cpp:261-263,
  // INCLUDING the fact that it only fires with newton_pair on: tightening it is
  // a behaviour change and belongs to lammps-iface-05-pbc, not to a migration.
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
            "scales the rest, so this pair style would be handed a cutoff graph "
            "missing its nearest neighbours — molnex applies no special scaling "
            "of its own, and a coefficient other than 1.0 says LAMMPS expects "
            "one");
      }
    }
  }
}

EvalResult CutoffGraphImpl::compute(const LammpsView& view) {
  // One guard, then everything below dereferences directly. Empty means the
  // input script reached a run without a complete pair_style / pair_coeff pair.
  if (!species_map_) {
    throw std::runtime_error(
        "pair_style molnex: pair_style/pair_coeff not initialized (the type -> "
        "Z table arrives with pair_coeff)");
  }

  // Migrated from pair_molnex.cpp:283-287, ahead of everything else: a
  // local-only export handed ghosts would evaluate a system silently missing
  // its periodic images, so nothing is staged and no force lands until this has
  // passed.
  if (!supports_pbc_ && view.nghost != 0) {
    throw std::runtime_error(
        "pair_molnex (supports_pbc=false) requires no ghost atoms (single "
        "molecule, 1 rank, non-periodic). nghost=" +
        std::to_string(view.nghost) +
        ". Re-export with supports_pbc=true for periodic / multi-rank MD.");
  }

  // Graph atom count: local-only when supports_pbc is false, local + ghost
  // otherwise (pair_molnex.cpp:292). The guard above is what makes the two
  // agree on the local-only path — there, nall() IS nlocal.
  const int n_graph = supports_pbc_ ? view.nall() : view.nlocal;

  // Four calls, one pipeline: stage_nodes copies Z / pos (fp64; the model-dtype
  // cast is the Exec's), GraphBridge collects, refilters against the cutoff and
  // sorts by (source, target), the Exec runs the model and hands back CPU
  // float64 outputs this method owns, ForceScatter accumulates them onto
  // view.f in LAMMPS units.
  const molnex::lammps_bridge::NodeTensors nodes =
      molnex::lammps_bridge::stage_nodes(view, *species_map_, n_graph);
  const at::Tensor edge_index = graph_bridge_.build(view, n_graph);
  const molnex::lammps_bridge::ModelOutputs out =
      exec_->run(nodes.Z, nodes.pos, edge_index);
  force_scatter_.scatter(view, out.forces, n_graph);

  EvalResult result;
  result.e_total = out.energy.item<double>() * energy_conv_;
  // `defined()` is the whole presence test — read_model_outputs leaves the
  // virial undefined unless it holds exactly six numbers, so a defined one is
  // readable in full. The `supports_virial_` conjunction is the contract's
  // side of the same statement: an export that declares no virial must not have
  // one tallied for it, whatever its execution strategy happens to return.
  if (supports_virial_ && out.virial.defined()) {
    result.virial = out.virial * energy_conv_;
  }
  // The reduction-ownership seam, over the same inputs and the same expression
  // as pair_molnex.cpp:340 evaluated inline. See EvalResult's docstring: this
  // spec answers "yes when there are ghosts", lammps-iface-05-pbc answers
  // "never", and neither the field nor the facade's call site moves.
  result.needs_pair_reverse_comm = supports_pbc_ && view.nghost > 0;
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
