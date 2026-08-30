// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The runtime-contract layer: what a `pair_style molnex` run actually DOES,
// once the export's meta.json has said which contract it speaks.
//
// Three layers meet at this file, and the gradient between them is the point:
//
//   bridge/  — LAMMPS-free primitives that do not know the meta schema exists
//              (SpeciesMap, stage_nodes, GraphBridge, Exec, ForceScatter).
//   impls/   — LAMMPS-free strategy that DOES know the contract. The meta
//              schema enters the C++ here and nowhere below.
//   facade   — pair_molnex.cpp, the only file that includes a LAMMPS header.
//
// This header therefore includes **no LAMMPS header** — not `pair.h`, not
// `atom.h`, not `error.h` — which is what keeps the whole of layer 2 + layer 3
// drivable by ctest on a machine with no LAMMPS build, no CUDA and no `.pt2`
// (the `Exec` is injected; see `CutoffGraphImpl`). It speaks the same two
// dialects as its neighbours: everything here throws std::runtime_error, and
// pair_molnex.cpp translates at the seam into `error->all(FLERR, ...)`.
//
// The three structs below are attribute-only aggregates, shaped after
// `src/molix/md/types.py`'s `ForceOutput` (named values travelling together, no
// methods) with the attribute-contract style of
// `src/molix/md/neighbors.py`'s `NeighborStrategy`.

#ifndef MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H
#define MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H

#include <memory>
#include <string>
#include <vector>

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"
#include "lammps_contract.h"

namespace molnex::interface {
class ModelRunner;
}  // namespace molnex::interface

namespace molnex::lammps {

/// What the impl needs LAMMPS to set up for it, as a request table.
///
/// Every field is EXECUTED MECHANICALLY by the facade — `full` and `ghost`
/// become `NeighConst` flags on one `neighbor->add_request`, `cutoff` is what
/// `init_one()` returns, and the two comm sizes are assigned to the pair
/// style's own `comm_reverse` / `comm_reverse_off`. Nothing here is
/// interpreted, which is why nothing here is a predicate.
///
/// In particular there is NO `single_rank` field. That constraint is real, but
/// a predicate in a request table forces the facade to explain it, i.e. to grow
/// exactly the semantic `if` this layer exists to remove; its enforcement is
/// `ContractImpl::init`'s.
struct NeighborNeeds {
  /// Request the full neighbour list (every ordered pair), which many-body
  /// aggregation requires — a half list would silently halve the messages.
  bool full;
  /// Request ghost neighbours (`REQ_GHOST`).
  ///
  /// Describes ONLY the neighbour-list request made to LAMMPS. It carries no
  /// statement about force reduction — that is
  /// `EvalResult::needs_pair_reverse_comm`, and the two move in opposite
  /// directions in lammps-iface-05-pbc, which is why they cannot be merged.
  bool ghost;
  /// Neighbour cutoff in angstrom, from the export meta.
  double cutoff;
  /// Doubles per atom the reverse direction packs (fx, fy, fz).
  int comm_reverse;
  /// Same, for the newton-off variant.
  int comm_reverse_off;
};

/// What one evaluation produced, in LAMMPS-facing form.
///
/// THERE IS NO `forces` MEMBER, AND THAT IS NOT AN OMISSION. The impl has
/// already accumulated the model's forces into `atom->f` (through
/// `ForceScatter`) before it returns, because `atom->f` is precisely the array
/// `comm->reverse_comm` reads and `virial_fdotr_compute` sums. A `forces` field
/// added here would either duplicate that array or invite a caller to apply it
/// twice. The asymmetry is deliberate: energy and virial come back as values,
/// forces are already where LAMMPS keeps them.
///
/// `needs_pair_reverse_comm` IS THE REDUCTION-OWNERSHIP SEAM, and it is stated
/// next to the forces asymmetry because it is the other half of the same
/// question. It answers, per step, exactly one thing: **must the pair style
/// itself issue `comm->reverse_comm(this)`** — not "were ghost forces
/// scattered", and not anything about `NeighborNeeds::ghost`, from which it is
/// independent. The facade reads it as a mechanical branch on a returned value
/// and reads no capability flag of its own. This spec's impl answers "yes when
/// there are ghosts"; lammps-iface-05-pbc answers "never, because the
/// integrator's own reduction owns it" — the same field and the same call site,
/// a different answer, no surgery on the protocol layer. A future contract that
/// scatters outside the newton-reduced integration fills it `true` again.
///
/// `virial` is well-formed-or-undefined BY CONSTRUCTION: `read_model_outputs`
/// (lammps-iface-03-graph) refuses anything that is not exactly six elements
/// before defining it, so a defined virial always has six readable components
/// and the facade performs no length check of its own. `virial.defined()` is
/// the whole presence test — there is no companion flag that could disagree
/// with it.
///
/// THERE IS ALSO NO `e_atom` OR PER-ATOM VIRIAL, and that too is a decision
/// rather than an oversight: nothing in this spec can PRODUCE either (a model
/// returns energy, forces and an optional global virial), and a field with no
/// producer is as dead as one with no consumer. `compute pe/atom` and
/// `compute stress/atom` are met by an explicit facade error instead of the
/// silent zeros they used to get. lammps-iface-05-pbc, whose whole subject is
/// the per-site energy sum, adds `e_atom` here WITH its producer — that arrival
/// is the extension this design anticipates, not a reversal of this decision.
///
/// ACCRETION RULE (the same one `LammpsRunEnv` carries, stated on the struct
/// most likely to be violated): every field of `EvalResult` is a per-step
/// OUTPUT of one evaluation, and each must land together with its producer and
/// its consumer. A future "the facade must also do X for me" obligation is not
/// an output and needs a different home. `needs_pair_reverse_comm` is the one
/// explicitly grandfathered member — it is that seam, by definition, and is
/// labelled as such above.
///
/// Every member is set on every path `CutoffGraphImpl::compute` can return
/// through; there is no "left at whatever the default was" case.
struct EvalResult {
  /// Total energy of this evaluation, already converted to the active LAMMPS
  /// unit style. The facade adds it to `eng_vdwl` and does nothing else to it.
  double e_total;
  /// Global virial `(6,)` in `xx, yy, zz, xy, xz, yz` order, already converted
  /// — or an undefined tensor when this evaluation produced none.
  at::Tensor virial;
  /// Whether the facade must issue `comm->reverse_comm(this)` this step.
  bool needs_pair_reverse_comm;
};

/// The LAMMPS globals the impl's `init` reads, as a read-only snapshot.
///
/// Assembled by the facade out of `force->newton_pair`, `comm->nprocs`,
/// `atom->molecular` and `force->special_lj` / `force->special_coul`, so that
/// nothing under `impls/` has to include a LAMMPS header to see them.
///
/// ACCRETION RULE (consumer-arrives): this struct carries only the globals some
/// code READS TODAY. A later spec adding a field must land it in the same batch
/// as the code that reads it — lammps-iface-05-pbc, for instance, will need the
/// ghost cutoff for its D3.2 check, and that field arrives with that check, not
/// before it. Written on the struct because this is exactly where a speculative
/// "while we are here, let us also pass..." would go.
struct LammpsRunEnv {
  /// `force->newton_pair`: 1 when pairwise Newton's third law is on.
  int newton_pair;
  /// `comm->nprocs`: the number of MPI ranks this run is decomposed over.
  int nprocs;
  /// `atom->molecular`: 0 for `atom_style atomic`, i.e. no bond topology and
  /// therefore no special pairs at all.
  int molecular;
  /// `force->special_lj`, `(4,)`: index 0 is unused, 1..3 are the 1-2 / 1-3 /
  /// 1-4 Lennard-Jones scale factors. Borrowed, never owned.
  const double* special_lj;
  /// `force->special_coul`, `(4,)`, same layout. Borrowed, never owned.
  const double* special_coul;
};

/// One runtime contract, implemented.
///
/// A `Contract` names where the model's edge set comes from; this class is what
/// a pair style DOES with such a model. Sub-classing is by contract family
/// (cutoff graph, fixed topology), never by network — `CutoffGraphImpl` serves
/// PiNet, MACE and Allegro alike, so "no per-model C++" is not at risk.
///
/// The six virtuals are the whole of what the facade delegates; between them
/// they hold every semantic decision `PairMolnex` used to make inline.
///
/// Implementations keep a PUBLIC constructor: direct construction is the
/// primary constructor story (CLAUDE.md), `for_contract` is the closed-set
/// dispatch on top of it, and the ctest cases construct directly.
class ContractImpl {
 public:
  virtual ~ContractImpl();

  ContractImpl(const ContractImpl&) = delete;
  ContractImpl& operator=(const ContractImpl&) = delete;

  /// What LAMMPS must set up for this contract.
  ///
  /// \returns A request table the caller executes verbatim; see
  ///          `NeighborNeeds`.
  virtual NeighborNeeds needs() const = 0;

  /// Interpret the residual `pair_coeff * *` tokens.
  ///
  /// The facade strips the two literal `*` and forwards the rest untouched:
  /// what those tokens MEAN is the contract's business, including how many of
  /// them there must be. Implementations parse with the standard library —
  /// never `utils::inumeric`, which is a LAMMPS symbol and would drag this
  /// whole layer back behind a LAMMPS build.
  ///
  /// \param tokens  The `pair_coeff` arguments after `* *`, in order.
  /// \param ntypes  `atom->ntypes`, the number of LAMMPS atom types.
  /// \throws std::runtime_error  On any token this contract cannot accept.
  virtual void coeff(const std::vector<std::string>& tokens, int ntypes) = 0;

  /// Check this run's LAMMPS configuration against the export.
  ///
  /// The hook for every constraint that is a PREDICATE over the run rather than
  /// a request to LAMMPS — which is why `NeighborNeeds` carries none.
  ///
  /// \param env  Read-only snapshot of the LAMMPS globals; see `LammpsRunEnv`.
  /// \throws std::runtime_error  If the run cannot be served safely.
  virtual void init(const LammpsRunEnv& env) = 0;

  /// Evaluate one step over the borrowed LAMMPS arrays.
  ///
  /// Forces are accumulated into `view.f` before returning; see `EvalResult`.
  ///
  /// \param view  Borrowed atom arrays and neighbour spans for this step.
  /// \returns The energy, optional virial and reverse-comm decision.
  /// \throws std::runtime_error  If the system does not match the export, or if
  ///                             any staged component refuses.
  virtual EvalResult compute(const molnex::lammps_bridge::LammpsView& view) = 0;

  /// Pack the reverse-communication buffer for `[first, first + n)`.
  ///
  /// \param view   Borrowed LAMMPS atom arrays.
  /// \param n      Number of atoms in the range.
  /// \param first  Index of the first atom of the range.
  /// \param buf    LAMMPS-owned destination with room for `3 * n` doubles.
  /// \returns The number of doubles written.
  virtual int pack_reverse(const molnex::lammps_bridge::LammpsView& view, int n,
                           int first, double* buf) = 0;

  /// Accumulate a reverse-communication buffer onto the atoms named by `list`.
  ///
  /// \param view  Borrowed LAMMPS atom arrays; `f` is written in place.
  /// \param n     Number of atoms in `list`.
  /// \param list  Local indices of the receiving atoms, `(n,)`.
  /// \param buf   Source, `(3 * n,)` doubles.
  virtual void unpack_reverse(const molnex::lammps_bridge::LammpsView& view,
                              int n, const int* list, const double* buf) = 0;

  /// The implementation for `contract`, over the closed set of contracts.
  ///
  /// Dispatch on a closed enum, not a wrapper around one type's construction —
  /// which is why it is named like `Foo.from_file` rather than `make_*` /
  /// `create_*`. Defined in `registry.cpp` as an EXHAUSTIVE `switch` with no
  /// `default:` label, so adding a `Contract` enumerator is a `-Wswitch`
  /// compile-time error here rather than a runtime miss.
  ///
  /// \param contract_kind      Which contract to implement, normally
  ///                           `contract.contract()`; passed separately so the
  ///                           dispatch reads its key explicitly.
  /// \param contract           The validated export contract the impl reads its
  ///                           cutoff, species, capabilities and unit table
  ///                           from.
  /// \param lammps_unit_style  `update->unit_style` — the unit style the
  ///                           RUNNING simulation asked for, which is the only
  ///                           piece of this that the meta cannot supply. The
  ///                           impl converts model energies and forces into it.
  /// \param runner             Loaded model runner, borrowed for the lifetime
  ///                           of the returned impl: the caller must keep it
  ///                           alive at least that long.
  /// \returns The implementation, never null.
  /// \throws std::runtime_error  If no implementation exists for
  ///                             `contract_kind` (the message lists every
  ///                             contract that does dispatch), or if the
  ///                             execution strategy cannot be constructed.
  static std::unique_ptr<ContractImpl> for_contract(
      Contract contract_kind, const LammpsContract& contract,
      const std::string& lammps_unit_style,
      molnex::interface::ModelRunner& runner);

 protected:
  ContractImpl() = default;
};

}  // namespace molnex::lammps

#endif  // MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H
