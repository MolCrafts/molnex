// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// What a `pair_style molnex` run does with a deployed model.
//
// Three LAMMPS-side layers meet at this file, over the engine-neutral
// `molnex_runtime` (metadata, artifact, atom-energy execution):
//
//   bridge/  — LAMMPS-free primitives that do not know the model exists
//              (SpeciesMap, stage_nodes, GraphBridge, ForceScatter).
//   impls/   — LAMMPS-free strategy over the runtime: the model's metadata and
//              execution enter the LAMMPS layer here and nowhere below.
//   facade   — pair_molnex.cpp, the only file that includes a LAMMPS header.
//
// This header therefore includes **no LAMMPS header** — not `pair.h`, not
// `atom.h`, not `error.h` — which is what keeps bridge/ + impls/ drivable by
// ctest on a machine with no LAMMPS build, no CUDA and no package (the
// execution is injected; see `CutoffGraphImpl`). It speaks the same two
// dialects as its neighbours: everything here throws std::runtime_error, and
// pair_molnex.cpp translates at the seam into `error->all(FLERR, ...)`.
//
// The three structs below are attribute-only aggregates, shaped after
// `src/molix/md/types.py`'s `ForceOutput` (named values travelling together, no
// methods) with the attribute-contract style of
// `src/molix/md/neighbors.py`'s `NeighborStrategy`.

#ifndef MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H
#define MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H

#include <array>
#include <string>
#include <vector>

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"

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
  /// `EvalResult::needs_pair_reverse_comm`, and the two moved in OPPOSITE
  /// directions in lammps-iface-05-pbc (this one to true for a periodic
  /// contract, that one to a constant false), which is the demonstration of why
  /// they could never have been merged.
  ///
  /// True means LAMMPS also gives ghost atoms neighbour lists of their own, and
  /// appends them to `ilist` as `LammpsView::gnum` extra centres. Without it,
  /// the ghost->local edges a many-body model needs at its first layer are
  /// simply not in the list.
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
/// twice. Energy comes back as a value; forces are already where LAMMPS
/// keeps them. The model returns no virial.
///
/// `needs_pair_reverse_comm` IS THE REDUCTION-OWNERSHIP SEAM, and it is stated
/// next to the forces asymmetry because it is the other half of the same
/// question. It answers, per step, exactly one thing: **must the pair style
/// itself issue `comm->reverse_comm(this)`** — not "were ghost forces
/// scattered", and not anything about `NeighborNeeds::ghost`, from which it is
/// independent. The facade reads it as a mechanical branch on a returned value
/// and reads no capability flag of its own. lammps-iface-04-facade's impl
/// answered "yes when there are ghosts"; lammps-iface-05-pbc answers "never,
/// because the integrator's own reduction owns it" — the same field and the
/// same call site, a different answer, and not one line of the protocol layer
/// touched. That is the value the seam was kept for: a physically substantial
/// change of reduction ownership cost zero facade edits. A future contract that
/// scatters outside the newton-reduced integration fills it `true` again, and
/// doing so is also the only thing that can re-activate
/// `ContractImpl::pack_reverse` / `unpack_reverse`.
///
/// `e_atom` is the local slice of the model's per-atom energies. A per-atom
/// virial has no producer; `vflag_atom` remains an error in the facade.
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
  /// Partition energy of this evaluation, already converted to the active
  /// LAMMPS unit style. The facade adds it to `eng_vdwl` and does nothing else
  /// to it.
  ///
  /// It is the sum of `e_atom`, i.e. of the LOCAL atom energies only. Summing
  /// over ghosts as well would add `sum_{i in G} E_i`, which is dominated by
  /// the per-species constant offset and is therefore tens of eV per ghost — an
  /// error that grows with the rank count and the cutoff, and that no
  /// single-rank test can see.
  double e_total;
  /// Per-atom energies, **length `nlocal`**, already converted.
  ///
  /// THE LENGTH IS THE CONTRACT, and it is `nlocal` rather than `nall`: only
  /// atoms this rank owns get an entry, because only their atom energies belong
  /// to this rank's partition. The facade iterates THIS TENSOR'S OWN EXTENT
  /// when `eflag_atom` is set and must not consult `atom->nlocal` for the bound
  /// — two independent notions of "which atoms are local" would silently
  /// overrun or under-count the moment they disagreed.
  ///
  /// A CPU float64 1-D tensor the caller owns, never a view of a model output.
  at::Tensor e_atom;
  /// Whether the facade must issue `comm->reverse_comm(this)` this step.
  bool needs_pair_reverse_comm;
  /// Whether `virial` holds this evaluation's global virial. False means the
  /// facade derives it from the forces itself (`virial_fdotr_compute`, which
  /// needs the ghost forces the ghost mode writes); true means the forces were
  /// folded onto owned atoms, `fdotr` would miss every periodic image, and the
  /// model's own strain derivative is the only correct source.
  bool virial_from_model;
  /// Global virial in LAMMPS' convention and order — `-dE/deps`, already in
  /// the active unit style, `xx yy zz xy xz yz` — when `virial_from_model`.
  std::array<double, 6> virial;
};

/// The LAMMPS globals the impl's `init` reads, as a read-only snapshot.
///
/// Assembled by the facade out of `force->newton_pair`, `comm->nprocs`,
/// `atom->molecular` and `force->special_lj` / `force->special_coul`, so that
/// nothing under `impls/` has to include a LAMMPS header to see them.
///
/// ACCRETION RULE (consumer-arrives): this struct carries only the globals some
/// code READS TODAY. A later spec adding a field must land it in the same batch
/// as the code that reads it — `ghost_cutoff` and `skin` arrived with
/// lammps-iface-05-pbc's fat-ghost check and not one batch earlier. Written on
/// the struct because this is exactly where a speculative "while we are here,
/// let us also pass..." would go.
///
/// EVERY FIELD IS A RAW SNAPSHOT. The facade transcribes LAMMPS' value and does
/// not interpret it; the impl interprets and does not read LAMMPS. That split
/// matters most for `ghost_cutoff`, whose meaning is conditional (see below):
/// putting the interpretation in the facade would give one rule two
/// implementations that could drift, and would grow back a semantic `if` in the
/// layer that exists to have none.
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
  /// `comm->cutghostuser` in angstrom: the ghost cutoff the input script asked
  /// for with `comm_modify cutoff`, RAW.
  ///
  /// **`0.0` means the user set none**, and LAMMPS then uses `cutoff + skin`.
  /// That rule is applied inside `CutoffGraphImpl::init` and nowhere else —
  /// this field is the number LAMMPS holds, not the number that is in force.
  double ghost_cutoff;
  /// `neighbor->skin` in angstrom: the neighbour-list skin, RAW.
  double skin;
};

/// One runtime contract, implemented.
///
/// What a pair style does with a package whose forward matches the protocol.
/// `CutoffGraphImpl` is that implementation for every model.
///
/// The six virtuals are the whole of what the facade delegates; between them
/// they hold every semantic decision `PairMolnex` used to make inline.
///
/// The public constructor is the constructor. The pair style builds
/// `CutoffGraphImpl` directly, and the ctest cases do the same.
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
  /// \returns The energy and the reverse-comm decision.
  /// \throws std::runtime_error  If the system does not match the export, or if
  ///                             any staged component refuses.
  virtual EvalResult compute(const molnex::lammps_bridge::LammpsView& view) = 0;

  /// Pack the reverse-communication buffer for `[first, first + n)`.
  ///
  /// NOT INVOKED ON THE CUTOFF-GRAPH CONTRACT, and that is a statement about
  /// this moment rather than about the method. LAMMPS reaches these two
  /// overrides through `Pair::pack_reverse_comm` / `unpack_reverse_comm`, and
  /// those run only inside a `comm->reverse_comm(this)` that the pair style
  /// itself issued. `CutoffGraphImpl` fills
  /// `EvalResult::needs_pair_reverse_comm` with a constant false — the
  /// integrator's own global reverse comm reduces ghost forces — so the
  /// facade's call site never fires and neither does this.
  ///
  /// THE SINGLE RE-ENTRY ROUTE, and there is no second entrance: a contract
  /// that fills `needs_pair_reverse_comm` with true re-activates the facade's
  /// existing call site, and these overrides start being called again with no
  /// edit to the facade at all. A fixed-graph or layer-wise-communication impl
  /// that scatters ghost forces outside newton-reduced integration is the
  /// expected such contract. They stay declared because a LAMMPS `Pair` must
  /// provide them whether or not they are reached.
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
  /// Currently unexercised for the same reason, and re-enabled by the same
  /// single route, as `pack_reverse` — see its docstring.
  ///
  /// \param view  Borrowed LAMMPS atom arrays; `f` is written in place.
  /// \param n     Number of atoms in `list`.
  /// \param list  Local indices of the receiving atoms, `(n,)`.
  /// \param buf   Source, `(3 * n,)` doubles.
  virtual void unpack_reverse(const molnex::lammps_bridge::LammpsView& view,
                              int n, const int* list, const double* buf) = 0;

 protected:
  ContractImpl() = default;
};

}  // namespace molnex::lammps

#endif  // MOLNEX_LAMMPS_IMPLS_CONTRACT_IMPL_H
