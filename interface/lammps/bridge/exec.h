// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The model-execution strategy, expressed backend-neutrally.
//
// `Exec` is the seam between "what the bridge staged" and "how this model
// happens to be run". Everything above it deals in CPU tensors with molnex
// semantics — `Z` int64, `pos` float64 angstrom, `edge_index` int64 `(ne, 2)`
// in the repo Edge Convention — and gets back CPU float64 energy / forces /
// optional virial. Device transfer, model-dtype casting, padding, masking and
// the copy back to the host are each implementation's private policy, because
// they are the parts that differ between backends and nothing above needs to
// know which one is in play.
//
// BACKEND-NEUTRAL, AND THAT IS LOAD-BEARING. No `.pt2`, no
// AOTIModelPackageLoader, no "the weights were baked in at export time" anywhere
// in this header's signatures. The closed spec
// `lammps-pair-molnex-01-aoti-force-export-wall` records a TorchScript route in
// which the C++ side holds a scripted module and calls `.backward()` at RUNTIME
// to obtain forces; a `TorchScriptExec` implementing that attaches here as a
// third implementation with no change to this interface, and the facade would
// not be able to tell. Bake an AOTI assumption into `run`'s parameters or into
// `ModelOutputs` and that route stops fitting — which is the whole reason the
// abstraction is drawn at the semantic level rather than at
// `std::vector<at::Tensor>`.
//
// OWNERSHIP OF RETURNED TENSORS IS PART OF THE CONTRACT, NOT A STYLE NOTE. See
// `Exec::run`. `ModelRunner::run_graphed` returns tensors that live at FIXED
// replay addresses; the next `replay()` overwrites them in place. An
// implementation that handed those back directly would silently rewrite the
// caller's previous-step results — a force field that looks "one step late",
// only on the CUDA-graph path. The migrated facade got this right by accident
// (its `.to(kDouble).to(kCPU)` chain happened to copy); componentising it turns
// the accident into a promise.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

namespace molnex::lammps_bridge {

/// What a model evaluation produced, in LAMMPS-facing form.
///
/// Attribute-only aggregate, mirroring `molix.md.types.ForceOutput`: named
/// tensors travelling together with no invariant beyond the ones their producer
/// already established. There is no `have_virial` flag beside them —
/// `virial.defined()` IS the presence test, so the two can never disagree.
///
/// Every defined member is a CPU float64 tensor the caller owns (see
/// `Exec::run`).
struct ModelOutputs {
  /// Total model energy, a single-element float64 CPU tensor, in model units.
  ///
  /// Under schema v3 this is the model's own masked site-energy sum. The
  /// cutoff-graph impl does NOT tally it: it re-sums `atomic_energy` over the
  /// local slice instead, so that the energy it reports and the per-atom
  /// energies it reports are one number and its parts rather than two
  /// independent computations LAMMPS' thermo could catch disagreeing. The
  /// output stays decoded and validated here because it is output zero of the
  /// declared contract.
  at::Tensor energy;
  /// Per-node forces `(n, 3)`, contiguous float64 on the CPU, in model units.
  at::Tensor forces;
  /// Per-site energies `(n,)`, contiguous float64 on the CPU, in model units.
  ///
  /// One entry per staged atom — locals AND ghosts — because the model is
  /// evaluated over the whole graph. Which of them enter the partition energy is
  /// the impl's decision, not this layer's. Mandatory under v3: it is the only
  /// thing a domain-decomposed energy can be summed from, so
  /// `read_model_outputs` refuses a model that does not return it rather than
  /// leaving it undefined for the impl to trip over.
  at::Tensor atomic_energy;
  /// Virial `(6,)` in `xx, yy, zz, xy, xz, yz` order — or **undefined** when the
  /// export declares no virial or the model emitted none. Never defined-but-
  /// malformed: `read_model_outputs` refuses anything that is not exactly six
  /// elements, so a reader may index all six without checking.
  at::Tensor virial;
};

/// What the caller already knows about the outputs it is about to decode.
///
/// Attribute-only, and it exists because `read_model_outputs` needed a second
/// contract fact and the header's own shape guard said how to add one: not as
/// another bool. `read_model_outputs(raw, true, 6)` says nothing at a call
/// site; `expect.n_atoms` does.
struct OutputExpectations {
  /// Whether the export declared a virial output.
  bool supports_virial;
  /// How many atoms were staged, i.e. the length `atomic_energy` must have.
  /// Checked rather than trusted: a per-atom energy of the wrong length would
  /// be sliced to `nlocal` downstream and silently sum the wrong sites.
  int64_t n_atoms;
};

/// Positions of the declared outputs, in `outputs()` order. THE SINGLE
/// DEFINITION POINT — no bare index appears anywhere below.
///
/// v2 was `[energy, forces, (virial)]`; v3 is
/// `[energy, forces, atomic_energy, (virial)]`, so the virial moved from 2 to
/// 3. THIS IS THE MOST DANGEROUS LINE OF THE WHOLE SCHEMA BUMP: a reader left
/// at the old index takes an `(N,)` per-atom energy for a `(6,)` virial. Two
/// guards catch that — `numel == 6` on the virial and `size(0) == n_atoms` on
/// the per-atom energy — but BOTH GO BLIND AT N == 6 SIMULTANEOUSLY, and six
/// atoms is benzene, two waters, and most hand-written fixtures. The value-level
/// ctest `bridge_read_model_outputs_six_atoms_no_transpose` is the only defence
/// at that size; do not remove it, and do not inline these constants.
constexpr std::size_t kEnergyIdx = 0;
constexpr std::size_t kForcesIdx = 1;
constexpr std::size_t kAtomicEnergyIdx = 2;
constexpr std::size_t kVirialIdx = 3;
/// Outputs every export must return, whatever its flags: the three above the
/// virial.
constexpr std::size_t kMandatoryOutputs = 3;

/// Convert a model's raw output vector into `ModelOutputs`, validating it.
///
/// The one place the output-side invariant lives, shared by every `Exec`
/// implementation. A free function rather than a method: it has no state and no
/// configuration, and it is a transformation of a value rather than the
/// construction of some type (hence `read_*`, not `make_*` / `create_*`).
///
/// The three mandatory outputs are read positionally through the named indices
/// above and moved to float64 on the CPU, with `forces` and `atomic_energy`
/// additionally made contiguous.
///
/// The virial is defined **if and only if** `expect.supports_virial` is true,
/// `raw` carries a fourth element, and that element holds exactly six numbers.
/// An absent fourth element is tolerated (the export flag says a virial MAY be
/// tallied, not that one arrives on every call); a present but wrong-length one
/// is refused here rather than at the facade's tally, which is what makes
/// `ModelOutputs::virial` well-formed-or-undefined by construction.
///
/// SHAPE GUARD, HONOURED: this function took two parameters and needed a third
/// contract fact, so the fact arrived inside `OutputExpectations` rather than as
/// another positional bool. Add the next one there too.
///
/// \param raw     Model outputs in declaration order: energy, forces,
///                atomic_energy, and optionally virial.
/// \param expect  What the caller already knows — the virial flag and the atom
///                count `atomic_energy` must match.
/// \returns `ModelOutputs` whose defined members are CPU float64 tensors.
/// \throws std::runtime_error  If `raw` holds fewer than the three mandatory
///                             outputs, if `atomic_energy` does not hold exactly
///                             `expect.n_atoms` entries, or if a virial is
///                             present and expected but does not hold exactly
///                             six elements. Every message states what was
///                             observed alongside what was required.
ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
                                const OutputExpectations& expect);

/// How this model is executed, with the backend kept out of the signature.
///
/// Implementations in this repo: `DynamicExec` (per-step shapes, delegating to
/// `ModelRunner::run`) and `GraphedExec` (fixed shapes with padded edges,
/// delegating to `ModelRunner::run_graphed`). A third, `TorchScriptExec`,
/// is described in the header comment above and needs nothing added here.
///
/// The three inputs every implementation needs to do that — the device, the
/// model dtype and the virial flag — are held HERE, once, because they are the
/// same three facts about the same export and two copies of them could drift.
/// The runner is deliberately NOT among them: which runtime object an
/// implementation borrows is exactly the part this interface refuses to name,
/// so `runner_` stays a member of each subclass and only the "borrowed still
/// means present" refusal is shared, through `require_runner`.
class Exec {
 public:
  virtual ~Exec();

  Exec(const Exec&) = delete;
  Exec& operator=(const Exec&) = delete;

  /// Evaluate the model over one staged configuration.
  ///
  /// \param Z           Atomic numbers `(n,)` int64 on the CPU, as
  ///                    `stage_nodes` produced them.
  /// \param pos_f64     Positions `(n, 3)` float64 on the CPU in angstrom. The
  ///                    cast to the model's own dtype belongs to the
  ///                    implementation — callers never need to know it.
  /// \param edge_index  Cutoff graph `(ne, 2)` int64 on the CPU, as
  ///                    `GraphBridge::build` produced it. Its ROW ORDER IS NOT
  ///                    PROMISED: scatter-class reductions are permutation-
  ///                    invariant, so the builder sorts only when the export
  ///                    declares `requires_sorted_edges`. An implementation
  ///                    that needs an order must say so through the contract,
  ///                    not assume one here.
  /// \param local_mask  Ownership `(n,)` bool on the CPU: true exactly on the
  ///                    atoms this rank owns. The model's energy is the site
  ///                    sum over those, so the mask is what makes a
  ///                    domain-decomposed run add up to one total instead of
  ///                    counting every ghost twice.
  /// \returns `ModelOutputs` whose defined tensors are CPU float64 copies the
  ///          CALLER OWNS. An implementation must not return a tensor that
  ///          aliases runtime-owned or replay-owned memory: `run_graphed`'s
  ///          outputs sit at fixed addresses that the next replay overwrites in
  ///          place, so a caller holding last step's forces would watch them
  ///          change under it. Owning the copy is what lets the facade keep a
  ///          result across the next step without knowing which backend ran.
  /// \throws std::runtime_error  On a shape the implementation's export cannot
  ///                             accept, or on a malformed output (see
  ///                             `read_model_outputs`).
  virtual ModelOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
                           const at::Tensor& edge_index,
                           const at::Tensor& local_mask) = 0;

  /// The model's input dtype as the export meta spells it: `"float32"` or
  /// `"float64"`.
  ///
  /// Read-only, and the single copy of that fact: the execution site is the
  /// only code that acts on the model dtype (it is what `pos_f64` is cast to),
  /// so the facade transcribes this string into its startup message rather than
  /// keeping a second flag that could drift out of step with the cast.
  const std::string& model_dtype_name() const { return model_dtype_name_; }

 protected:
  /// \param device          Device the model's package was built for; every
  ///                        implementation moves its inputs there.
  /// \param model_dtype     The dtype `pos` is cast to before the model sees
  ///                        it. Also held as its name (see
  ///                        `model_dtype_name`), so the string a reader sees
  ///                        and the `at::ScalarType` the cast uses cannot
  ///                        disagree.
  /// \param supports_virial Whether the export declared a virial output; the
  ///                        flag every implementation hands to
  ///                        `read_model_outputs`.
  Exec(at::Device device, at::ScalarType model_dtype, bool supports_virial);

  /// Refuse a null borrowed runner at the line that failed to supply one.
  ///
  /// Takes `const void*` rather than a runner type on purpose: `runner_` stays
  /// a member of each implementation, because the backend it points at is not
  /// this interface's business (the recorded TorchScript route holds a scripted
  /// module, not a `ModelRunner`). What IS shared is the refusal — "borrowed"
  /// still means "present", and without it a null only surfaces as a segfault
  /// inside the first timestep, far from the `pair_style` line that caused it.
  ///
  /// \param runner     The borrowed pointer an implementation was handed.
  /// \param impl_name  Class name for the diagnostic, e.g. `"DynamicExec"`.
  /// \throws std::runtime_error  If `runner` is null.
  static void require_runner(const void* runner, const std::string& impl_name);

  /// Device the implementation moves its inputs to.
  at::Device device_;
  /// Dtype `pos` is cast to before the model sees it.
  at::ScalarType model_dtype_;
  /// Whether the export declared a virial output.
  bool supports_virial_;

 private:
  std::string model_dtype_name_;
};

}  // namespace molnex::lammps_bridge
