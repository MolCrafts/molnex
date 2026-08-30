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
  at::Tensor energy;
  /// Per-node forces `(n, 3)`, contiguous float64 on the CPU, in model units.
  at::Tensor forces;
  /// Virial `(6,)` in `xx, yy, zz, xy, xz, yz` order — or **undefined** when the
  /// export declares no virial or the model emitted none. Never defined-but-
  /// malformed: `read_model_outputs` refuses anything that is not exactly six
  /// elements, so a reader may index all six without checking.
  at::Tensor virial;
};

/// Convert a model's raw output vector into `ModelOutputs`, validating it.
///
/// The one place the output-side invariant lives, shared by every `Exec`
/// implementation. A free function rather than a method: it has no state and no
/// configuration, and it is a transformation of a value rather than the
/// construction of some type (hence `read_*`, not `make_*` / `create_*`).
///
/// `raw[0]` becomes `energy` and `raw[1]` becomes `forces`, both moved to
/// float64 on the CPU with `forces` additionally made contiguous — the exact
/// conversion the migrated facade did inline in each of its two execution
/// branches.
///
/// The virial is defined **if and only if** `supports_virial` is true, `raw`
/// carries a third element, and that element holds exactly six numbers. An
/// absent third element is tolerated (the export flag says a virial MAY be
/// tallied, not that one arrives on every call); a present but wrong-length one
/// is refused here rather than at the facade's tally, which is what makes
/// `ModelOutputs::virial` well-formed-or-undefined by construction.
///
/// SHAPE GUARD: this function currently takes two parameters. If a third
/// contract switch ever needs to reach it, it should arrive as a small
/// expectation struct — not as another bool. Two booleans at a call site are
/// already unreadable, and each one added multiplies the paths this single
/// invariant has to hold across.
///
/// \param raw             Model outputs in declaration order: energy, forces,
///                        and optionally virial.
/// \param supports_virial Whether the export declared a virial output.
/// \returns `ModelOutputs` whose defined members are CPU float64 tensors.
/// \throws std::runtime_error  If `raw` holds fewer than the two mandatory
///                             outputs, or if a virial is present and expected
///                             but does not hold exactly six elements (the
///                             message states the observed count).
ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
                                bool supports_virial);

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
  /// \param edge_index  Cutoff graph `(ne, 2)` int64 on the CPU, sorted by
  ///                    `(source, target)`, as `GraphBridge::build` produced it.
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
                           const at::Tensor& edge_index) = 0;

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
