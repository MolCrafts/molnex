// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `exec.h` for the interface's design notes and its two contracts
// (backend-neutral signatures, caller-owned CPU outputs).

#include "bridge/exec.h"

#include <stdexcept>
#include <string>

#include <ATen/ATen.h>

namespace molnex::lammps_bridge {
namespace {

/// The export meta's spelling of a model input dtype.
///
/// Deliberately not `c10::toString` (which yields `"Double"` / `"Float"`): this
/// string is transcribed into the pair style's startup message next to the
/// values the meta.json carried, so it must read the same way the meta does.
std::string meta_dtype_name(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "float32";
    case at::kDouble:
      return "float64";
    default:
      throw std::runtime_error(
          std::string("Exec: unsupported model dtype ") + c10::toString(dtype) +
          "; the export meta declares model_dtype as float32 or float64 and "
          "the execution path casts positions to exactly one of those");
  }
}

}  // namespace

Exec::Exec(at::Device device, at::ScalarType model_dtype, bool supports_virial)
    : device_(device),
      model_dtype_(model_dtype),
      supports_virial_(supports_virial),
      model_dtype_name_(meta_dtype_name(model_dtype)) {}

// Out of line on purpose: an inline pure-virtual destructor would emit the
// class's vtable in every translation unit that includes the header.
Exec::~Exec() = default;

void Exec::require_runner(const void* runner, const std::string& impl_name) {
  if (runner == nullptr) {
    throw std::runtime_error(
        impl_name +
        ": model runner is null; the pair style must load the model before it "
        "constructs an execution strategy");
  }
}

ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
                                const OutputExpectations& expect) {
  // Energy, forces and the per-atom energy are not optional. The migrated code
  // indexed raw[0] and raw[1] unchecked, so a model exported with a single
  // output turned into an out-of-range std::vector read rather than a
  // diagnostic; the bound is stated here because this is the line that reads
  // them. `atomic_energy` joined the mandatory set with schema v3 — it is what
  // a domain-decomposed energy is summed from, so a model without it cannot be
  // run at all rather than run to a silently zero energy.
  if (raw.size() < kMandatoryOutputs) {
    throw std::runtime_error(
        "read_model_outputs: model returned " + std::to_string(raw.size()) +
        " output(s); at least " + std::to_string(kMandatoryOutputs) +
        " are required (energy, forces, atomic_energy). A model returning only "
        "energy and forces is a pre-v3 export: re-export it with "
        "molix.engine.export_for_lammps");
  }

  ModelOutputs out;
  // `.to()` is a no-op when the tensor already has the requested dtype and
  // device, so this chain is a conversion, not an unconditional copy. That is
  // sufficient for the caller-owned contract in `exec.h` on every path that
  // exists: `run_graphed`'s replay buffers are CUDA-resident, so `.to(kCPU)`
  // always copies them, and `run`'s outputs are freshly allocated per call.
  // Kept verbatim from the two facade branches it replaces — forcing a further
  // clone would add a per-step copy to a migration that changes no numbers.
  out.energy = raw[kEnergyIdx].to(at::kDouble).to(at::kCPU);
  out.forces = raw[kForcesIdx].to(at::kDouble).to(at::kCPU).contiguous();
  out.atomic_energy =
      raw[kAtomicEnergyIdx].to(at::kDouble).to(at::kCPU).contiguous();

  // One entry per staged atom. The impl slices the first `nlocal` of these and
  // hands them to LAMMPS as the per-atom energy, so a length that does not match
  // what was staged would sum the wrong sites — with a plausible number coming
  // out the other end. Checked HERE, in the single decoder, rather than in each
  // Exec: the two implementations do device and dtype movement, not contract
  // validation.
  const int64_t site_count = out.atomic_energy.size(0);
  if (out.atomic_energy.dim() != 1 || site_count != expect.n_atoms) {
    throw std::runtime_error(
        "read_model_outputs: atomic_energy has " + std::to_string(site_count) +
        " entries for " + std::to_string(expect.n_atoms) +
        " staged atom(s); the per-atom energy must carry exactly one value per "
        "atom the model was given, because the partition energy is its sum over "
        "the locally owned ones");
  }

  // Tolerant about absence, strict about shape. `supports_virial` says the
  // facade MAY tally a virial, not that one arrives every call — hence the
  // size check. What is new is where the length check lives: the facade used to
  // receive a defined virial and then guard against reading six elements out of
  // it, which asked the protocol layer to clean up after the execution layer.
  //
  // Note the index: v3 put `atomic_energy` ahead of the virial, so this reads
  // `raw[3]`. Reading `raw[2]` here would take the per-atom energy for a
  // virial, and at N == 6 both length guards accept the swap — which is why
  // the six-atom value-level ctest exists.
  if (expect.supports_virial && raw.size() > kVirialIdx) {
    const int64_t numel = raw[kVirialIdx].numel();
    if (numel != 6) {
      throw std::runtime_error(
          "read_model_outputs: supports_virial but the virial output has " +
          std::to_string(numel) +
          " elements (need exactly 6: xx, yy, zz, xy, xz, yz)");
    }
    out.virial = raw[kVirialIdx].to(at::kDouble).to(at::kCPU).contiguous();
  }

  return out;
}

}  // namespace molnex::lammps_bridge
