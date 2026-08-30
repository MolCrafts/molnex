// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `exec.h` for the interface's design notes and its two contracts
// (backend-neutral signatures, caller-owned CPU outputs).

#include "bridge/exec.h"

#include <stdexcept>
#include <string>
#include <utility>

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

Exec::Exec(at::ScalarType model_dtype)
    : model_dtype_name_(meta_dtype_name(model_dtype)) {}

// Out of line on purpose: an inline pure-virtual destructor would emit the
// class's vtable in every translation unit that includes the header.
Exec::~Exec() = default;

ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw,
                                bool supports_virial) {
  // Energy and forces are not optional. The migrated code indexed raw[0] and
  // raw[1] unchecked, so a model exported with a single output turned into an
  // out-of-range std::vector read rather than a diagnostic; the bound is stated
  // here because this is the line that reads them.
  if (raw.size() < 2) {
    throw std::runtime_error(
        "read_model_outputs: model returned " + std::to_string(raw.size()) +
        " output(s); at least 2 are required (energy, forces)");
  }

  ModelOutputs out;
  // `.to()` is a no-op when the tensor already has the requested dtype and
  // device, so this chain is a conversion, not an unconditional copy. That is
  // sufficient for the caller-owned contract in `exec.h` on every path that
  // exists: `run_graphed`'s replay buffers are CUDA-resident, so `.to(kCPU)`
  // always copies them, and `run`'s outputs are freshly allocated per call.
  // Kept verbatim from the two facade branches it replaces — forcing a further
  // clone would add a per-step copy to a migration that changes no numbers.
  out.energy = raw[0].to(at::kDouble).to(at::kCPU);
  out.forces = raw[1].to(at::kDouble).to(at::kCPU).contiguous();

  // Tolerant about absence, strict about shape. `supports_virial` says the
  // facade MAY tally a virial, not that one arrives every call — hence the
  // size check, which reproduces the migrated `out.size() >= 3`. What is new is
  // where the length check lives: the facade used to receive a defined virial
  // and then guard against reading six elements out of it, which asked the
  // protocol layer to clean up after the execution layer.
  if (supports_virial && raw.size() >= 3) {
    const int64_t numel = raw[2].numel();
    if (numel != 6) {
      throw std::runtime_error(
          "read_model_outputs: supports_virial but the virial output has " +
          std::to_string(numel) +
          " elements (need exactly 6: xx, yy, zz, xy, xz, yz)");
    }
    out.virial = raw[2].to(at::kDouble).to(at::kCPU).contiguous();
  }

  return out;
}

}  // namespace molnex::lammps_bridge
