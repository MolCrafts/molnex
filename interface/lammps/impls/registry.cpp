// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// See `registry.h` for the four rules this dispatch keeps (exhaustive switch,
// no table, public constructors, loud refusal). This translation unit also
// carries `ContractImpl`'s out-of-line destructor, so the base's vtable is
// emitted once here rather than in every file that includes the header.

#include "impls/registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "bridge/dynamic_exec.h"
#include "bridge/exec.h"
#include "bridge/graphed_exec.h"
#include "impls/cutoff_graph_impl.h"
#include "molnex/interface/model_runner.h"

namespace molnex::lammps {
namespace {

/// The device the loaded package was built for.
///
/// Read from the RUNNER, not from the contract: the `lammps` block says nothing
/// about where the model was exported to — that is the meta's top-level
/// `device` field, which `ModelRunner` already parsed and cross-checked against
/// the package it loaded. Asking the runner is what keeps one answer to that
/// question in the process.
at::Device device_of(const molnex::interface::ModelRunner& runner) {
  return runner.device() == "cuda" ? at::Device(at::kCUDA)
                                   : at::Device(at::kCPU);
}

/// The scalar type the export's `model_dtype` names.
///
/// Same reading as the pre-refactor facade: `"float64"` is double, anything
/// else is float. `Exec`'s constructor is what rejects a dtype it cannot cast
/// to, and it names the meta field when it does.
at::ScalarType dtype_of(const LammpsContract& contract) {
  return contract.model_dtype() == "float64" ? at::kDouble : at::kFloat;
}

/// The execution strategy the export asks for.
///
/// Same shape as the contract switch below — explicit case per enumerator, no
/// `default:` — so a third `Execution` value (the recorded `torchscript` route)
/// is a compile-time error here too. Kept file-local: it has exactly one call
/// site and no test targets it, so extracting it into the header would be a
/// symbol with nothing to justify it.
///
/// The two cuda-graph preconditions live in the `cuda_graph` case rather than
/// in `CutoffGraphImpl::init`, because they are statements about THIS EXPORT
/// against THIS PROCESS — the meta's declared strategy against the device the
/// runner actually loaded, and against a capability that makes the captured
/// shapes impossible — and both inputs are in hand exactly here. They used to
/// sit in the facade (pair_molnex.cpp, pre-refactor), where they were two of
/// the semantic `if`s that had to leave; their text is carried over word for
/// word, because that text is what a user greps for.
///
/// \param contract  Validated export contract.
/// \param runner    Loaded model runner, borrowed by the returned strategy for
///                  its whole lifetime.
/// \returns The strategy, never null.
/// \throws std::runtime_error  If the strategy refuses this build or this meta
///                             (a cuda-graph export loaded onto a CPU device or
///                             declaring supports_pbc, a cuda-graph export on a
///                             CPU-only build, a dtype that is neither float32
///                             nor float64, ...).
std::unique_ptr<molnex::lammps_bridge::Exec> exec_for(
    const LammpsContract& contract, molnex::interface::ModelRunner& runner) {
  switch (contract.execution()) {
    case Execution::dynamic:
      return std::make_unique<molnex::lammps_bridge::DynamicExec>(
          &runner, device_of(runner), dtype_of(contract),
          contract.supports_virial());
    case Execution::cuda_graph:
      // A capture replays a fixed sequence of GPU kernels; there is no such
      // thing on a CPU package, and the runner is the one authority on where
      // the package it loaded lives.
      if (runner.device() != "cuda") {
        throw std::runtime_error(
            "pair_style molnex: meta execution=cuda-graph but device=" +
            runner.device() + " (need cuda)");
      }
      // Ghosts and a capture are mutually exclusive by construction: the ghost
      // count changes as atoms move between domains, while the captured graph
      // is pinned to one N. Refused while loading the model rather than at the
      // first step that grows a ghost.
      if (contract.supports_pbc()) {
        throw std::runtime_error(
            "pair_style molnex: execution=cuda-graph is incompatible with "
            "supports_pbc (ghost count varies; re-export with cuda_graph=False "
            "for PBC MD)");
      }
      return std::make_unique<molnex::lammps_bridge::GraphedExec>(
          &runner, device_of(runner), dtype_of(contract),
          contract.supports_virial(), contract.n_atoms(), contract.e_max());
  }
  // Unreachable: every enumerator returns above, and -Wswitch is what keeps
  // that true as the enum grows. Present only for the return-path analysis.
  throw std::runtime_error(
      "molnex: unhandled execution strategy; this is a bug in the plugin's "
      "dispatch, not in the export");
}

}  // namespace

// Out of line on purpose: an inline pure-virtual destructor would emit the
// class's vtable in every translation unit that includes contract_impl.h.
ContractImpl::~ContractImpl() = default;

std::unique_ptr<ContractImpl> ContractImpl::for_contract(
    Contract contract_kind, const LammpsContract& contract,
    const std::string& lammps_unit_style,
    molnex::interface::ModelRunner& runner) {
  switch (contract_kind) {
    case Contract::cutoff_graph:
      // The public constructor, called directly. The dispatch's only job is
      // choosing WHICH one and building the strategy it takes.
      return std::make_unique<CutoffGraphImpl>(contract, lammps_unit_style,
                                               exec_for(contract, runner));
    case Contract::fixed_graph:
      // A loud refusal, listing every contract that does dispatch. Running a
      // fixed-topology export down the cutoff path would substitute a different
      // edge set — a `(2, N)` COO covalent topology is not the `(E, 2)` cutoff
      // neighbour graph — and so different physics, with no diagnostic.
      throw std::runtime_error(
          "pair_style molnex: contract 'fixed-graph' has no implementation yet "
          "(this pair style builds its edge set from a cutoff; a fixed "
          "covalent "
          "topology needs an impl that does not exist); valid contracts: "
          "cutoff-graph");
  }
  // Unreachable, as above: -Wswitch guarantees every enumerator is handled.
  throw std::runtime_error(
      "molnex: unhandled contract; this is a bug in the plugin's dispatch, not "
      "in the export");
}

}  // namespace molnex::lammps
