// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The contract dispatch's home: include this to reach
// `ContractImpl::for_contract`.
//
// The declaration itself is a static member of `ContractImpl` (contract_impl.h)
// — a caller that has the base type has the dispatch — and this header is the
// thin include that names where the DEFINITION lives, so a reader who wants the
// closed-set rules below finds one file rather than grepping for them. It adds
// no symbol of its own; `registry.cpp` is the translation unit.
//
// THE RULES THAT DEFINITION MUST KEEP, recorded here because they are the whole
// reason the dispatch is a `switch` rather than a table:
//
//   1. EXHAUSTIVE `switch`, one explicit `case` per enumerator, and NO
//      `default:` label. Adding a `Contract` (or `Execution`) enumerator then
//      fails to compile under `-Wswitch` — at the point that must handle it,
//      before anything ships. A `std::map` would instead give one runtime miss,
//      on someone's machine, in the middle of a run. The unreachable `throw`
//      after each switch exists only to satisfy the compiler's return-path
//      analysis, and must stay unreachable.
//   2. No function-local static table and no self-registration. The set of
//      contracts is closed and short; a registry that discovers itself would
//      make the miss message unable to enumerate what it missed.
//   3. The dispatch calls each implementation's PUBLIC constructor. Direct
//      construction remains the primary constructor story (CLAUDE.md), which is
//      why the ctest cases construct `CutoffGraphImpl` themselves and only the
//      two `registry_*` cases come through here. No symbol named `make_*` or
//      `create_*` is introduced by this layer.
//   4. A contract with no implementation is a LOUD refusal that lists every
//      contract that does dispatch (the message style of
//      `src/molix/engine/adapter.py`'s "valid adapters: ..."). Falling back to
//      the cutoff path would silently substitute a different edge set, and so
//      different physics.
//
// Like everything else under `impls/`, this file includes no LAMMPS header.

#ifndef MOLNEX_LAMMPS_IMPLS_REGISTRY_H
#define MOLNEX_LAMMPS_IMPLS_REGISTRY_H

#include "impls/contract_impl.h"

#endif  // MOLNEX_LAMMPS_IMPLS_REGISTRY_H
