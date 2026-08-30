// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Ghost-force reverse communication: the pack / unpack halves of the LAMMPS
// `comm->reverse_comm` protocol, expressed over a `LammpsView`.
//
// Two free functions, not a type: zero state, zero configuration. A class here
// would be a function in a wrapper, and — worse — would invite the private
// buffer that must NOT exist (see the defect note below).
//
// KNOWN DEFECT, DELIBERATELY PRESERVED VERBATIM (routed to
// `lammps-iface-05-pbc`). These functions pack and unpack the **shared**
// `atom->f`, not a contribution buffer this pair style owns. Combined with the
// pair-level `comm->reverse_comm(this)` call at `pair_molnex.cpp:351-352` (the
// pack itself is `:359-382` in the spec's line references), that
// double-communicates ghost contributions when `force->newton` is on:
// `Verlet::run` issues a *global* reverse comm inside the integration step and
// delivers the same ghost force to its owner a second time, so forces come out
// too large by an amount that scales with the ghost count. Either half alone is
// harmless; it is the pair of them that is the bug, which is why the decision
// belongs to one spec (`lammps-iface-05-pbc`, whose fix is to drop the
// pair-level self-call and let the integrator's global reverse comm do the
// reduction) and not to this relocation. `lammps-iface-03-graph` moves the code
// unchanged: introducing a private ghost-force buffer here would smuggle that
// fix into a behaviour-preserving migration, and the component is therefore
// named for what it does today (`comm_reverse`, not `CommBuf`) so the name
// endorses no repair that has not happened.
//
// Range validation is likewise absent on purpose. `n`, `first` and `list` come
// from LAMMPS' comm layer, which owns those ranges and derives them from the
// same atom count as `view.f`; adding a throw here would change behaviour, and
// this relocation changes none.

#pragma once

#include "bridge/lammps_view.h"

namespace molnex::lammps_bridge {

/// Pack the forces of the contiguous atom range `[first, first + n)` into `buf`.
///
/// Writes `fx, fy, fz` per atom, atoms in ascending index order, and reports
/// how many doubles that was — the return value LAMMPS' comm layer uses to
/// advance its own cursor.
///
/// \param view   Borrowed LAMMPS atom arrays; `f` is read, never written.
/// \param n      Number of atoms in the range.
/// \param first  Index of the first atom of the range.
/// \param buf    Destination, LAMMPS-owned, with room for at least `3 * n`
///               doubles.
/// \returns `3 * n`, the number of doubles written.
int pack_reverse_forces(const LammpsView& view, int n, int first, double* buf);

/// Accumulate `buf` onto the forces of the atoms named by `list`.
///
/// Exactly `f[list[i]][k] += buf[3 * i + k]`. Accumulation, never assignment:
/// `atom->f` already carries every other contribution to those atoms.
///
/// \param view  Borrowed LAMMPS atom arrays; `f` is written in place.
/// \param n     Number of atoms in `list`.
/// \param list  Local indices of the receiving atoms, `(n,)`.
/// \param buf   Source, `(3 * n,)` doubles in `fx, fy, fz` order per atom.
void unpack_reverse_forces(const LammpsView& view, int n, const int* list,
                           const double* buf);

}  // namespace molnex::lammps_bridge
