// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Ghost-force reverse communication: the pack / unpack halves of the LAMMPS
// `comm->reverse_comm` protocol, expressed over a `LammpsView`.
//
// Two free functions, not a type: zero state, zero configuration. A class here
// would be a function in a wrapper, and — worse — would invite a private
// contribution buffer nothing needs.
//
// These functions pack and unpack the **shared** `atom->f` — deliberately.
// Reduction ownership (settled by `lammps-iface-05-pbc`): the integrator's
// *global* reverse comm inside `Verlet::run` delivers ghost forces to their
// owners, so a pair-level `comm->reverse_comm(this)` on top would double-count
// them under `newton on`. `CutoffGraphImpl` therefore returns a constant
// `EvalResult::needs_pair_reverse_comm = false` and the facade's call site
// stays dormant. The seam — these two functions plus the facade forwarding —
// is kept for a future contract family that must reduce mid-compute; such an
// impl owns the newton bookkeeping that makes flipping the flag correct.
//
// Range validation is likewise absent on purpose. `n`, `first` and `list` come
// from LAMMPS' comm layer, which owns those ranges and derives them from the
// same atom count as `view.f`; adding a throw here would change behaviour, and
// this relocation changes none.

#pragma once

#include "bridge/lammps_view.h"

namespace molnex::lammps_bridge {

/// Pack the forces of the contiguous atom range `[first, first + n)` into
/// `buf`.
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
