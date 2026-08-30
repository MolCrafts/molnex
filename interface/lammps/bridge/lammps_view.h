// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The one type isolation layer between LAMMPS and the molnex bridge.
//
// Every other file under `bridge/` takes its atom data through this struct and
// therefore includes **no LAMMPS header** — not `atom.h`, not `neigh_list.h`,
// not `pair.h`. That is what makes the bridge components drivable by ctest on
// a machine with no LAMMPS build at all: a test hands the same row-pointer
// shape out of its own std::vector storage that LAMMPS hands out of `atom->x`.
//
// The view **borrows** LAMMPS-owned memory and owns nothing: no allocation, no
// deallocation, no lifetime extension. It is valid only for as long as the
// arrays it points at stay put, which for LAMMPS means "until the next
// reneighbouring / grow" — so build one per `compute()` call rather than
// caching it across steps.
//
// Field set follows the consumers that exist today: there is no `tag` (the
// tag<->local mapping belongs to a future fixed-graph contract), no `q` (no
// reader) and no neighbour spans. `lammps-iface-03-graph` adds the
// `ilist` / `numneigh` / `firstneigh` spans plus a `neigh_mask` member when
// `GraphBridge` — their only consumer — arrives.

#pragma once

namespace molnex::lammps_bridge {

/// Borrowed, non-owning view of the LAMMPS per-atom arrays the bridge reads.
///
/// A POD aggregate on purpose: no user-provided constructor, so the facade (and
/// a test) writes `LammpsView{x, f, type, nlocal, nghost}` at the seam and
/// nothing downstream can accidentally take ownership. `double**` converts to
/// `const double* const*` as a qualification conversion, so the facade needs no
/// `const_cast` to hand over `atom->x`.
///
/// All arrays are indexed by *local* atom index in `[0, nall())`, with the
/// first `nlocal` entries owned by this rank and the remainder ghosts.
struct LammpsView {
  /// Positions, `(nall, 3)` in angstrom: `atom->x`, read-only through the view.
  const double* const* x;
  /// Forces, `(nall, 3)`: `atom->f`, writable — this is where a pair style
  /// accumulates its contribution.
  double* const* f;
  /// Per-atom LAMMPS types, `(nall,)`, **1-based**: `atom->type`.
  const int* type;
  /// Number of atoms owned by this rank.
  int nlocal;
  /// Number of ghost atoms (periodic images and atoms owned by other ranks).
  int nghost;

  /// Total number of addressable atoms: owned plus ghosts.
  ///
  /// \returns `nlocal + nghost`, the exclusive upper bound of every array above.
  int nall() const { return nlocal + nghost; }
};

}  // namespace molnex::lammps_bridge
