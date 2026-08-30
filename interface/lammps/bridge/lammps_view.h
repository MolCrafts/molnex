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
// Field set follows the consumers that exist today: the neighbour spans
// (`ilist` / `numneigh` / `firstneigh` / `inum` / `neigh_mask`) arrived with
// `lammps-iface-03-graph` because `GraphBridge` — their only consumer — did.
// There is still no `tag` (the tag<->local mapping belongs to a future
// fixed-graph contract) and no `q` (no reader).
//
// The neighbour mask travels as **data**, applied by the `neighbor()` accessor:
// that is what keeps the bridge from restating a LAMMPS constant. The facade
// fills the member from LAMMPS' own value at the seam.

#pragma once

namespace molnex::lammps_bridge {

/// Borrowed, non-owning view of the LAMMPS per-atom arrays the bridge reads.
///
/// A POD aggregate on purpose: no user-provided constructor, so the facade (and
/// a test) writes `LammpsView{x, f, type, nlocal, nghost, ilist, numneigh,
/// firstneigh, inum, neigh_mask}` at the seam and nothing downstream can
/// accidentally take ownership. `double**` converts to `const double* const*`
/// as a qualification conversion, so the facade needs no `const_cast` to hand
/// over `atom->x`. A caller with no neighbour data to offer may brace-initialise
/// the atom fields alone; the neighbour members then value-initialise, and only
/// the graph path reads them.
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

  /// Local indices of the atoms to evaluate, `(inum,)`: `NeighList::ilist`.
  ///
  /// Under the `REQ_FULL` request this pair style makes, LAMMPS lists only
  /// atoms in `[0, nlocal)` here, so every centre is owned by this rank.
  const int* ilist;
  /// Neighbour counts, indexed by *local atom index* (not by `ilist` position):
  /// `NeighList::numneigh`.
  const int* numneigh;
  /// Neighbour lists, indexed by local atom index: `NeighList::firstneigh`,
  /// with `firstneigh[i]` spanning `numneigh[i]` entries. `int**` converts to
  /// `const int* const*` as a qualification conversion, so the facade hands
  /// over `list->firstneigh` with no `const_cast`.
  const int* const* firstneigh;
  /// Number of entries in `ilist`.
  int inum;
  /// Mask stripping the high flag bits LAMMPS packs into a neighbour entry.
  ///
  /// The value is LAMMPS' own, filled in by the facade — it is deliberately
  /// **data** rather than a constant restated here, which is what keeps every
  /// file under `bridge/` free of LAMMPS vocabulary.
  int neigh_mask;

  /// Total number of addressable atoms: owned plus ghosts.
  ///
  /// \returns `nlocal + nghost`, the exclusive upper bound of every array above.
  int nall() const { return nlocal + nghost; }

  /// The `jj`-th neighbour of local atom `i`, with the flag bits stripped.
  ///
  /// \param i   Local atom index, as taken from `ilist`.
  /// \param jj  Position within that atom's neighbour list, in
  ///            `[0, numneigh[i])`.
  /// \returns   `firstneigh[i][jj] & neigh_mask`: a local atom index in
  ///            `[0, nall())`.
  int neighbor(int i, int jj) const { return firstneigh[i][jj] & neigh_mask; }
};

}  // namespace molnex::lammps_bridge
