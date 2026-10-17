// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Model forces back onto the LAMMPS force array, in LAMMPS units.
//
// A type rather than a free function because it has one piece of state with a
// lifetime of its own: `force_conv` is fixed once in `pair_style` settings from
// the model's unit contract and reused every step — construct -> configure ->
// one concern.
//
// The scatter **accumulates**. LAMMPS' `atom->f` already carries every earlier
// pair style's, bond style's and fix's contribution when a pair style runs, and
// `virial_fdotr_compute` reads the same array afterwards; an overwrite would
// silently delete all of it.

#pragma once

#include <ATen/Tensor.h>

#include "bridge/lammps_view.h"

namespace molnex::lammps_bridge {

/// Adds a model's forces onto the LAMMPS force array with unit conversion.
class ForceScatter {
 public:
  /// \param force_conv  Model force unit expressed in the LAMMPS unit style,
  ///                    e.g. 23.060549 for eV/angstrom -> kcal/mol/angstrom.
  ///                    Equal to the energy factor, which holds only because
  ///                    `CutoffGraphImpl` asserts the model length unit is
  ///                    angstrom.
  explicit ForceScatter(double force_conv);

  /// Accumulate `forces` onto the first `n_graph` rows of `view.f`.
  ///
  /// Exactly `f[i][k] += force_conv * forces[i][k]`, for `i` in `[0, n_graph)`
  /// and `k` in `[0, 3)`. Rows past `n_graph` — the ghosts of a local-only
  /// export — are left untouched.
  ///
  /// \param view     Borrowed LAMMPS atom arrays; `f` is written in place.
  /// \param forces   Model forces, `(n_graph, 3)` float64 on the CPU.
  /// \param n_graph  Number of leading atoms the model was run on.
  /// \throws std::runtime_error  If `n_graph` is outside `[0, view.nall()]`, or
  ///                             if `forces.size(0) != n_graph`. Both fire
  ///                             before any write, because `view.f` belongs to
  ///                             LAMMPS: an unchecked overrun corrupts the heap
  ///                             and surfaces somewhere unrelated.
  void scatter(const LammpsView& view, const at::Tensor& forces,
               int n_graph) const;

 private:
  double force_conv_;
};

}  // namespace molnex::lammps_bridge
