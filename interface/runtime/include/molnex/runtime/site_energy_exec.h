// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// The site-energy calling convention, executed over any `ModelArtifact`.
//
// The convention is `molix.deploy.SiteEnergyModel` (metadata schema 2):
//
//     (Z, pos, pair_index, shifts, local_mask)
//         -> (energy, forces, site_energy, virial)
//
// An edge's displacement is `pos[target] - pos[source] + shifts`. `energy` is
// the sum of the site energies `local_mask` selects, `forces` its negative
// gradient with respect to every position, masked-out sites included, and
// `virial` its derivative with respect to a homogeneous strain, `(3, 3)`,
// molnex's sign (`+dE/deps`; an engine whose convention is `-dE/deps`
// negates). An engine that stages ghost copies passes zero shifts; one that
// folds periodic images onto their owners passes the lattice translations.
// Everything an engine hands in is a CPU tensor: Z int64, pos and shifts
// float64, pair_index int64 `(E, 2)` (column 0 the source, column 1 the
// target), local_mask bool. Everything it gets back is a CPU float64 copy it
// owns. Device transfer and the model-dtype cast happen here, once.

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>

#include "molnex/runtime/model_artifact.h"

namespace molnex::runtime {

/// What one evaluation produced. Every member is a CPU float64 tensor the
/// caller owns, in the model's own units.
struct SiteEnergyOutputs {
  /// The masked site-energy sum, a single element.
  at::Tensor energy;
  /// Forces `(n, 3)`, contiguous, for every staged atom.
  at::Tensor forces;
  /// Site energies `(n,)`, contiguous, one per staged atom (masked or not).
  at::Tensor site_energy;
  /// `d(energy)/d(strain)` `(3, 3)`, contiguous, symmetric.
  at::Tensor virial;
};

/// Runs a site-energy model.
///
/// Example:
/// \code
///   SiteEnergyExec exec(*artifact, metadata.dtype);
///   const SiteEnergyOutputs out =
///       exec.run(Z, pos, pair_index, shifts, local_mask);
/// \endcode
class SiteEnergyExec {
 public:
  /// \param artifact  The loaded package. Borrowed: it must outlive this.
  /// \param dtype     The model's input dtype; `pos` is cast to it.
  SiteEnergyExec(ModelArtifact& artifact, at::ScalarType dtype);

  /// Evaluate one staged configuration.
  ///
  /// \param Z           Atomic numbers `(n,)` int64, CPU.
  /// \param pos_f64     Positions `(n, 3)` float64, CPU, in the model's
  ///                    length unit.
  /// \param pair_index  Pairs `(E, 2)` int64, CPU.
  /// \param shifts_f64  Lattice translation of each pair's target `(E, 3)`
  ///                    float64, CPU, in the model's length unit.
  /// \param local_mask  `(n,)` bool, CPU: the sites whose energy is summed.
  /// \throws std::runtime_error  If the package does not return exactly four
  ///                             tensors, `forces` is not `(n, 3)`,
  ///                             `site_energy` is not `(n,)`, or `virial` is
  ///                             not `(3, 3)`.
  SiteEnergyOutputs run(const at::Tensor& Z, const at::Tensor& pos_f64,
                        const at::Tensor& pair_index,
                        const at::Tensor& shifts_f64,
                        const at::Tensor& local_mask);

 private:
  ModelArtifact* artifact_;
  at::ScalarType dtype_;
};

}  // namespace molnex::runtime
