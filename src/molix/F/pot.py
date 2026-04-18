"""Functional API for potential operations (molix).

Current C++ ops registered under ``torch.ops.molix``:

* ``pme_direct(positions, charges, neighbors, deltas, distances,
  exclusions, alpha, coulomb)`` — direct-space Ewald contribution.
* ``pme_reciprocal(positions, charges, box_vectors, gridx, gridy, gridz,
  order, alpha, coulomb, xmoduli, ymoduli, zmoduli)`` — reciprocal-space
  contribution.

The higher-level ``pme_kernel`` façade that used to live here was wired
to a non-existent C++ op; real users of PME should compose
``pme_direct`` + ``pme_reciprocal`` explicitly via the ops library or a
future ``PMESolver`` wrapper.
"""

from __future__ import annotations

from torch import Tensor, ops


def pme_direct(
    positions: Tensor,
    charges: Tensor,
    neighbors: Tensor,
    deltas: Tensor,
    distances: Tensor,
    exclusions: Tensor,
    alpha: float,
    coulomb: float,
) -> Tensor:
    """Direct-space PME contribution."""
    return ops.molix.pme_direct(positions, charges, neighbors, deltas, distances, exclusions, alpha, coulomb)


def pme_reciprocal(
    positions: Tensor,
    charges: Tensor,
    box_vectors: Tensor,
    gridx: int,
    gridy: int,
    gridz: int,
    order: int,
    alpha: float,
    coulomb: float,
    xmoduli: Tensor,
    ymoduli: Tensor,
    zmoduli: Tensor,
) -> Tensor:
    """Reciprocal-space PME contribution."""
    return ops.molix.pme_reciprocal(
        positions, charges, box_vectors, gridx, gridy, gridz, order, alpha, coulomb,
        xmoduli, ymoduli, zmoduli,
    )
