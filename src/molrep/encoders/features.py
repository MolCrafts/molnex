"""What every encoder gives the force-field heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["EncoderFeatures"]


@dataclass
class EncoderFeatures:
    """What every encoder gives the force-field heads: one row per atom and per bond.

    The interface between an encoder and the heads (spec
    ``.claude/specs/rondo-enc-01-graph-encoder.md`` §2). Angle, torsion and improper
    features are not here: the heads assemble them from these rows with the
    symmetry each term needs, so a head never depends on which encoder ran.

    Attributes:
        atom: Per-atom features ``(N, D)`` in the batch atom order.
        bond: Per-bond features ``(N_bonds, D)`` in ``bonds.bond_index`` order,
            the same for both directions of a bond. Rows of topologically
            equivalent atoms (bonds) are equal.
    """

    atom: torch.Tensor
    bond: torch.Tensor
