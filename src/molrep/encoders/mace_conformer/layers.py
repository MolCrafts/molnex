"""The readouts from MACE invariants onto the force-field heads' atom and bond features.

Spec: ``.claude/specs/rondo-enc-02-mace.md`` §2.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["AtomReadout", "BondReadout"]


class AtomReadout(nn.Module):
    """``MLP[s_i ‖ c_i]`` → the heads' atom width.

    Args:
        scalars: Width of the MACE invariants per atom (every layer's ``0e`` block).
        width: Output width.
        hidden: Hidden width.
        charge_dim: Width of the formal-charge embedding.
        dtype: Floating dtype of every parameter.
    """

    #: Formal charge is clamped to ±2, as ByteFF-26's encoder clamps it.
    MAX_FORMAL_CHARGE = 2

    def __init__(
        self, scalars: int, *, width: int, hidden: int, charge_dim: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.charge = nn.Embedding(2 * self.MAX_FORMAL_CHARGE + 1, charge_dim, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(scalars + charge_dim, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, width, dtype=dtype),
        )

    def forward(self, scalars: torch.Tensor, formal_charge: torch.Tensor) -> torch.Tensor:
        """Atom features ``(N, width)``.

        Args:
            scalars: MACE invariants ``(N, S)``.
            formal_charge: Integer formal charge ``(N,)``.
        """
        index = (formal_charge + self.MAX_FORMAL_CHARGE).clamp(0, 2 * self.MAX_FORMAL_CHARGE)
        return self.mlp(torch.cat((scalars, self.charge(index)), dim=-1))


class BondReadout(nn.Module):
    """``½(MLP[s_i ‖ s_j ‖ φ(r)] + MLP[s_j ‖ s_i ‖ φ(r)])`` + order and ring embeddings.

    Symmetric in the bond's direction by construction. ``φ(r)`` is a Gaussian
    basis of the reference bond length: finite at ``r = 0`` (a dead bond row
    joining two ghosts), unlike a Bessel basis.

    Args:
        scalars: Width of the MACE invariants per atom.
        width: Output width.
        hidden: Hidden width.
        lengths: ``(low, high, count)`` of the Gaussian centres, Å.
        dtype: Floating dtype of every parameter and buffer.
    """

    #: Bond orders single, double, triple, aromatic (``bonds.bond_type`` 1..4).
    NUM_BOND_ORDERS = 4

    def __init__(
        self,
        scalars: int,
        *,
        width: int,
        hidden: int,
        lengths: tuple[float, float, int],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        low, high, count = lengths
        centres = torch.linspace(float(low), float(high), int(count), dtype=dtype)
        self.register_buffer("centres", centres, persistent=False)
        self.centres: torch.Tensor
        #: Gaussian width: the centre spacing.
        self.spread = (float(high) - float(low)) / max(int(count) - 1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * scalars + int(count), hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, width, dtype=dtype),
        )
        self.order = nn.Embedding(self.NUM_BOND_ORDERS, width, dtype=dtype)
        self.in_ring = nn.Embedding(2, width, dtype=dtype)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        length: torch.Tensor,
        order: torch.Tensor,
        in_ring: torch.Tensor,
    ) -> torch.Tensor:
        """Bond features ``(E, width)``.

        Args:
            source: Invariants of each bond's first atom ``(E, S)``.
            target: Invariants of its second atom ``(E, S)``.
            length: Reference bond length ``(E,)``, Å.
            order: Bond order index ``(E,)`` in ``[0, 4)``.
            in_ring: ``(E,)`` 0 or 1.
        """
        basis = torch.exp(-(((length.unsqueeze(-1) - self.centres) / self.spread) ** 2))
        forward = self.mlp(torch.cat((source, target, basis), dim=-1))
        backward = self.mlp(torch.cat((target, source, basis), dim=-1))
        return 0.5 * (forward + backward) + self.order(order) + self.in_ring(in_ring)
