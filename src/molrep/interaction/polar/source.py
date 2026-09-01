"""Environment-dependent spin-resolved multipole source (PolarMACE).

A single ``cuet.Linear`` maps node features onto two concatenated copies of
``Y_{l_max}`` in ``cue.ir_mul``. Polar-1-M uses ``l_max=1`` so the flat output
is ``(N, 8)`` and ``view (N, 2, 4)`` is one spin channel of monopole + dipole.

Do not wrap :class:`~molrep.interaction.linear.EquivariantLinear` — that would
prefix checkpoint keys with ``linear.linear.``.

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn

from molix import config
from molrep.interaction.product import sh_irreps_from_l_max


class EnvironmentDependentSpinSourceBlock(nn.Module):
    """Linear map from hidden irreps to two spin channels of GTO multipoles.

    Args:
        irreps_in: Node-feature irreps, e.g. ``"512x0e+512x1o"``.
        l_max: Highest multipole order (polar-1-M dump: 1).
        zero_charges: If True, zero the monopole (0e) of each spin channel.

    Returns:
        ``(N, 2, (l_max+1)**2)`` — channel 0/1 are spin-up/down; last axis is
        one copy of ``Y_{l_max}`` (monopole then dipole for ``l_max=1``).
        Physical units of the coefficients are assigned by the GTO energy (e / e·Å).
    """

    def __init__(self, *, irreps_in: str, l_max: int, zero_charges: bool = False) -> None:
        super().__init__()
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        self.l_max = l_max
        self.zero_charges = zero_charges
        self.y_dim = (l_max + 1) ** 2
        # Two concatenated Y copies, not simplify(2x0e+2x1o), so view (N, 2, y_dim)
        # lands monopoles at [:, 0, 0] and [:, 1, 0].
        y = sh_irreps_from_l_max(l_max).replace(" ", "")
        irreps_out = f"{y}+{y}"
        self.linear = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", irreps_out),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Map node features to spin-resolved multipole coefficients.

        Args:
            node_feats: ``(N, irreps_in.dim)`` in ``ir_mul``.

        Returns:
            ``(N, 2, (l_max+1)**2)``.
        """
        flat = self.linear(node_feats)
        sources = flat.view(flat.shape[0], 2, self.y_dim)
        if self.zero_charges:
            sources = sources.clone()
            sources[..., 0] = 0
        return sources
