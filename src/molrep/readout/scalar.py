"""Non-linear scalar readout head (MACE ``NonLinearBiasReadoutBlock``).

Maps per-atom equivariant features to a scalar (energy) via:
``Linear → SiLU → o3.Linear(+bias) → SiLU → o3.Linear(+bias)``. The first
linear is an equivariant ``cuet.Linear`` projecting to ``MLP_irreps`` scalars;
the two subsequent biased linears act on scalars only and reproduce e3nn's
``o3.Linear`` normalisation (``out = (x @ W.reshape(in,out)) / sqrt(in) + b``).
The SiLU activations carry e3nn's ``normalize2mom`` scaling.

Sub-layer names mirror MACE (``linear_1``, ``linear_mid``, ``linear_2``) so the
official weights transfer by direct copy.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import math

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config
from molrep.embedding.mlp import normalize2mom


class _ScalarO3Linear(nn.Module):
    """Scalar-only e3nn ``o3.Linear`` with bias: ``(x @ W/sqrt(in)) + b``."""

    def __init__(self, in_mul: int, out_mul: int) -> None:
        super().__init__()
        self.in_mul, self.out_mul = in_mul, out_mul
        self.weight = nn.Parameter(torch.zeros(in_mul * out_mul, dtype=config.ftype))
        self.bias = nn.Parameter(torch.zeros(out_mul, dtype=config.ftype))
        self._alpha = 1.0 / math.sqrt(in_mul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(self.in_mul, self.out_mul)
        return self._alpha * (x @ w) + self.bias


class NonLinearBiasReadout(nn.Module):
    """Gated non-linear scalar readout to per-atom energy (single head).

    Args:
        irreps_in: Input node feature irreps (last-layer product output).
        mlp_dim: Hidden scalar width (``MLP_irreps`` count), e.g. 16.
    """

    def __init__(self, *, irreps_in: str, mlp_dim: int = 16) -> None:
        super().__init__()
        self.linear_1 = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self._act_cst = normalize2mom(F.silu)
        self.linear_mid = _ScalarO3Linear(mlp_dim, mlp_dim)
        self.linear_2 = _ScalarO3Linear(mlp_dim, 1)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * self._act_cst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-atom scalar energy.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom energy ``(N, 1)``.
        """
        x = self._act(self.linear_1(x))
        x = self._act(self.linear_mid(x))
        return self.linear_2(x)


class LinearReadout(nn.Module):
    """Equivariant linear projection of node features to a per-atom scalar.

    MACE's ``LinearReadoutBlock``: one ``cuet.Linear`` onto ``1x0e``. Used for
    every interaction layer except the last, where the non-linear readout
    (:class:`NonLinearReadout`) takes over.

    Args:
        irreps_in: Input node feature irreps, e.g. ``"128x0e+128x1o"``.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, *, irreps_in: str) -> None:
        super().__init__()
        self.linear = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project node features to a per-atom scalar.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom scalar ``(N, 1)``.
        """
        return self.linear(x)


class NonLinearReadout(nn.Module):
    """Bias-free gated non-linear scalar readout (MACE ``NonLinearReadoutBlock``).

    ``Linear → c·SiLU → Linear``, both equivariant ``cuet.Linear`` layers with no
    bias; ``c`` is e3nn's moment normalisation constant. This is the readout on
    the **last** interaction layer of the MACE-MP / MatPES foundation models.

    Distinct from :class:`NonLinearBiasReadout`, which is the OMOL variant with
    biases and an extra middle layer — do not substitute one for the other, the
    weight layouts differ.

    Args:
        irreps_in: Input node feature irreps (last-layer product output).
        mlp_dim: Hidden scalar width (MACE's ``MLP_irreps``), e.g. 16.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, *, irreps_in: str, mlp_dim: int = 16) -> None:
        super().__init__()
        self.linear_1 = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self.linear_2 = cuet.Linear(
            cue.Irreps("O3", f"{mlp_dim}x0e"),
            cue.Irreps("O3", "1x0e"),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self._act_cst = normalize2mom(F.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-atom scalar energy.

        Args:
            x: Node features ``(N, irreps_in.dim)``.

        Returns:
            Per-atom energy ``(N, 1)``.
        """
        return self.linear_2(F.silu(self.linear_1(x)) * self._act_cst)
