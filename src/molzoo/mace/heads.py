"""MACE scalar readouts.

The scalar readouts that turn the final node features into per-atom energies.
A product chooses which readouts to attach, and how many (the charge/spin
conditioning embedding at the other end of the body is
:mod:`molrep.encoders.mace.conditioning`):

- :class:`LinearReadout` — one ``molrep.equivariance.Linear`` onto ``1x0e``.
- :class:`NonLinearReadout` — bias-free ``Linear → c·SiLU → Linear``.
- :class:`NonLinearBiasReadout` — ``Linear → SiLU → biased linear → SiLU →
  biased linear``. ``out_mul`` is 1 for an energy and 2 when a caller wants
  two scalar channels (the polar Fukui source).

Sub-layer names (``embedders.<name>``, ``project.0``, ``linear_1``,
``linear_mid``, ``linear_2``) stay as they are so an official checkpoint still
copies onto them.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022;
    OMol charge/spin conditioning. https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import molrep.eq as meq
from molrep.embedding.mlp import normalize2mom
from molrep.equivariance import Linear


class _ScalarO3Linear(nn.Module):
    """Scalar-only equivariant linear with bias (upstream MACE): ``(x @ W/sqrt(in)) + b``.

    Initialisation follows upstream MACE's linear-layer convention: the weight is drawn
    from the **global** RNG as standard normal ``N(0, 1)`` and the path
    normalisation ``1/sqrt(in_mul)`` is applied in :meth:`forward`
    (``self._alpha``) rather than folded into the init, so the stored weight
    stays unit-variance. The bias is zero-initialised.

    A zero weight (the previous init) makes the layer emit a constant per-atom
    energy on an untrained model, which silently zeroes the forces and voids
    every downstream force / parity assertion. Loading a checkpoint overwrites
    the random init exactly, so weight-transfer paths are unaffected.
    """

    def __init__(self, in_mul: int, out_mul: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_mul, self.out_mul = in_mul, out_mul
        self.weight = nn.Parameter(torch.randn(in_mul * out_mul, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_mul, dtype=dtype))
        self._alpha = 1.0 / math.sqrt(in_mul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(self.in_mul, self.out_mul)
        return self._alpha * (x @ w) + self.bias


class NonLinearBiasReadout(nn.Module):
    """Gated non-linear scalar readout to per-atom energy (single head).

    Args:
        irreps_in: Input node feature irreps (last-layer product output).
        mlp_dim: Hidden scalar width (``MLP_irreps`` count), e.g. 16.
        out_mul: Output scalar multiplicity. Energy heads use 1; Polar's
            Fukui source map uses 2.
    """

    def __init__(
        self,
        *,
        irreps_in: str,
        mlp_dim: int = 16,
        out_mul: int = 1,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(
            meq.Irreps("O3", irreps_in),
            meq.Irreps("O3", f"{mlp_dim}x0e"),
            layout=meq.ir_mul,
            dtype=dtype,
        )
        self._act_cst = normalize2mom(F.silu)
        self.linear_mid = _ScalarO3Linear(mlp_dim, mlp_dim, dtype=dtype)
        self.linear_2 = _ScalarO3Linear(mlp_dim, out_mul, dtype=dtype)

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

    MACE's ``LinearReadoutBlock``: one ``molrep.equivariance.Linear`` onto ``1x0e``. Used for
    every interaction layer except the last, where the non-linear readout
    (:class:`NonLinearReadout`) takes over.

    Args:
        irreps_in: Input node feature irreps, e.g. ``"128x0e+128x1o"``.

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, *, irreps_in: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear = Linear(
            meq.Irreps("O3", irreps_in),
            meq.Irreps("O3", "1x0e"),
            layout=meq.ir_mul,
            dtype=dtype,
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

    ``Linear → c·SiLU → Linear``, both equivariant ``molrep.equivariance.Linear`` layers with no
    bias; ``c`` is the moment normalisation constant (``normalize2mom``). This is the readout on
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

    def __init__(self, *, irreps_in: str, mlp_dim: int = 16, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear_1 = Linear(
            meq.Irreps("O3", irreps_in),
            meq.Irreps("O3", f"{mlp_dim}x0e"),
            layout=meq.ir_mul,
            dtype=dtype,
        )
        self.linear_2 = Linear(
            meq.Irreps("O3", f"{mlp_dim}x0e"),
            meq.Irreps("O3", "1x0e"),
            layout=meq.ir_mul,
            dtype=dtype,
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
