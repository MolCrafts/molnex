"""MACE-only readout heads.

Two readout families, both used exclusively by the MACE encoders:

**Scalar readouts** (relocated from :mod:`molrep.readout.scalar`) map per-atom
equivariant features to a scalar (energy):

- :class:`LinearReadout` — MACE's ``LinearReadoutBlock``: one ``cuet.Linear``
  onto ``1x0e``, used on every interaction layer except the last.
- :class:`NonLinearReadout` — bias-free ``Linear → c·SiLU → Linear``
  (``NonLinearReadoutBlock``); the last-layer readout of MACE-MP / MatPES.
- :class:`NonLinearBiasReadout` — the OMOL variant
  ``Linear → SiLU → o3.Linear(+bias) → SiLU → o3.Linear(+bias)``. The first
  linear is an equivariant ``cuet.Linear`` projecting to ``MLP_irreps`` scalars;
  the two subsequent biased linears act on scalars only and reproduce e3nn's
  ``o3.Linear`` normalisation (``out = (x @ W.reshape(in,out)) / sqrt(in) + b``).
  The SiLU activations carry e3nn's ``normalize2mom`` scaling.

**Product head** (relocated from :mod:`molrep.readout.product`):
:class:`ProductHead` combines symmetric basis contraction + projection + linear
readout into a single-responsibility prediction head.

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
from pydantic import BaseModel, ConfigDict, Field

from molix import config
from molrep.embedding.mlp import normalize2mom
from molrep.interaction.contraction import SymmetricContraction
from molrep.interaction.product import irreps_from_l_max
from molrep.readout.projection import BasisProjection

Key = str | tuple[str, ...]


class _ScalarO3Linear(nn.Module):
    """Scalar-only e3nn ``o3.Linear`` with bias: ``(x @ W/sqrt(in)) + b``.

    Initialisation follows e3nn's ``o3.Linear`` convention: the weight is drawn
    from the **global** RNG as standard normal ``N(0, 1)`` and the path
    normalisation ``1/sqrt(in_mul)`` is applied in :meth:`forward`
    (``self._alpha``) rather than folded into the init, so the stored weight
    stays unit-variance. The bias is zero-initialised.

    A zero weight (the previous init) makes the layer emit a constant per-atom
    energy on an untrained model, which silently zeroes the forces and voids
    every downstream force / parity assertion. Loading a checkpoint overwrites
    the random init exactly, so weight-transfer paths are unaffected.
    """

    def __init__(self, in_mul: int, out_mul: int) -> None:
        super().__init__()
        self.in_mul, self.out_mul = in_mul, out_mul
        self.weight = nn.Parameter(torch.randn(in_mul * out_mul, dtype=config.ftype))
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


class ProductHeadSpec(BaseModel):
    """Configuration for product prediction head.

    Combines multi-body basis construction (via SymmetricContraction),
    optional basis projection, and linear readout to scalars.

    Attributes:
        hidden_dim: Dimension of input node features.
        out_dim: Dimension of output predictions (1 for scalar energy).
        num_radial: Number of radial basis functions.
        l_max: Maximum angular momentum.
        max_body_order: Maximum body order for multi-body expansion.
        num_species: Number of atomic species.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_dim: int = Field(..., gt=0)
    out_dim: int = Field(..., gt=0)
    num_radial: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    max_body_order: int = Field(2, ge=1, le=3)
    num_species: int = Field(118, gt=0)
    use_fallback: bool = True


class ProductHead(nn.Module):
    """Product layer head for multi-body-aware scalar predictions.

    Single-responsibility module that:
    1. Constructs symmetric multi-body basis (SymmetricContraction)
    2. Projects basis features (BasisProjection)
    3. Applies linear transformation to output dimension

    Does NOT apply pooling - that is the responsibility of a separate
    pooling module. Returns node-level predictions only.

    Architecture:
        node_features (n_nodes, hidden_dim) + atom_types (n_nodes,)
                                 ↓
                    [SymmetricContraction]
                                 ↓
                         basis (n_nodes, hidden_dim)
                                 ↓
                    [BasisProjection]
                                 ↓
                     features (n_nodes, hidden_dim)
                                 ↓
                     [Linear(hidden_dim → out_dim)]
                                 ↓
                     predictions (n_nodes, out_dim)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        out_dim: int,
        num_radial: int = 8,
        l_max: int = 2,
        max_body_order: int = 2,
        num_species: int = 118,
        use_fallback: bool = True,
    ):
        """Initialize product head.

        Args:
            hidden_dim: Dimension of node features.
            out_dim: Dimension of output predictions.
            num_radial: Number of radial basis functions.
            l_max: Maximum angular momentum.
            max_body_order: Maximum body order (1-3).
            num_species: Number of atomic species.
            use_fallback: Pure-torch cuEq path for the symmetric contraction
                (default ``True``, functorch-safe). Set ``False`` for the
                fused kernels when forces use the autograd backend.
        """
        super().__init__()

        self.config = ProductHeadSpec(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_radial=num_radial,
            l_max=l_max,
            max_body_order=max_body_order,
            num_species=num_species,
            use_fallback=use_fallback,
        )

        # ``hidden_dim`` is the *full* mixed-l feature dim emitted by the
        # interaction block (e.g. 144 = 16x0e+16x1o+16x2e for l_max=2). Recover
        # the per-l multiplicity (scalar channel count) so the contraction can
        # be told the *real* irreps. Declaring this mixed-l tensor as pure
        # scalars is the rotation-invariance bug this head exists to avoid.
        per_channel_dim = (l_max + 1) ** 2
        if hidden_dim % per_channel_dim != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} is not a multiple of (l_max+1)^2="
                f"{per_channel_dim}; cannot infer the mixed-l irreps multiplicity."
            )
        num_features = hidden_dim // per_channel_dim
        irreps_in = irreps_from_l_max(l_max, num_features)
        irreps_out = f"{num_features}x0e"  # invariant scalar output

        # Single-responsibility sub-modules
        self.symmetric_contraction = SymmetricContraction(
            hidden_dim=hidden_dim,
            num_species=num_species,
            max_body_order=max_body_order,
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            use_fallback=use_fallback,
        )

        self.basis_projection = BasisProjection(
            hidden_dim=hidden_dim,
            num_radial=num_radial,
            l_max=l_max,
            max_body_order=max_body_order,
        )

        # The contraction emits ``num_features`` invariant scalars; the readout
        # linear maps those scalars (not the full mixed-l dim) to ``out_dim``.
        self.linear = nn.Linear(num_features, out_dim, dtype=config.ftype)

    def forward(
        self,
        node_features: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute node-level predictions from features.

        Args:
            node_features: Node features (n_nodes, hidden_dim)
            atom_types: Atomic numbers (n_nodes,)

        Returns:
            Predictions (n_nodes, out_dim).
        """
        # Step 1: Symmetric multi-body basis via cuEquivariance
        basis = self.symmetric_contraction(node_features, atom_types)

        # Step 2: Project basis features (currently passthrough with cuEquivariance)
        features = self.basis_projection(basis)

        # Step 3: Linear transformation to output dimension
        predictions = self.linear(features)

        return predictions
