"""Product layer head for final scalar predictions.

Combines symmetric basis contraction + projection + linear readout into
a single-responsibility prediction head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from molix import config
from molrep.interaction.contraction import SymmetricContraction
from molrep.interaction.product import irreps_from_l_max
from molrep.readout.projection import BasisProjection

Key = str | tuple[str, ...]


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
