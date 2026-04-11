"""Symmetric basis construction via cuEquivariance.

Constructs multi-body geometric features by contracting node features using
cuEquivariance's SymmetricContraction, ensuring SO(3) rotation equivariance.
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from molix import config

Key = str | tuple[str, ...]


class SymmetricContractionSpec(BaseModel):
    """Configuration for symmetric basis construction.

    Constructs multi-body geometric features by contracting self-products of
    single-particle representations via Clebsch-Gordan coefficients.

    Attributes:
        hidden_dim: Dimension of input node features.
        num_species: Number of atomic species (0 to num_species inclusive).
        max_body_order: Maximum body order for multi-body expansion.
            1: two-body only
            2: up to three-body (default)
            3: up to four-body
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_dim: int = Field(..., gt=0, description="Dimension of input features")
    num_species: int = Field(..., gt=0, description="Number of atomic species")
    max_body_order: int = Field(2, ge=1, le=3, description="Maximum body order")


class SymmetricContraction(nn.Module):
    """Multi-body symmetric basis construction using cuEquivariance.

    Generates symmetric (invariant under permutations) basis functions through
    Clebsch-Gordan tensor product contractions, enabling the model to learn
    multi-body interactions up to max_body_order.

    Architecture:
        Input node features (n_nodes, hidden_dim) with scalars at L=0
                                ↓
        cuEquivariance SymmetricContraction with degree = max_body_order
                                ↓
        Output multi-body contracted features (n_nodes, hidden_dim)

    Example:
        >>> contraction = SymmetricContraction(
        ...     hidden_dim=128,
        ...     num_species=118,
        ...     max_body_order=2,
        ... )
        >>> node_features = torch.randn(10, 128)
        >>> atom_types = torch.randint(0, 118, (10,))
        >>> basis = contraction(node_features, atom_types)
        >>> basis.shape  # (10, 128)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_species: int,
        max_body_order: int = 2,
    ):
        """Initialize symmetric contraction layer.

        Args:
            hidden_dim: Dimension of input node features.
            num_species: Number of atomic species.
            max_body_order: Maximum body order (1-3).
        """
        super().__init__()

        self.config = SymmetricContractionSpec(
            hidden_dim=hidden_dim,
            num_species=num_species,
            max_body_order=max_body_order,
        )

        # Build cuEquivariance irreps for scalars only (L=0)
        irreps_in = f"{hidden_dim}x0e"
        irreps_out = irreps_in

        cue_irreps_in = cue.Irreps("O3", irreps_in)
        cue_irreps_out = cue.Irreps("O3", irreps_out)

        # Create cuEquivariance SymmetricContraction
        # This performs symmetric tensor product contractions up to max_body_order
        # dtype: Use global config.FLOAT_DTYPE
        self.symmetric_contraction = cuet.SymmetricContraction(
            cue_irreps_in,
            cue_irreps_out,
            contraction_degree=max_body_order,
            num_elements=num_species,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
            original_mace=True,
            dtype=config.ftype,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric multi-body basis functions.

        Args:
            node_features: Node features from Interaction layer (n_nodes, hidden_dim)
            atom_types: Atomic numbers for species-specific contraction (n_nodes,)

        Returns:
            Contracted multi-body features (n_nodes, hidden_dim)

        Mathematical:
            Returns multi-body contracted tensors via symmetric products:
            output = Σ_{ν,η} C^{LM}_{1,...,ν} ∏_j A_{i,k_j}
            where C are Clebsch-Gordan coefficients.
        """
        return self.symmetric_contraction(node_features, atom_types)
