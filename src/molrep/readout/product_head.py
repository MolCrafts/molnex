"""Product layer head for final scalar predictions.

Combines symmetric basis contraction + projection + linear readout into
a single-responsibility prediction head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict

from molrep.interaction.symmetric_contraction import SymmetricContraction
from molrep.readout.basis_projection import BasisProjection
from molix import config


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
    ):
        """Initialize product head.
        
        Args:
            hidden_dim: Dimension of node features.
            out_dim: Dimension of output predictions.
            num_radial: Number of radial basis functions.
            l_max: Maximum angular momentum.
            max_body_order: Maximum body order (1-3).
            num_species: Number of atomic species.
        """
        super().__init__()
        
        self.config = ProductHeadSpec(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_radial=num_radial,
            l_max=l_max,
            max_body_order=max_body_order,
            num_species=num_species,
        )
        
        # Single-responsibility sub-modules
        self.symmetric_contraction = SymmetricContraction(
            hidden_dim=hidden_dim,
            num_species=num_species,
            max_body_order=max_body_order,
        )
        
        self.basis_projection = BasisProjection(
            hidden_dim=hidden_dim,
            num_radial=num_radial,
            l_max=l_max,
            max_body_order=max_body_order,
        )
        
        self.linear = nn.Linear(hidden_dim, out_dim, dtype=config.ftype)
    
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
