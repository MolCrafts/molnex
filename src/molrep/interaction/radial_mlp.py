"""Radial weight MLP for generating tensor product weights from edge features.

Maps radial basis features (e.g. Bessel RBF output) to tensor product
weight vectors. Used in any architecture with radial-modulated tensor
product convolutions.

Example:
    >>> mlp = RadialWeightMLP(in_dim=8, hidden_dim=64, out_dim=128)
    >>> edge_feats = torch.randn(100, 8)  # 100 edges, 8 Bessel features
    >>> tp_weights = mlp(edge_feats)       # (100, 128) TP weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict


class RadialWeightMLPSpec(BaseModel):
    """Configuration for RadialWeightMLP.

    Attributes:
        in_dim: Input dimension (e.g. num_bessel).
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension (e.g. tensor product weight_numel).
        num_layers: Number of hidden layers (default 2).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_dim: int = Field(..., gt=0)
    hidden_dim: int = Field(..., gt=0)
    out_dim: int = Field(..., gt=0)
    num_layers: int = Field(2, ge=1)


class RadialWeightMLP(nn.Module):
    """MLP that maps radial edge features to tensor product weights.

    Single responsibility: transform scalar radial basis features into
    weight vectors for equivariant tensor product operations.

    Architecture:
        edge_feats (E, in_dim)
            -> [Linear(in_dim, hidden_dim) -> SiLU] x num_layers
            -> Linear(hidden_dim, out_dim)
            -> tp_weights (E, out_dim)

    Args:
        in_dim: Input dimension (number of radial basis functions).
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension (tensor product weight_numel).
        num_layers: Number of hidden layers (default 2).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()

        self.config = RadialWeightMLPSpec(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
        )

        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """Generate tensor product weights from radial edge features.

        Args:
            edge_feats: Radial basis features ``(n_edges, in_dim)``.

        Returns:
            Tensor product weights ``(n_edges, out_dim)``.
        """
        return self.mlp(edge_feats)
