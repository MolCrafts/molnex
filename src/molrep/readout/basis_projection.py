"""Basis function projection for multi-body features.

Projects multi-body symmetric basis functions to refined output features
via learnable linear combinations, but when using cuEquivariance contraction,
acts as a passthrough (contraction already mixes body orders).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

Key = str | tuple[str, ...]


class BasisProjectionSpec(BaseModel):
    """Configuration for basis function projection.

    Attributes:
        hidden_dim: Dimension of basis input features.
        num_radial: Number of radial basis functions (context only).
        l_max: Maximum angular momentum (context only).
        max_body_order: Maximum body order (context only).

    Note:
        When using cuEquivariance SymmetricContraction, the basis is already
        optimally mixed, so projection acts as identity passthrough.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_dim: int = Field(..., gt=0, description="Input feature dimension")
    num_radial: int = Field(8, gt=0, description="Radial basis dimension")
    l_max: int = Field(2, ge=0, description="Maximum angular momentum")
    max_body_order: int = Field(2, ge=1, le=3, description="Maximum body order")


class BasisProjection(nn.Module):
    """Basis function projection module.

    When using cuEquivariance's symmetric contraction, basis functions are already
    optimally combined across body orders. This module acts as a learnable projection
    that could filter or recombine features if needed, but typically serves as
    identity passthrough.

    Future Extension:
        Could implement learnable weighting of different (body_order, L) channels
        for additional flexibility, e.g.:
            m_{i,kLM} = Σ_{ν,η} W_{k,L,ν,η} B_{i,νηkLM}

    Example:
        >>> proj = BasisProjection(
        ...     hidden_dim=128,
        ...     num_radial=8,
        ...     l_max=2,
        ...     max_body_order=2,
        ... )
        >>> basis = torch.randn(10, 128)  # contracted basis features
        >>> output = proj(basis)
        >>> output.shape  # (10, 128)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_radial: int = 8,
        l_max: int = 2,
        max_body_order: int = 2,
    ):
        """Initialize basis projection layer.

        Args:
            hidden_dim: Dimension of basis features.
            num_radial: Number of radial basis functions (unused in passthrough).
            l_max: Maximum angular momentum (unused in passthrough).
            max_body_order: Maximum body order (unused in passthrough).
        """
        super().__init__()

        self.config = BasisProjectionSpec(
            hidden_dim=hidden_dim,
            num_radial=num_radial,
            l_max=l_max,
            max_body_order=max_body_order,
        )

        # Currently implemented as identity passthrough
        # Since cuEquivariance SymmetricContraction already optimally mixes
        # all body-order contributions via Clebsch-Gordan contraction.

    def forward(self, basis_tensor: torch.Tensor) -> torch.Tensor:
        """Project basis functions (passthrough when using cuEquivariance contraction).

        Args:
            basis_tensor: Multi-body basis features (n_nodes, hidden_dim)

        Returns:
            Projected basis features (n_nodes, hidden_dim)

        Note:
            Currently returns basis_tensor unchanged since cuEquivariance
            SymmetricContraction already performs optimal mixing via CG coefficients.

            Future: Could become learnable if per-body-order weighting is desired.
        """
        return basis_tensor
