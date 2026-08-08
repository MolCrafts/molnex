"""Element-specific residual update layer using cuEquivariance.

Applies element-specific linear transformations for state fusion across network
layers using cuEquivariance's indexed linear layer, enabling efficient
chemical-specific feature updates with hardware acceleration.

Reference:
    NVIDIA cuEquivariance Skip_tp/Indexed Linear tutorial:
    https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from molix import config

Key = str | tuple[str, ...]


class ElementUpdateSpec(BaseModel):
    """Configuration for element-specific residual update.

    Implements chemical-aware state fusion:
        h_i^(l+1) = h_i^(l) + W_z[i] @ m_i^(l)

    where W_z is element-specific and enables different update rules per atom type.

    Attributes:
        hidden_dim: Dimension of node features.
        num_species: Number of atomic species (0 to num_species inclusive).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_dim: int = Field(..., gt=0, description="Dimension of node features")
    num_species: int = Field(..., gt=0, description="Number of atomic species")


class ElementUpdate(nn.Module):
    """Element-specific residual update using cuEquivariance indexed linear.

    Performs efficient element-dependent residual state fusion:

        $$h_i^{(\\ell+1)} = h_i^{(\\ell)} + W_{z[i]} \\otimes m_i^{(\\ell)}$$

    where each element type $z \\in [0, \\text{num_species}]$ has its own
    $(\\text{hidden_dim} \\times \\text{hidden_dim})$ weight matrix $W_z$.

    Uses cuEquivariance's ``naive`` indexed-weights backend on every device.
    The ``indexed_linear`` CUDA kernel requires *sorted* species indices, and
    the argsort + un-permute round-trip that requirement forces was measured
    **2.8x slower** than the naive path at production shapes (N=672, H=128,
    GH200) — the kernel's own advantage never survives the reordering.

    Physical Interpretation:
        "I take my current state and add element-specific weighted information
        about my environment from the Product layer."

    Architecture:
        $$W = \\text{cuEquivariance Linear}(\\text{hidden_dim} \\to \\text{hidden_dim})$$
        with `weight_classes=num_species` enabling per-element different weights

    Example:
        >>> update = ElementUpdate(hidden_dim=128, num_species=118)
        >>> h_prev = torch.randn(10, 128)      # Previous layer features
        >>> m_curr = torch.randn(10, 128)      # Product layer output
        >>> Z = torch.tensor([6, 8, 1, 1, 6, 8, 1, 1, 6, 8])  # Atomic numbers
        >>> h_new = update(h_prev, m_curr, Z)
        >>> h_new.shape  # (10, 128)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_species: int,
    ):
        """Initialize element update layer.

        Args:
            hidden_dim: Dimension of node/message features.
            num_species: Number of atomic species (0 to num_species inclusive).
        """
        super().__init__()

        self.config = ElementUpdateSpec(
            hidden_dim=hidden_dim,
            num_species=num_species,
        )

        # Create cuEquivariance irreps (scalars only for hidden features)
        irreps = cue.Irreps("O3", f"{hidden_dim}x0e")

        # ``naive`` on every device: the alternative ``indexed_linear`` CUDA
        # kernel asserts sorted indices, and the argsort + un-permute round
        # trip that costs measured 2.8x slower than naive at production
        # shapes (0.974 ms vs 0.343 ms, N=672 H=128, GH200).
        self._linear_naive = cuet.Linear(
            irreps_in=irreps,
            irreps_out=irreps,
            internal_weights=False,
            weight_classes=num_species,
            layout=cue.ir_mul,
            method="naive",
            dtype=config.ftype,
        )

        # Initialize element-specific weight matrices
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.randn(num_species, hidden_dim * hidden_dim, dtype=config.ftype)
                / (hidden_dim**0.5)
            ),
        )

    def forward(
        self,
        h_prev: torch.Tensor,
        m_curr: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> torch.Tensor:
        """Update node features via element-specific residual connection.

        Args:
            h_prev: Previous layer features $(n\\_{nodes}, \\text{hidden_dim})$
            m_curr: Product/message output $(n\\_{nodes}, \\text{hidden_dim})$
            atom_types: Atomic numbers $(n\\_{nodes})$, value in $[0, \\text{num_species}]$

        Returns:
            Updated features $(n\\_{nodes}, \\text{hidden_dim})$ via:
            $$h\\_new[i] = h\\_prev[i] + W_{Z[i]} \\otimes m\\_curr[i]$$

        Performance:
            The ``naive`` indexed-weights path runs on any device with
            arbitrary index order. cuEq's ``indexed_linear`` kernel needs
            sorted indices; the argsort + un-permute round-trip that forces
            was measured 2.8x slower than this path at production shapes
            (GH200), so it is deliberately not used.
        """
        m_transformed = self._linear_naive(
            m_curr,
            weight=self.weight,
            weight_indices=atom_types,
        )
        return h_prev + m_transformed
