"""Equivariant linear layer using cuEquivariance.

Implements SO(3)-equivariant linear transformations for node features while 
preserving irreducible representation structure using cuEquivariance's 
hardware-accelerated kernels.

Reference:
    NVIDIA cuEquivariance Linear layers tutorial:
    https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pydantic import BaseModel

import cuequivariance as cue
import cuequivariance_torch as cuet

from molix import config


Key = str | tuple[str, ...]


class EquivariantLinearSpec(BaseModel):
    """Specification for equivariant linear transformation.
    
    Defines parameters for an SO(3)-equivariant linear layer that transforms
    node features while preserving their irreducible representation structure.
    
    Attributes:
        irreps_in: Input irreps string (e.g., "128x0e+128x1o").
            Defines the structure of input features.
        irreps_out: Output irreps string (e.g., "128x0e+128x1o").
            Defines the structure of output features. Can be different from
            irreps_in to change channel dimensions.
    """

    irreps_in: str
    irreps_out: str


class EquivariantLinear(nn.Module):
    """SO(3)-equivariant linear transformation module.
    
    Applies an equivariant linear transformation to features represented as
    direct sums of irreducible representations. This layer preserves the
    geometric structure (l-values) while allowing channel mixing.
    
    The transformation is performed using cuEquivariance's Linear layer,
    which respects the symmetry constraints of the O3 group.
    
    Attributes:
        config: EquivariantLinearSpec configuration.
        irreps_in: Input irreps object.
        irreps_out: Output irreps object.
        linear: cuEquivariance Linear layer backend.
    """

    def __init__(
        self,
        *,
        irreps_in: str,
        irreps_out: str,
    ):
        """Initialize equivariant linear module.
        
        Args:
            irreps_in: Input irreps specification (e.g., "128x0e+64x1o+32x2e").
            irreps_out: Output irreps specification. Can differ in channel
                dimensions but typically preserves the l-structure.
        """
        super().__init__()

        self.config = EquivariantLinearSpec(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

        # Parse irreps
        self.irreps_in = cue.Irreps("O3", self.config.irreps_in)
        self.irreps_out = cue.Irreps("O3", self.config.irreps_out)

        # Initialize cuEquivariance Linear layer
        self.linear = cuet.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation.
        
        Args:
            features: Input features. Shape: (..., irreps_in.dim)
                
        Returns:
            Transformed features. Output shape: (..., irreps_out.dim)
            
        Note:
            The transformation preserves SO(3) equivariance: if the input
            is rotated, the output rotates accordingly.
        """
        return self.linear(features)