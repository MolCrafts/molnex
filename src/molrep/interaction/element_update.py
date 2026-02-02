"""Element-specific residual update layer using cuEquivariance.

Applies element-specific linear transformations for state fusion across network
layers using cuEquivariance's indexed linear layer, enabling efficient
chemical-specific feature updates with hardware acceleration.

Reference:
    NVIDIA cuEquivariance Skip_tp/Indexed Linear tutorial:
    https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict

import cuequivariance as cue
import cuequivariance_torch as cuet

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
    
    Uses cuEquivariance's **indexed_linear** backend for 8-11x speedup:
        - **indexed_linear**: Hardware-optimized kernel for sorted species indices
        - **naive**: Fallback for general cases
    
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
        
        # Initialize cuEquivariance Linear layer with weight classes
        # weight_classes=num_species: different weights for each element
        # method='indexed_linear': optimized kernel for sorted indices (8-11x faster)
        # dtype: Use global config.ftype
        self.linear = cuet.Linear(
            irreps_in=irreps,
            irreps_out=irreps,
            internal_weights=False,  # Provide weights externally
            weight_classes=num_species,  # Different W matrix per element
            layout=cue.ir_mul,
            method='indexed_linear',  # Hardware-optimized for sorted species
            dtype=config.ftype,
        )
        
        # Initialize element-specific weight matrices
        self.register_parameter(
            'weight',
            nn.Parameter(
                torch.randn(num_species, hidden_dim * hidden_dim, dtype=config.ftype) 
                / (hidden_dim ** 0.5)
            )
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
        
        Implementation (cuEquivariance indexed_linear):
            Uses hardware-optimized kernel that:
            1. Performs indexed lookup of weight matrices (W[species])
            2. Applies element-specific linear transformation in single kernel
            3. Aggregates with h_prev for residual connection
            
        Performance:
            - **sorted indices**: 8-11x faster than naive loop
            - **unsorted indices**: Falls back to naive but still indexed
        """
        # Apply element-specific linear transformation via cuEquivariance
        # This internally does: W[species[i]] @ m_curr[i] for each node
        m_transformed = self.linear(
            m_curr,
            weight=self.weight,
            weight_indices=atom_types,
        )
        
        # Residual connection
        h_new = h_prev + m_transformed
        
        return h_new
