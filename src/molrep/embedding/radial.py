from __future__ import annotations

import math
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class BesselRBFSpec(BaseModel):
    """Specification for Bessel radial basis function.
    
    Defines parameters for computing Bessel RBF features from distance values.
    The Bessel RBF provides a smooth, localized representation of distances
    that is commonly used in message-passing neural networks.
    
    Attributes:
        r_cut: Cutoff radius. Distances are normalized by this value.
            Must be positive.
        num_radial: Number of radial basis functions. Must be positive.
        eps: Small constant to avoid division by zero. Defaults to 1e-8.
    """
    
    r_cut: float = Field(..., gt=0)
    num_radial: int = Field(..., gt=0)
    eps: float = 1e-8


class BesselRBF(nn.Module):
    """Bessel radial basis function module.
    
    Computes Bessel RBF features from distance values using the formula:
        phi_n(r) = sqrt(2/r_cut) * sin(n*pi*r/r_cut) / (r + eps)
    
    where n ranges from 1 to num_radial. These features provide a smooth,
    localized representation of interatomic distances.
    
    Attributes:
        config: BesselRBFSpec configuration.
        freqs: Buffer storing frequency values n*pi/r_cut.
        prefactor: Buffer storing normalization constant sqrt(2/r_cut).
        eps: Small constant for numerical stability.
    
    Input shape:
        r: (...,) tensor of distance values.
        
    Output shape:
        phi: (..., num_radial) tensor of RBF features.
    """

    def __init__(
        self,
        *,
        r_cut: float,
        num_radial: int,
        eps: float = 1e-8,
    ) -> None:
        """Initialize Bessel RBF module.
        
        Args:
            r_cut: Cutoff radius for normalization.
            num_radial: Number of radial basis functions.
            eps: Small constant to avoid division by zero. Defaults to 1e-8.
        """
        super().__init__()
        
        self.config = BesselRBFSpec(
            r_cut=r_cut,
            num_radial=num_radial,
            eps=eps,
        )

        self.r_cut = float(self.config.r_cut)
        num = int(self.config.num_radial)

        # Compute frequencies: n*pi/r_cut for n=1..num_radial
        freqs = torch.arange(1, num + 1, dtype=torch.float32) * (math.pi / self.r_cut)
        self.register_buffer("freqs", freqs, persistent=False)
        self.freqs: torch.Tensor

        # Compute normalization prefactor: sqrt(2/r_cut)
        prefactor = torch.tensor(math.sqrt(2.0 / self.r_cut), dtype=torch.float32)
        self.register_buffer("prefactor", prefactor, persistent=False)
        self.prefactor: torch.Tensor

        self.eps = float(self.config.eps)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute Bessel RBF features from distances.
        
        Args:
            r: Input distances. Expected shape: (...,)
                
        Returns:
            RBF features. Output shape: (..., num_radial)
        """
        r = r.float()

        # Expand dims for broadcasting: (..., 1) * (num_radial,) -> (..., num_radial)
        rr = r.unsqueeze(-1)
        x = rr * self.freqs

        phi = self.prefactor * torch.sin(x) / (rr + self.eps)

        return phi
