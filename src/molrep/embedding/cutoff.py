from __future__ import annotations

import math
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


Key = str | tuple[str, ...]


class CosineCutoffSpec(BaseModel):
    """Specification for cosine cutoff function.
    
    Defines parameters for a smooth cosine cutoff function that transitions
    from 1 to 0 over the range [0, r_cut].
    
    Attributes:
        r_cut: Cutoff radius. Values are 0 for r >= r_cut. Must be positive.
    """
    
    r_cut: float = Field(..., gt=0)


class CosineCutoff(nn.Module):
    """Cosine cutoff function module.
    
    Applies a smooth cosine cutoff to distance values:
        c(r) = 0.5 * (cos(pi * r / r_cut) + 1)  for r < r_cut
        c(r) = 0                                for r >= r_cut
    
    This provides a smooth transition to zero at the cutoff radius, which is
    important for avoiding discontinuities in neural network potentials.
    
    Attributes:
        config: CosineCutoffSpec configuration.
        r_cut: Buffer storing cutoff radius.
        _pi: Buffer storing pi constant.
    """

    def __init__(self, *, r_cut: float):
        """Initialize cosine cutoff module.
        
        Args:
            r_cut: Cutoff radius.
        """
        super().__init__()
        
        self.config = CosineCutoffSpec(
            r_cut=r_cut,
        )

        # Register buffers with type annotations
        r_cut_tensor = torch.tensor(float(self.config.r_cut))
        self.register_buffer("r_cut", r_cut_tensor, persistent=False)
        self.r_cut: torch.Tensor

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply cosine cutoff to distances.
        
        Args:
            r: Input distances.
                
        Returns:
            Cutoff values. Values range from 1.0 (at r=0) to 0.0 (at r>=r_cut).
        """
        r = r.float()

        # Create mask for distances within cutoff
        mask = r < self.r_cut

        # Compute cosine cutoff
        x = r / self.r_cut
        c = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        c = torch.where(mask, c, torch.zeros_like(c))

        return c


class PolynomialCutoffSpec(BaseModel):
    """Specification for polynomial cutoff function.
    
    Defines parameters for a smooth polynomial cutoff function with
    continuous derivatives up to a specified order.
    
    Attributes:
        r_cut: Cutoff radius. Values are 0 for r >= r_cut. Must be positive.
        exponent: Polynomial exponent controlling smoothness. Higher values
            give smoother cutoffs. Defaults to 6. Must be positive.
    """
    
    r_cut: float = Field(..., gt=0)
    exponent: int = Field(6, gt=0)


class PolynomialCutoff(nn.Module):
    """Polynomial cutoff function module.
    
    Applies a smooth polynomial cutoff to distance values:
        E(u) = 1 - p*u^n + q*u^(n-1) - s*u^(n-2)  for u < 1
        E(u) = 0                                    for u >= 1
    
    where u = r/r_cut and n is the exponent. For the default n=6:
        E(u) = 1 - 6u^5 + 15u^4 - 10u^3
    
    This cutoff has continuous derivatives up to order (n-3), providing
    smooth transitions that are important for force calculations in
    molecular dynamics.
    
    Attributes:
        config: PolynomialCutoffSpec configuration.
        r_cut: Buffer storing cutoff radius.
        exponent: Polynomial exponent.
    """

    def __init__(self, *, r_cut: float, exponent: int = 6):
        """Initialize polynomial cutoff module.
        
        Args:
            r_cut: Cutoff radius.
            exponent: Polynomial exponent controlling smoothness. Defaults to 6.
        """
        super().__init__()
        
        self.config = PolynomialCutoffSpec(
            r_cut=r_cut,
            exponent=exponent,
        )

        # Register r_cut buffer
        r_cut_tensor = torch.tensor(float(self.config.r_cut))
        self.register_buffer("r_cut", r_cut_tensor, persistent=False)
        self.r_cut: torch.Tensor
        
        self.exponent = int(self.config.exponent)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply polynomial cutoff to distances.
        
        Args:
            r: Input distances.
                
        Returns:
            Cutoff values. Values range from 1.0 (at r=0) to 0.0 (at r>=r_cut).
        """
        r = r.float()

        # Normalized distance: u = r / r_cut
        u = r / self.r_cut
        
        # Create mask for distances within cutoff
        mask = u < 1.0

        # Polynomial coefficients: p*u^n - q*u^(n-1) + s*u^(n-2)
        p = self.exponent
        q = p * (p - 1) // 2
        s = p * (p - 1) * (p - 2) // 6
        
        # Precompute powers for efficiency
        u2 = u * u
        u3 = u2 * u
        
        if self.exponent == 6:
            # Optimized path for common case (n=6)
            c = 1.0 - 6.0 * u3 * u2 + 15.0 * u2 * u2 - 10.0 * u3
        else:
            # General path for arbitrary exponent
            u_n = torch.pow(u, self.exponent)
            u_n1 = torch.pow(u, self.exponent - 1)
            u_n2 = torch.pow(u, self.exponent - 2)
            c = 1.0 - p * u_n + q * u_n1 - s * u_n2
        
        # Apply mask to set values outside cutoff to zero
        result = torch.where(mask, c, torch.zeros_like(r))

        return result
