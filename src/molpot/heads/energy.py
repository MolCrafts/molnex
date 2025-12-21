"""Energy prediction head."""

import torch
import torch.nn as nn

from molpot.readout.pooling import SumPooling


class EnergyHead(nn.Module):
    """Energy prediction head.
    
    Aggregates atomic energies to molecular energy via sum pooling.
    
    Attributes:
        pooling: Pooling operation (default: SumPooling)
    """
    
    def __init__(self, pooling: str = "sum"):
        """Initialize energy head.
        
        Args:
            pooling: Pooling method ("sum", "mean")
        """
        super().__init__()
        
        if pooling == "sum":
            from molpot.readout.pooling import SumPooling
            self.pooling = SumPooling()
        elif pooling == "mean":
            from molpot.readout.pooling import MeanPooling
            self.pooling = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    
    def forward(
        self,
        atomic_energies: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute molecular energies from atomic energies.
        
        Args:
            atomic_energies: Atomic energy contributions [num_atoms, 1] or [num_atoms]
            batch: Batch/molecule indices [num_atoms]
            
        Returns:
            Molecular energies [num_molecules]
        """
        # Ensure atomic_energies is 2D
        if atomic_energies.dim() == 1:
            atomic_energies = atomic_energies.unsqueeze(-1)
        
        # Pool to molecular level
        molecular_energies = self.pooling(atomic_energies, batch)  # [num_molecules, 1]
        
        # Squeeze to 1D
        molecular_energies = molecular_energies.squeeze(-1)  # [num_molecules]
        
        return molecular_energies
    
    def __repr__(self) -> str:
        return f"EnergyHead(pooling={self.pooling})"
