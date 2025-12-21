"""Force prediction head."""

import torch
import torch.nn as nn


class ForceHead(nn.Module):
    """Force prediction head via negative gradient of energy.
    
    Computes forces as F = -dE/dpos using autograd.
    
    Note:
        Requires positions to have requires_grad=True during forward pass.
    """
    
    def __init__(self):
        """Initialize force head."""
        super().__init__()
    
    def forward(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forces from energy via autograd.
        
        Args:
            energy: Molecular or total energy (scalar or [batch])
            pos: Atomic positions [num_atoms, 3] with requires_grad=True
            
        Returns:
            Forces [num_atoms, 3]
        """
        if not pos.requires_grad:
            raise ValueError("Positions must have requires_grad=True for force computation")
        
        # Compute gradient
        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training,  # Keep graph for training
            retain_graph=self.training,
        )[0]
        
        return forces
    
    def __repr__(self) -> str:
        return "ForceHead()"
