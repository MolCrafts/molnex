"""Prediction heads for molecular property prediction."""

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """Predict molecular energy from atomic representations.
    
    Performs node-to-graph pooling after an atomic MLP.
    """
    
    def __init__(self, hidden_dim: int = 64):
        """Initialize energy head.
        
        Args:
            hidden_dim: Dimension of hidden representation
        """
        super().__init__()
        self.atomic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, atoms_h: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
        """Predict molecular energy.
        
        Args:
            atoms_h: Atomic hidden states [N, D]
            graph_batch: Molecule indices [N]
            
        Returns:
            Molecular energies [B]
        """
        # Compute atomic energies
        atomic_energies = self.atomic_mlp(atoms_h).squeeze(-1)
        
        # Pool to molecular level
        num_molecules = int(graph_batch.max().item()) + 1
        molecular_energies = torch.zeros(
            num_molecules,
            dtype=atomic_energies.dtype,
            device=atomic_energies.device
        )
        molecular_energies.index_add_(0, graph_batch, atomic_energies)
        
        return molecular_energies


class ForceHead(nn.Module):
    """Derive forces from energy via autograd.
    
    F = -dE/dx
    """
    
    def __init__(self):
        """Initialize force head."""
        super().__init__()
    
    def forward(self, target_energy: torch.Tensor, atoms_x: torch.Tensor) -> torch.Tensor:
        """Derive forces from energy.
        
        Args:
            target_energy: Molecular energy [B]
            atoms_x: Atomic positions [N, 3]
            
        Returns:
            Atomic forces [N, 3]
        """
        if not atoms_x.requires_grad:
            raise RuntimeError(
                "ForceHead requires atoms.x to have gradients enabled. "
                "Set atoms.x.requires_grad = True before calling ForceHead."
            )
        
        # F = -dE/dx
        forces = -torch.autograd.grad(
            target_energy.sum(),
            atoms_x,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]
        
        return forces


class TypeHead(nn.Module):
    """Predict atom types from atomic representations."""
    
    def __init__(self, hidden_dim: int = 64, num_types: int = 100):
        """Initialize type head.
        
        Args:
            hidden_dim: Dimension of hidden representation
            num_types: Number of atom types to predict
        """
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_types),
        )
    
    def forward(self, atoms_h: torch.Tensor) -> torch.Tensor:
        """Predict atom type logits.
        
        Args:
            atoms_h: Atomic hidden states [N, D]
            
        Returns:
            Type logits [N, num_types]
        """
        return self.module(atoms_h)
