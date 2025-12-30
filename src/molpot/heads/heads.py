"""Prediction head TensorDictModule components.

All components properly inherit from TensorDictModule.
"""

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class EnergyHead(TensorDictModule):
    """Predict molecular energy from atomic representations.
    
    in_keys: [("atoms", "h"), ("graph", "batch")]
    out_keys: [("target", "energy")]
    """
    
    def __init__(self, hidden_dim: int = 64):
        """Initialize energy head.
        
        Args:
            hidden_dim: Dimension of hidden representation
        """
        module = _EnergyHeadModule(hidden_dim)
        super().__init__(
            module=module,
            in_keys=[("atoms", "h"), ("graph", "batch")],
            out_keys=[("target", "energy")],
        )


class _EnergyHeadModule(torch.nn.Module):
    """Internal module for EnergyHead."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.atomic_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, atoms_h: torch.Tensor, graph_batch: torch.Tensor):
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


class ForceHead(TensorDictModule):
    """Derive forces from energy via autograd.
    
    in_keys: [("target", "energy"), ("atoms", "x")]
    out_keys: [("atoms", "f")]
    """
    
    def __init__(self):
        """Initialize force head."""
        module = _ForceHeadModule()
        super().__init__(
            module=module,
            in_keys=[("target", "energy"), ("atoms", "x")],
            out_keys=[("atoms", "f")],
        )


class _ForceHeadModule(torch.nn.Module):
    """Internal module for ForceHead."""
    
    def forward(self, target_energy: torch.Tensor, atoms_x: torch.Tensor):
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
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return forces


class TypeHead(TensorDictModule):
    """Predict atom types from atomic representations.
    
    in_keys: [("atoms", "h")]
    out_keys: [("atoms", "type_logits")]
    """
    
    def __init__(self, hidden_dim: int = 64, num_types: int = 100):
        """Initialize type head.
        
        Args:
            hidden_dim: Dimension of hidden representation
            num_types: Number of atom types to predict
        """
        module = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_types),
        )
        super().__init__(
            module=module,
            in_keys=[("atoms", "h")],
            out_keys=[("atoms", "type_logits")],
        )
