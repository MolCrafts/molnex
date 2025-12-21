"""Simplified PiNet model implementation.

This is a simplified version of PiNet that demonstrates the component-first architecture.
It uses all the reusable components from MolPot.
"""

import torch
import torch.nn as nn

from molpot.potentials.base import BasePotential
from molpot.feats.rbf import GaussianRBF
from molpot.feats.cutoff import CosineCutoff
from molpot.graph.radius_graph import radius_graph
from molpot.nn.mlp import MLP
from molpot.heads.energy import EnergyHead


class PiNet(BasePotential):
    """Simplified PiNet model for molecular energy prediction.
    
    This model demonstrates component-first architecture by composing
    reusable building blocks from MolPot.
    
    ML potential - learns from atomic numbers, no types needed.
    
    Attributes:
        embedding: Atomic number embedding
        rbf: Radial basis functions
        cutoff: Cutoff function
        interaction_mlps: List of interaction MLPs
        update_mlps: List of update MLPs
        energy_mlp: Final MLP for atomic energies
        energy_head: Energy prediction head
    """
    
    name = "pinet"
    type = "ml"
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        max_z: int = 100,
    ):
        """Initialize PiNet model.
        
        Args:
            hidden_dim: Hidden dimension for MLPs
            num_layers: Number of interaction layers
            cutoff: Cutoff radius for neighbor search
            num_rbf: Number of radial basis functions
            max_z: Maximum atomic number for embedding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff_radius = cutoff
        
        # Atomic number embedding
        self.embedding = nn.Embedding(max_z, hidden_dim)
        
        # Radial basis functions and cutoff
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff = CosineCutoff(cutoff=cutoff)
        
        # Interaction and update MLPs
        self.interaction_mlps = nn.ModuleList([
            MLP(
                in_dim=hidden_dim + num_rbf,
                out_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                activation="silu",
            )
            for _ in range(num_layers)
        ])
        
        self.update_mlps = nn.ModuleList([
            MLP(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                activation="silu",
            )
            for _ in range(num_layers)
        ])
        
        # Final MLP for atomic energies
        self.energy_mlp = MLP(
            in_dim=hidden_dim,
            out_dim=1,
            hidden_dims=[hidden_dim // 2],
            activation="silu",
        )
        
        # Energy head for pooling
        self.energy_head = EnergyHead(pooling="sum")
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: AtomicTD or Frame with fields:
                - ["atoms"]["z"]: Atomic numbers [num_atoms]
                - ["atoms"]["x"]: Atomic positions [num_atoms, 3]
                - ["graph"]["batch"]: Batch/molecule indices [num_atoms]
                
        Returns:
            Molecular energies (scalar for single molecule, or sum for batch)
        """
        # Extract data using nested access
        z = data["atoms"]["z"]
        pos = data["atoms"]["x"]
        batch_idx = data["graph"]["batch"]
        
        # Convert numpy to torch if needed
        if not isinstance(z, torch.Tensor):
            import numpy as np
            z = torch.from_numpy(z).long()
            pos = torch.from_numpy(pos).float()
            batch_idx = torch.from_numpy(batch_idx).long()
        
        # Initial embedding
        h = self.embedding(z)  # [num_atoms, hidden_dim]
        
        # Build graph
        edge_index, edge_vec = radius_graph(
            pos, batch_idx, cutoff=self.cutoff_radius
        )
        
        # Compute edge features
        distances = torch.norm(edge_vec, dim=-1)  # [num_edges]
        rbf_features = self.rbf(distances)  # [num_edges, num_rbf]
        cutoff_values = self.cutoff(distances)  # [num_edges]
        
        # Message passing layers
        for interaction_mlp, update_mlp in zip(self.interaction_mlps, self.update_mlps):
            # Interaction: compute messages
            h_i = h[edge_index[0]]  # [num_edges, hidden_dim]
            
            # Concatenate node features with edge features
            edge_features = torch.cat([h_i, rbf_features], dim=-1)  # [num_edges, hidden_dim + num_rbf]
            
            # Compute messages
            messages = interaction_mlp(edge_features)  # [num_edges, hidden_dim]
            
            # Apply cutoff
            messages = messages * cutoff_values.unsqueeze(-1)
            
            # Aggregate messages
            aggregated = torch.zeros_like(h)  # [num_atoms, hidden_dim]
            aggregated.index_add_(0, edge_index[1], messages)
            
            # Update node features
            h = h + update_mlp(aggregated)
        
        # Compute atomic energies
        atomic_energies = self.energy_mlp(h)  # [num_atoms, 1]
        
        # Pool to molecular energies
        molecular_energies = self.energy_head(atomic_energies, batch_idx)  # [num_molecules]
        
        # Return sum for compatibility with BasePotential
        return molecular_energies.sum()
    
    def __repr__(self) -> str:
        return (
            f"PiNet("
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"cutoff={self.cutoff_radius})"
        )
