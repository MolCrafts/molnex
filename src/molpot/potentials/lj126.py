"""Lennard-Jones 12-6 potential implementation."""

import torch
from molpot.potentials.base import BasePotential
from molpot.data.atomic_td import AtomicTD
from molpot.graph.radius_graph import radius_graph


class LJ126(BasePotential):
    """Lennard-Jones 12-6 potential.
    
    Energy formula:
        E_ij = 4 * epsilon_ij * [(sigma_ij / r_ij)^12 - (sigma_ij / r_ij)^6]
    
    Parameters are stored as type-indexed matrices:
        epsilon[type_i, type_j]: Well depth
        sigma[type_i, type_j]: Zero-crossing distance
    
    Attributes:
        epsilon: Well depth parameters [num_types, num_types]
        sigma: Zero-crossing distance parameters [num_types, num_types]
        cutoff: Cutoff radius for neighbor search
    """
    
    name = "lj126_torch"
    type = "pair"
    
    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        cutoff: float = 10.0,
    ):
        """Initialize LJ126 potential.
        
        Args:
            epsilon: Well depth matrix [num_types, num_types]
            sigma: Zero-crossing distance matrix [num_types, num_types]
            cutoff: Cutoff radius for neighbor search (default: 10.0 Å)
        """
        super().__init__()
        
        # Validate shapes
        if epsilon.shape != sigma.shape:
            raise ValueError(
                f"epsilon and sigma must have same shape, got "
                f"epsilon: {epsilon.shape}, sigma: {sigma.shape}"
            )
        
        if epsilon.ndim != 2 or epsilon.shape[0] != epsilon.shape[1]:
            raise ValueError(
                f"epsilon must be square matrix [num_types, num_types], "
                f"got shape {epsilon.shape}"
            )
        
        # Register as buffers (not trainable parameters by default)
        self.register_buffer("epsilon", epsilon)
        self.register_buffer("sigma", sigma)
        self.cutoff = cutoff
    
    def forward(self, data) -> torch.Tensor:
        """Compute LJ126 energy.
        
        Args:
            data: AtomicTD or Frame with fields:
                - ["atoms"]["x"]: Positions [N, 3]
                - ["atoms"]["type"]: Atom types [N]
                - ["graph"]["batch"]: Batch indices [N]
                
        Returns:
            Total LJ energy (scalar)
        """
        # Extract data using nested access (works for both Frame and AtomicTD)
        pos = data["atoms"]["x"]
        atom_types = data["atoms"]["type"]
        batch = data["graph"]["batch"]
        
        # Convert numpy to torch if needed (for Frame compatibility)
        if not isinstance(pos, torch.Tensor):
            import numpy as np
            pos = torch.from_numpy(pos).float()
            atom_types = torch.from_numpy(atom_types).long()
            batch = torch.from_numpy(batch).long()
        
        # Build neighbor list
        edge_index, edge_vec = radius_graph(pos, batch, cutoff=self.cutoff)
        
        # Get atom types for each edge
        type_i = atom_types[edge_index[0]]  # [num_edges]
        type_j = atom_types[edge_index[1]]  # [num_edges]
        
        # Look up parameters for each edge
        epsilon_ij = self.epsilon[type_i, type_j]  # [num_edges]
        sigma_ij = self.sigma[type_i, type_j]  # [num_edges]
        
        # Compute distances
        distances = torch.norm(edge_vec, dim=-1)  # [num_edges]
        
        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-6)
        
        # Compute LJ potential
        # E = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
        sigma_over_r = sigma_ij / distances
        sigma_over_r_6 = sigma_over_r ** 6
        sigma_over_r_12 = sigma_over_r_6 ** 2
        
        energy_per_edge = 4.0 * epsilon_ij * (sigma_over_r_12 - sigma_over_r_6)
        
        # Sum over all edges (divide by 2 to avoid double counting)
        total_energy = 0.5 * energy_per_edge.sum()
        
        return total_energy
    
    def __repr__(self) -> str:
        num_types = self.epsilon.shape[0]
        return f"LJ126(num_types={num_types}, cutoff={self.cutoff})"
