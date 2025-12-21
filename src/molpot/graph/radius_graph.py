"""Pure PyTorch radius graph construction.

Builds molecular graphs from atomic positions without PyTorch Geometric dependency.
"""

from typing import Tuple

import torch
import torch.nn as nn


class RadiusGraph(nn.Module):
    """Radius graph constructor.
    
    Builds edges between atoms within cutoff distance, respecting batch boundaries.
    Uses pure PyTorch operations (no PyTorch Geometric).
    
    Attributes:
        cutoff: Cutoff radius for neighbor search
        max_neighbors: Maximum number of neighbors per atom
    """
    
    def __init__(
        self,
        cutoff: float = 5.0,
        max_neighbors: int = 32,
    ):
        """Initialize radius graph constructor.
        
        Args:
            cutoff: Cutoff radius for neighbor search
            max_neighbors: Maximum number of neighbors per atom (for memory efficiency)
        """
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct radius graph from atomic positions.
        
        Args:
            pos: Atomic positions [N, 3]
            batch: Batch/molecule indices [N]
            
        Returns:
            edge_index: Edge connectivity [2, num_edges] (LongTensor)
            edge_vec: Edge displacement vectors [num_edges, 3] (FloatTensor)
        """
        num_atoms = pos.size(0)
        device = pos.device
        
        # Compute pairwise distances
        pos_i = pos.unsqueeze(1)  # [N, 1, 3]
        pos_j = pos.unsqueeze(0)  # [1, N, 3]
        
        edge_vec_all = pos_j - pos_i  # [N, N, 3]
        distances = torch.norm(edge_vec_all, dim=-1)  # [N, N]
        
        # Mask: same batch and within cutoff (exclude self-loops)
        batch_i = batch.unsqueeze(1)  # [N, 1]
        batch_j = batch.unsqueeze(0)  # [1, N]
        
        same_batch = (batch_i == batch_j)
        within_cutoff = (distances <= self.cutoff)
        not_self = ~torch.eye(num_atoms, dtype=torch.bool, device=device)
        
        mask = same_batch & within_cutoff & not_self
        
        # Get edge indices
        edge_index = mask.nonzero(as_tuple=False).t()  # [2, num_edges]
        
        # Limit neighbors per atom
        edge_index = self._limit_neighbors(edge_index, num_atoms, device)
        
        # Get edge vectors
        edge_vec = self._get_edge_vectors(edge_vec_all, edge_index, pos.dtype, device)
        
        return edge_index, edge_vec
    
    def _limit_neighbors(
        self,
        edge_index: torch.Tensor,
        num_atoms: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Limit number of neighbors per atom."""
        num_edges = edge_index.size(1)
        
        # Early return for empty or small graphs
        empty_check = num_edges == 0
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        
        # Count neighbors per source atom
        source_atoms = edge_index[0]
        
        # Sample neighbors for atoms with too many
        for atom_idx in range(num_atoms):
            atom_edges = (source_atoms == atom_idx).nonzero(as_tuple=False).squeeze(-1)
            num_atom_edges = len(atom_edges)
            
            # Mark edges to keep
            over_limit = num_atom_edges > self.max_neighbors
            keep_mask[atom_edges] = ~over_limit
            
            # Sample if over limit
            sampled_indices = torch.randperm(num_atom_edges, device=device)[:self.max_neighbors]
            keep_mask[atom_edges[sampled_indices]] = over_limit
        
        return edge_index[:, keep_mask * ~empty_check + empty_check]
    
    def _get_edge_vectors(
        self,
        edge_vec_all: torch.Tensor,
        edge_index: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract edge vectors from full pairwise matrix."""
        has_edges = edge_index.size(1) > 0
        
        edge_vec = edge_vec_all[edge_index[0], edge_index[1]] * has_edges
        empty_vec = torch.zeros((0, 3), dtype=dtype, device=device)
        
        return edge_vec * has_edges + empty_vec * ~has_edges

def radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 5.0,
    max_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct radius graph from atomic positions.
    
    Builds edges between atoms within cutoff distance, respecting batch boundaries.
    Uses pure PyTorch operations (no PyTorch Geometric).
    
    Args:
        pos: Atomic positions [N, 3]
        batch: Batch/molecule indices [N]
        cutoff: Cutoff radius for neighbor search
        max_neighbors: Maximum number of neighbors per atom (for memory efficiency)
        
    Returns:
        edge_index: Edge connectivity [2, num_edges] (LongTensor)
        edge_vec: Edge displacement vectors [num_edges, 3] (FloatTensor)
        
    Example:
        >>> pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        >>> batch = torch.tensor([0, 0, 0])
        >>> edge_index, edge_vec = radius_graph(pos, batch, cutoff=2.0)
        >>> edge_index.shape  # [2, num_edges]
        >>> edge_vec.shape  # [num_edges, 3]
    """
    num_atoms = pos.size(0)
    device = pos.device
    
    # Compute pairwise distances
    # pos[i] - pos[j] for all i, j
    pos_i = pos.unsqueeze(1)  # [N, 1, 3]
    pos_j = pos.unsqueeze(0)  # [1, N, 3]
    
    edge_vec_all = pos_j - pos_i  # [N, N, 3]
    distances = torch.norm(edge_vec_all, dim=-1)  # [N, N]
    
    # Mask: same batch and within cutoff (exclude self-loops)
    batch_i = batch.unsqueeze(1)  # [N, 1]
    batch_j = batch.unsqueeze(0)  # [1, N]
    
    same_batch = (batch_i == batch_j)
    within_cutoff = (distances <= cutoff)
    not_self = ~torch.eye(num_atoms, dtype=torch.bool, device=device)
    
    mask = same_batch & within_cutoff & not_self
    
    # Get edge indices
    edge_index = mask.nonzero(as_tuple=False).t()  # [2, num_edges]
    
    # Limit neighbors per atom if needed
    if max_neighbors is not None and edge_index.size(1) > 0:
        # Count neighbors per source atom
        source_atoms = edge_index[0]
        unique_sources, counts = torch.unique(source_atoms, return_counts=True)
        
        # If any atom has too many neighbors, sample
        if counts.max() > max_neighbors:
            keep_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
            
            for atom_idx in range(num_atoms):
                atom_edges = (source_atoms == atom_idx).nonzero(as_tuple=False).squeeze(-1)
                
                if len(atom_edges) > max_neighbors:
                    # Sample max_neighbors edges
                    perm = torch.randperm(len(atom_edges), device=device)[:max_neighbors]
                    keep_mask[atom_edges[perm]] = True
                else:
                    keep_mask[atom_edges] = True
            
            edge_index = edge_index[:, keep_mask]
    
    # Get edge vectors
    if edge_index.size(1) > 0:
        edge_vec = edge_vec_all[edge_index[0], edge_index[1]]  # [num_edges, 3]
    else:
        edge_vec = torch.zeros((0, 3), dtype=pos.dtype, device=device)
    
    return edge_index, edge_vec
