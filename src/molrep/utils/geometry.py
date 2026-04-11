"""Geometry plumbing utilities for molrep."""

import math
import torch
import torch.nn as nn


class NeighborGraphBuilder:
    """Build neighbor graphs from atomic positions.
    
    Uses efficient PyTorch operations for radius-based neighbor search.
    """
    
    def __init__(self, cutoff: float, max_neighbors: int | None = None):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
    
    def __call__(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build neighbor graph.
        
        Args:
            positions: Atomic positions [N, 3]
            batch: Molecule indices [N]
            
        Returns:
            edge_index [E, 2], edge_vec [E, 3], edge_dist [E]
        """
        device = positions.device
        num_mols = int(batch.max().item()) + 1
        
        edge_indices = []
        edge_vecs = []
        edge_dists = []
        
        for mol_idx in range(num_mols):
            mask = batch == mol_idx
            mol_pos = positions[mask]
            mol_indices = torch.where(mask)[0]
            n = mol_pos.shape[0]
            
            if n == 0:
                continue
            
            # Pairwise distances
            diff = mol_pos.unsqueeze(0) - mol_pos.unsqueeze(1)  # [N, N, 3]
            dist = diff.norm(dim=-1)  # [N, N]
            
            # Neighbor mask
            neighbor_mask = (dist < self.cutoff) & (dist > 0)
            
            # Apply max_neighbors if specified
            if self.max_neighbors is not None and n > self.max_neighbors:
                dist_masked = dist.clone()
                dist_masked[~neighbor_mask] = float('inf')
                _, topk_idx = dist_masked.topk(
                    min(self.max_neighbors, n - 1), dim=1, largest=False
                )
                new_mask = torch.zeros_like(neighbor_mask)
                for i in range(n):
                    new_mask[i, topk_idx[i]] = neighbor_mask[i, topk_idx[i]]
                neighbor_mask = new_mask
            
            src, dst = torch.where(neighbor_mask)
            if len(src) > 0:
                global_src = mol_indices[src]
                global_dst = mol_indices[dst]
                edge_indices.append(torch.stack([global_src, global_dst], dim=1))
                edge_vecs.append(diff[src, dst])
                edge_dists.append(dist[src, dst])
        
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=0)
            edge_vec = torch.cat(edge_vecs, dim=0)
            edge_dist = torch.cat(edge_dists, dim=0)
        else:
            edge_index = torch.zeros((0, 2), dtype=torch.long, device=device)
            edge_vec = torch.zeros((0, 3), dtype=positions.dtype, device=device)
            edge_dist = torch.zeros((0,), dtype=positions.dtype, device=device)
        
        return edge_index, edge_vec, edge_dist


class SphericalBasis(nn.Module):
    """Spherical harmonics basis for angular features."""
    
    def __init__(self, max_l: int = 2, normalize: bool = True):
        super().__init__()
        self.max_l = max_l
        self.normalize = normalize
        self.num_features = (max_l + 1) ** 2
    
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics from direction vectors."""
        norm = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x, y, z = (vectors / norm).unbind(dim=-1)
        
        outputs = []
        
        # l=0
        if self.max_l >= 0:
            c0 = 0.5 * math.sqrt(1 / math.pi)
            outputs.append(torch.ones_like(x) * c0)
        
        # l=1
        if self.max_l >= 1:
            c1 = 0.5 * math.sqrt(3 / math.pi)
            outputs.extend([c1 * y, c1 * z, c1 * x])
        
        # l=2
        if self.max_l >= 2:
            c2_0 = 0.25 * math.sqrt(5 / math.pi)
            c2_1 = 0.5 * math.sqrt(15 / math.pi)
            c2_2 = 0.25 * math.sqrt(15 / math.pi)
            outputs.extend([
                c2_2 * (x * y),
                c2_1 * (y * z),
                c2_0 * (3 * z * z - 1),
                c2_1 * (x * z),
                c2_2 * (x * x - y * y),
            ])
        
        return torch.stack(outputs, dim=-1)


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions."""
    
    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0, trainable: bool = False):
        super().__init__()
        centers = torch.linspace(0, cutoff, num_rbf)
        widths = torch.full((num_rbf,), cutoff / num_rbf)
        
        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((distances.unsqueeze(-1) - self.centers) ** 2) / (self.widths ** 2))


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function."""
    
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1)
        return cutoffs * (distances < self.cutoff).float()
