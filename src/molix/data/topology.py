"""Data pipeline TensorDictModule components.

All components properly inherit from TensorDictModule.
"""

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class TopologyBuilder(TensorDictModule):
    """Build neighbor topology from atomic positions.
    
    in_keys: [("atoms", "x"), ("graph", "batch")]
    out_keys: [("bonds", "i"), ("bonds", "j"), ("bonds", "vec"), ("bonds", "dist")]
    """
    
    def __init__(self, cutoff: float = 5.0, max_neighbors: int | None = None):
        """Initialize topology builder.
        
        Args:
            cutoff: Distance cutoff for neighbor search (Angstroms)
            max_neighbors: Maximum neighbors per atom (optional)
        """
        module = _TopologyBuilderModule(cutoff, max_neighbors)
        super().__init__(
            module=module,
            in_keys=[("atoms", "x"), ("graph", "batch")],
            out_keys=[("bonds", "i"), ("bonds", "j"), ("bonds", "vec"), ("bonds", "dist")],
        )


class _TopologyBuilderModule(torch.nn.Module):
    """Internal module for TopologyBuilder."""
    
    def __init__(self, cutoff: float, max_neighbors: int | None):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
    
    def forward(self, atoms_x: torch.Tensor, graph_batch: torch.Tensor):
        """Build topology.
        
        Args:
            atoms_x: Atomic positions [N, 3]
            graph_batch: Molecule indices [N]
            
        Returns:
            Tuple of (bond_i, bond_j, bond_vec, bond_dist)
        """
        device = atoms_x.device
        dtype = atoms_x.dtype
        num_mols = int(graph_batch.max().item()) + 1
        
        bond_i_list = []
        bond_j_list = []
        bond_vec_list = []
        bond_dist_list = []
        
        for mol_idx in range(num_mols):
            mask = graph_batch == mol_idx
            mol_pos = atoms_x[mask]
            mol_indices = torch.where(mask)[0]
            n = mol_pos.shape[0]
            
            if n == 0:
                continue
            
            # Pairwise distances
            diff = mol_pos.unsqueeze(0) - mol_pos.unsqueeze(1)
            dist = diff.norm(dim=-1)
            
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
                bond_i_list.append(global_src)
                bond_j_list.append(global_dst)
                bond_vec_list.append(diff[src, dst])
                bond_dist_list.append(dist[src, dst])
        
        if bond_i_list:
            bond_i = torch.cat(bond_i_list, dim=0)
            bond_j = torch.cat(bond_j_list, dim=0)
            bond_vec = torch.cat(bond_vec_list, dim=0)
            bond_dist = torch.cat(bond_dist_list, dim=0)
        else:
            bond_i = torch.zeros(0, dtype=torch.long, device=device)
            bond_j = torch.zeros(0, dtype=torch.long, device=device)
            bond_vec = torch.zeros((0, 3), dtype=dtype, device=device)
            bond_dist = torch.zeros(0, dtype=dtype, device=device)
        
        return bond_i, bond_j, bond_vec, bond_dist


class GeometryPreprocessor(TensorDictModule):
    """Precompute geometric features from topology.
    
    in_keys: [("atoms", "x"), ("bonds", "i"), ("bonds", "j")]
    out_keys: [("bonds", "vec"), ("bonds", "dist")]
    """
    
    def __init__(self):
        """Initialize geometry preprocessor."""
        module = _GeometryPreprocessorModule()
        super().__init__(
            module=module,
            in_keys=[("atoms", "x"), ("bonds", "i"), ("bonds", "j")],
            out_keys=[("bonds", "vec"), ("bonds", "dist")],
        )


class _GeometryPreprocessorModule(torch.nn.Module):
    """Internal module for GeometryPreprocessor."""
    
    def forward(self, atoms_x: torch.Tensor, bonds_i: torch.Tensor, bonds_j: torch.Tensor):
        """Compute bond vectors and distances.
        
        Args:
            atoms_x: Atomic positions [N, 3]
            bonds_i: Bond source indices [E]
            bonds_j: Bond target indices [E]
            
        Returns:
            Tuple of (bond_vec, bond_dist)
        """
        if len(bonds_i) > 0:
            bond_vec = atoms_x[bonds_j] - atoms_x[bonds_i]
            bond_dist = torch.norm(bond_vec, dim=-1)
        else:
            bond_vec = torch.zeros((0, 3), dtype=atoms_x.dtype, device=atoms_x.device)
            bond_dist = torch.zeros(0, dtype=atoms_x.dtype, device=atoms_x.device)
        
        return bond_vec, bond_dist


class Normalizer(TensorDictModule):
    """Normalize atomic positions.
    
    in_keys: [("atoms", "x"), ("graph", "batch")]
    out_keys: [("atoms", "x")]
    """
    
    def __init__(self, center: bool = True):
        """Initialize normalizer.
        
        Args:
            center: Whether to center molecules at origin
        """
        module = _NormalizerModule(center)
        super().__init__(
            module=module,
            in_keys=[("atoms", "x"), ("graph", "batch")],
            out_keys=[("atoms", "x")],
        )


class _NormalizerModule(torch.nn.Module):
    """Internal module for Normalizer."""
    
    def __init__(self, center: bool):
        super().__init__()
        self.center = center
    
    def forward(self, atoms_x: torch.Tensor, graph_batch: torch.Tensor):
        """Normalize positions.
        
        Args:
            atoms_x: Atomic positions [N, 3]
            graph_batch: Molecule indices [N]
            
        Returns:
            Normalized positions [N, 3]
        """
        if not self.center:
            return atoms_x
        
        # Center each molecule
        num_mols = int(graph_batch.max().item()) + 1
        centered_positions = atoms_x.clone()
        
        for mol_idx in range(num_mols):
            mask = graph_batch == mol_idx
            mol_pos = atoms_x[mask]
            if len(mol_pos) > 0:
                center = mol_pos.mean(dim=0)
                centered_positions[mask] = mol_pos - center
        
        return centered_positions
