import torch
import torch.nn as nn
from ..data import Batch
from .base import Encoder

class GeometricEncoder(nn.Module, Encoder):
    """
    3D Encoder using radial basis functions for distances.
    Naive implementation for baseline (Cutoff graph).
    """
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, num_atom_types: int = 100, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.layers = nn.ModuleList([
             nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.activation = nn.ReLU()
        
    def forward(self, batch: Batch) -> torch.Tensor:
        # 1. Embed atoms
        x = self.embedding(batch.z)
        
        # 2. Compute distances (Should use neighbor list for efficiency in prod, this is baseline naive O(N^2) block-wise)
        # For baseline, we assume small batch size or implement simple masking
        
        # To avoid O(N^2) on the whole batch, strictly we should use `batch` vector to mask.
        # Minimal implementation: just use coords.
        
        # Note: This is an unoptimized baseline MVP.
        # Future: Use torch-cluster radius_graph
        
        pos = batch.pos
        
        # Only computing local neighbors within batch is tricky without scatter/cluster ops.
        # Fallback: Treat edges as neighborhood if available, OR just do simple MLP on features for MVP if radius graph is too slow.
        # BUT requirement says "geometry-aware". 
        
        # Let's implement a very simple RBF attention if edge_index is present, weighting by distance.
        # If no edge_index, we can't do message passing easily without deps.
        # Assuming edge_index provided by Topology or generated on the fly.
        
        edge_index = batch.edge_index
        if edge_index is not None:
            row, col = edge_index
            dist = (pos[row] - pos[col]).norm(dim=-1)
            
            # Simple RBF expansion
            rbf = torch.exp(-dist) # Simple decay
            
            for layer in self.layers:
                msg = x[col] * rbf.unsqueeze(-1)
                
                out = torch.zeros_like(x)
                out.index_add_(0, row, msg)
                
                x = x + out
                x = layer(x)
                x = self.activation(x)

        return x
