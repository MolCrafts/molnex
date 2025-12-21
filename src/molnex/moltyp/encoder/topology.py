import torch
import torch.nn as nn
from ..data import Batch
from .base import Encoder

class TopologyEncoder(nn.Module, Encoder):
    """
    Simple GNN encoder using node features and connectivity.
    """
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, num_atom_types: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.activation = nn.ReLU()

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.embedding(batch.z)
        
        edge_index = batch.edge_index
        
        # Simple message passing: x_i' = x_i + sum(x_j)
        # Using native scatter_add_ for simple GCN-like aggregation
        for layer in self.layers:
            # Message aggregation
            if edge_index is not None:
                row, col = edge_index
                
                # Message 
                msg = x[col]
                
                # Aggregate
                out = torch.zeros_like(x)
                out.index_add_(0, row, msg)
                
                # Update
                x = x + out
            
            x = layer(x)
            x = self.activation(x)
            
        return x
