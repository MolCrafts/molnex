"""Pooling functions for aggregating atom features to graph-level features.

Provides masked pooling operations and scatter-based pooling for handling
both variable-length molecules with padding and batched graph representations.
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict


class PoolingSpec(BaseModel):
    """Configuration for scatter pooling."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    strategy: str = Field("sum", pattern="^(sum|mean|none)$")


class ScatterPooling(nn.Module):
    """Generic scatter-based pooling for batched graph data.
    
    Supports pooling node-level features to graph-level features using scatter
    operations. Handles both single graphs and batched graphs with batch indices.
    
    This is useful for aggregating properties across entire molecular graphs,
    as required by many GNN architectures for graph-level predictions.
    
    Example:
        Aggregate node-level energies to graph-level total energy:
        
        >>> pooling = ScatterPooling(strategy="sum")
        >>> node_energies = torch.randn(100, 1)  # 100 atoms
        >>> batch = torch.tensor([0]*25 + [1]*30 + [2]*45)  # 3 graphs
        >>> graph_energies = pooling(node_energies, batch)
        >>> graph_energies.shape
        torch.Size([3, 1])
    """
    
    def __init__(self, strategy: str = "sum"):
        """Initialize ScatterPooling.
        
        Args:
            strategy: Pooling strategy ("sum", "mean", "none").
                - "sum": Sum node features to get graph features
                - "mean": Average node features to get graph features
                - "none": Return node features unchanged
        """
        super().__init__()
        if strategy not in ("sum", "mean", "none"):
            raise ValueError(f"strategy must be 'sum', 'mean', or 'none', got '{strategy}'")
        self.strategy = strategy
    
    def forward(
        self,
        y: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply pooling to node-level predictions.
        
        Args:
            y: Node-level features, shape $(N, D)$ where $N$ is number of atoms
            batch: Optional batch indices, shape $(N,)$. If None, treats all nodes
                as single graph. For batched graphs, contains graph ID per node.
        
        Returns:
            Graph-level features. Shape:
                - $(N, D)$ if strategy="none"
                - $(1, D)$ if single graph and strategy in {"sum", "mean"}
                - $(B, D)$ if batched graphs and strategy in {"sum", "mean"}
                  where $B$ is number of graphs.
        
        Mathematical formulas:
            - **Sum pooling**: $y_g = \\sum_{i \\in \\mathcal{G}_g} y_i$
            - **Mean pooling**: $y_g = \\frac{1}{|\\mathcal{G}_g|} \\sum_{i \\in \\mathcal{G}_g} y_i$
            
            where $\\mathcal{G}_g$ denotes nodes in graph $g$.
        """
        if self.strategy == "none":
            return y
        
        if batch is None:
            # Single graph: pool across all nodes
            if self.strategy == "sum":
                return y.sum(dim=0, keepdim=True)
            else:  # mean
                return y.mean(dim=0, keepdim=True)
        
        # Batched graphs: use scatter operations
        n_graphs = int(batch.max().item()) + 1
        output = torch.zeros(
            (n_graphs, y.shape[1]),
            dtype=y.dtype,
            device=y.device,
        )
        
        if self.strategy == "sum":
            output.scatter_add_(
                0,
                batch.unsqueeze(-1).expand_as(y),
                y
            )
        else:  # mean
            output.scatter_add_(
                0,
                batch.unsqueeze(-1).expand_as(y),
                y
            )
            # Divide by count per graph
            counts = torch.zeros(
                (n_graphs,),
                dtype=y.dtype,
                device=y.device,
            )
            counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=y.dtype))
            output = output / counts.unsqueeze(-1).clamp(min=1e-8)
        
        return output


def masked_sum_pooling(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum pooling over valid atoms only.
    
    Aggregates atom-level features to graph-level features by summing
    over valid (non-padded) atoms.
    
    Args:
        features: Atom features [B, L, d_model]
        mask: Atom mask [B, L] (True = valid atom, False = padding)
        
    Returns:
        Graph-level features [B, d_model]
        
    Example:
        >>> features = torch.randn(2, 10, 128)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> mask[0, 7:] = False  # First molecule has 7 atoms
        >>> mask[1, 5:] = False  # Second molecule has 5 atoms
        >>> graph_features = masked_sum_pooling(features, mask)
        >>> graph_features.shape
        torch.Size([2, 128])
    """
    # Mask out padding positions
    masked_features = features * mask.unsqueeze(-1).float()  # [B, L, d_model]
    
    # Sum over atoms
    graph_features = masked_features.sum(dim=1)  # [B, d_model]
    
    return graph_features


def masked_mean_pooling(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling over valid atoms only.
    
    Aggregates atom-level features to graph-level features by averaging
    over valid (non-padded) atoms.
    
    Args:
        features: Atom features [B, L, d_model]
        mask: Atom mask [B, L] (True = valid atom, False = padding)
        
    Returns:
        Graph-level features [B, d_model]
        
    Example:
        >>> features = torch.randn(2, 10, 128)
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> mask[0, 7:] = False  # First molecule has 7 atoms
        >>> mask[1, 5:] = False  # Second molecule has 5 atoms
        >>> graph_features = masked_mean_pooling(features, mask)
        >>> graph_features.shape
        torch.Size([2, 128])
    """
    # Mask out padding positions
    masked_features = features * mask.unsqueeze(-1).float()  # [B, L, d_model]
    
    # Count valid atoms per molecule
    num_atoms = mask.sum(dim=1, keepdim=True).float()  # [B, 1]
    num_atoms = num_atoms.clamp(min=1)  # Avoid division by zero
    
    # Sum and normalize
    graph_features = masked_features.sum(dim=1) / num_atoms  # [B, d_model]
    
    return graph_features
