"""Prediction heads for molecular properties.

Provides TensorDict-based prediction heads for various molecular properties,
starting with scalar energy prediction.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molrep.readout.pooling import masked_sum_pooling, masked_mean_pooling


class EnergyHead(TensorDictModuleBase):
    """Prediction head for scalar molecular energy.
    
    Takes atom-level features, pools them to graph-level, and predicts
    a scalar energy value per molecule.
    
    Args:
        d_model: Input feature dimension
        hidden_dim: Hidden layer dimension (default: same as d_model)
        num_layers: Number of MLP layers (default: 2)
        pooling: Pooling strategy - "sum" or "mean" (default: "mean")
        
    Example:
        >>> head = EnergyHead(d_model=128, pooling="mean")
        >>> td = TensorDict({
        ...     ("rep", "h"): torch.randn(2, 10, 128),  # [B, L, d_model]
        ...     ("atoms", "mask"): torch.ones(2, 10, dtype=torch.bool),
        ... }, batch_size=[])
        >>> td = head(td)
        >>> td["pred", "energy"].shape
        torch.Size([2, 1])
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.in_keys = [("rep", "h"), ("atoms", "mask")]
        self.out_keys = [("pred", "energy")]
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model
        self.num_layers = num_layers
        self.pooling = pooling
        
        if pooling not in ["sum", "mean"]:
            raise ValueError(f"pooling must be 'sum' or 'mean', got {pooling}")
        
        # Build MLP
        layers = []
        in_dim = d_model
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
            ])
            in_dim = self.hidden_dim
        
        # Final layer to scalar
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Predict molecular energy.
        
        Args:
            td: TensorDict with:
                - ("rep", "h"): Atom features [B, L, d_model]
                - ("atoms", "mask"): Atom mask [B, L]
                
        Returns:
            TensorDict with ("pred", "energy"): [B, 1] added
        """
        h = td["rep", "h"]  # [B, L, d_model]
        mask = td["atoms", "mask"]  # [B, L]
        
        # Pool to graph-level features
        if self.pooling == "sum":
            h_graph = masked_sum_pooling(h, mask)  # [B, d_model]
        else:  # mean
            h_graph = masked_mean_pooling(h, mask)  # [B, d_model]
        
        # Predict energy
        energy = self.mlp(h_graph)  # [B, 1]
        
        td["pred", "energy"] = energy
        return td
    
    def __repr__(self) -> str:
        return (
            f"EnergyHead(d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, pooling={self.pooling})"
        )
