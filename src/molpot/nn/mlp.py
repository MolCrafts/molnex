"""Multi-layer perceptron (MLP) implementation."""

from typing import List, Optional

import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.
    
    Attributes:
        layers: List of linear layers
        activation: Activation function
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize MLP.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dims: List of hidden layer dimensions (default: [])
            activation: Activation function name ("relu", "silu", "gelu", "tanh")
            use_layer_norm: Whether to use layer normalization after each layer
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = []
        
        # Build layer dimensions
        dims = [in_dim] + hidden_dims + [out_dim]
        
        # Create layers
        layers = []
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation (except for last layer)
            if i < len(dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor [..., in_dim]
            
        Returns:
            Output tensor [..., out_dim]
        """
        return self.net(x)
    
    def __repr__(self) -> str:
        return f"MLP({self.net})"
