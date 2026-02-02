"""Keyed MLP for TensorDict inputs."""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


Key = str | tuple[str, ...]


class KeyedMLPSpec(BaseModel):
    """Specification for a keyed MLP."""

    input_key: Key
    output_key: Key
    in_dim: int = Field(..., gt=0)
    hidden_dims: list[int] = Field(..., min_length=1)
    out_dim: int = Field(..., gt=0)
    activation: str = Field("silu", pattern="^(silu|relu|gelu|tanh)$")
    use_bias: bool = True

    @property
    def key(self) -> Key:
        return self.input_key


class KeyedMLP(nn.Module):
    """MLP that operates on specified keys of a container (dict or AtomTD)."""

    def __init__(
        self,
        *,
        input_key: str,
        output_key: str,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: str = "silu",
        use_bias: bool = True,
    ):
        super().__init__()

        self.input_key = input_key
        self.output_key = output_key

        activation_map = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        act_fn = activation_map[activation.lower()]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dims[0], bias=use_bias))
        layers.append(act_fn)

        for idx in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1], bias=use_bias))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dims[-1], out_dim, bias=use_bias))

        self.mlp = nn.Sequential(*layers)

    def forward(self, data: Any) -> Any:
        """Apply MLP to input_key and store result in output_key.
        
        Args:
            data: dict or dataclass (e.g. AtomTD) containing the input_key
            
        Returns:
            The modified data container
        """
        # Extract features
        if isinstance(data, dict):
            features = data[self.input_key]
        else:
            features = getattr(data, self.input_key)
            
        # Apply MLP
        out = self.mlp(features)
        
        # Store result
        if isinstance(data, dict):
            data[self.output_key] = out
        else:
            setattr(data, self.output_key, out)
            
        return data
