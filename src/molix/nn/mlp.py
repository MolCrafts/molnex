"""Keyed MLP over dictionary-like containers."""

from typing import Any

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
    """MLP that reads one key and writes one key in a mutable container."""

    def __init__(
        self,
        *,
        input_key: Key,
        output_key: Key,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: str = "silu",
        use_bias: bool = True,
    ):
        super().__init__()

        self.config = KeyedMLPSpec(
            input_key=input_key,
            output_key=output_key,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            activation=activation,
            use_bias=use_bias,
        )

        self.input_key = self.config.input_key
        self.output_key = self.config.output_key

        activation_map = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        act_fn = activation_map[self.config.activation.lower()]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(self.config.in_dim, self.config.hidden_dims[0], bias=self.config.use_bias))
        layers.append(act_fn)

        for idx in range(len(self.config.hidden_dims) - 1):
            layers.append(
                nn.Linear(
                    self.config.hidden_dims[idx],
                    self.config.hidden_dims[idx + 1],
                    bias=self.config.use_bias,
                )
            )
            layers.append(act_fn)

        layers.append(nn.Linear(self.config.hidden_dims[-1], self.config.out_dim, bias=self.config.use_bias))

        self.mlp = nn.Sequential(*layers)

    def forward(self, data: Any) -> Any:
        """Apply MLP to input_key and store result in output_key.
        
        Args:
            data: dict or dataclass containing the input_key
            
        Returns:
            The modified data container
        """
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
