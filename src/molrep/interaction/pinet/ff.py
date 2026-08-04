"""Feed-forward building block for PiNet layers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from molix import config


def activation_from_name(name: str | type[nn.Module] | None) -> type[nn.Module] | None:
    """Resolve an activation name or module class to a zero-arg module type."""
    if name is None:
        return None
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    table: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
    }
    try:
        return table[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}.") from exc


class FFLayer(nn.Module):
    """Feed-forward stack applied to the last tensor dimension.

    All linears are fully specified at construction (no ``LazyLinear``) so the
    module is export / ``torch.compile`` / functorch safe without a warm-up
    materialisation pass.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        activation: str | type[nn.Module] | None = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        act_cls = activation_from_name(activation)
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for width in n_nodes:
            w = int(width)
            layers.append(nn.Linear(prev, w, bias=use_bias, dtype=config.ftype))
            if act_cls is not None:
                layers.append(act_cls())
            prev = w
        self.layers = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.output_dim = prev if n_nodes else int(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) if len(self.layers) else x
