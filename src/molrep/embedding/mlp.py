"""Variance-preserving scalar MLP — port of ``nequip.nn.mlp.ScalarMLPFunction``.

Lives in :mod:`molrep` so any encoder (``molzoo``) or head (``molpot``)
can compose it without taking a cross-package dependency. The reference
implementation is what the Allegro paper (Musaelian et al., 2023) uses
end-to-end; sharing this exact init across encoder *and* readout is the
only way to keep forward activation variance constant through the whole
graph (the encoder's careful init is otherwise undone by a default-init
readout immediately afterwards).

Reference:
    ``nequip/nn/mlp.py::ScalarLinearLayer``
    ``nequip/nn/mlp.py::ScalarMLPFunction``
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from molix import config


class ScalarLinearLayer(nn.Module):
    """Linear layer with ``α``-scaled weights and reference's variance-preserving init.

    Mirrors ``nequip.nn.mlp.ScalarLinearLayer``: weights are drawn from
    ``U(-√3, √3)`` (so per-element variance is 1) and rescaled by a
    constant ``α`` at every forward, where the caller computes
    ``α = gain / √fan_in``. This keeps the underlying weights themselves
    unit-variance (good for optimisation) while preserving forward
    activation variance after multiplication by α.

    No bias by default, matching the reference's ``bias=False`` everywhere.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), dtype=config.ftype), persistent=False
        )
        self.weight = nn.Parameter(torch.empty((in_features, out_features), dtype=config.ftype))
        nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=config.ftype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.alpha
        if self.bias is None:
            return x @ w
        return torch.addmm(self.bias, x, w)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, alpha={self.alpha.item():.6f}"
        )


class ScalarMLPFunction(nn.Module):
    """Forward-variance-preserving MLP. Mirrors ``nequip.nn.mlp.ScalarMLPFunction``.

    Per-element weight std is ``1`` (uniform init); each Linear's effective
    weight is multiplied by ``α = gain / √fan_in`` at forward time. With
    ``forward_weight_init=True`` (the reference default), the gain is
    ``√2`` for hidden layers (matches ReLU/SiLU forward variance) and
    ``1`` for the first layer.

    Args:
        input_dim: Input feature dim.
        output_dim: Output feature dim.
        hidden_layers_depth: Number of hidden layers (``0`` → bare linear).
        hidden_layers_width: Hidden width. Required if depth > 0.
        nonlinearity: ``nn.Module`` class to insert between hidden layers,
            or ``None`` for a deep linear stack.
        bias: Whether to include bias on every linear (reference: ``False``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: Optional[int] = None,
        nonlinearity: Optional[type[nn.Module]] = nn.SiLU,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_layers_depth != 0:
            assert (
                hidden_layers_depth > 0
                and hidden_layers_width is not None
                and (hidden_layers_width > 0)
            ), "hidden_layers_width must be a positive int when depth > 0"
        hidden_dims = [hidden_layers_width] * hidden_layers_depth if hidden_layers_depth > 0 else []
        self.dims = [input_dim, *hidden_dims, output_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        self.is_nonlinear = (nonlinearity is not None) and (self.num_layers > 1)

        layers: list[nn.Module] = []
        for layer_idx, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            gain = 1.0 if (nonlinearity is None or layer_idx == 0) else math.sqrt(2.0)
            layers.append(
                ScalarLinearLayer(
                    in_features=h_in,
                    out_features=h_out,
                    alpha=gain / math.sqrt(h_in),
                    bias=bias,
                )
            )
            if layer_idx != self.num_layers - 1 and nonlinearity is not None:
                layers.append(nonlinearity())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
