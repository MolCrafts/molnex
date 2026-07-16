"""Property ↔ interaction message layers (scalar + equivariant)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from molix import config

from .ff import FFLayer


class PILayer(nn.Module):
    """Property-to-interaction layer for scalar ``P1`` features."""

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        n_basis: int,
        activation: str | type[nn.Module] | None = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if not n_nodes:
            raise ValueError("PILayer requires at least one output width.")
        if int(in_dim) <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        self.n_basis = int(n_basis)
        self.out_dim = int(n_nodes[-1])
        widths = [int(v) for v in n_nodes]
        widths[-1] = widths[-1] * self.n_basis
        self.ff_layer = FFLayer(
            widths, in_dim=int(in_dim), activation=activation, use_bias=use_bias
        )

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        prop: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        inter = torch.cat([prop[src], prop[dst]], dim=-1)
        weights = self.ff_layer(inter).reshape(-1, self.out_dim, self.n_basis)
        # Broadcast multiply + sum (not einsum) so torch.compile fuses into one
        # triton reduction instead of a memory-bound cuBLAS gemv.
        return (weights * basis.unsqueeze(1)).sum(-1)


class IPLayer(nn.Module):
    """Interaction-to-property scatter sum."""

    def forward(
        self,
        src: torch.Tensor,
        prop: torch.Tensor,
        inter: torch.Tensor,
    ) -> torch.Tensor:
        out = prop.new_zeros(prop.shape[0], *inter.shape[1:])
        # Accumulate in ``out``'s dtype under AMP (fp32 reduction is correct).
        out.index_add_(0, src, inter.to(out.dtype))
        return out


class PIXLayer(nn.Module):
    """Equivariant property-to-interaction layer."""

    def __init__(self, *, channels: int, weighted: bool = False) -> None:
        super().__init__()
        self.weighted = bool(weighted)
        self.channels = int(channels)
        if self.weighted:
            self.wi = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)
            self.wj = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        px_i = px[src]
        px_j = px[dst]
        if self.weighted:
            return self.wi(px_i) + self.wj(px_j)
        return px_j


class ScaleLayer(nn.Module):
    """Scale an equivariant tensor by scalar channels."""

    def forward(self, px: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        return px * p1.unsqueeze(-2)


class DotLayer(nn.Module):
    """Dot product over equivariant spatial components."""

    def __init__(self, *, channels: int, weighted: bool = False) -> None:
        super().__init__()
        self.weighted = bool(weighted)
        self.channels = int(channels)
        if self.weighted:
            self.wi = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)
            self.wj = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        if self.weighted:
            return (self.wi(px) * self.wj(px)).sum(1)
        return (px * px).sum(1)
