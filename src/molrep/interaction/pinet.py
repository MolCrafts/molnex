"""PiNet graph-convolution layers.

PiNet uses a GC-block message-passing architecture with separate scalar (P1),
vector (P3), and rank-5 (P5) property tracks. Each block applies invariant
and equivariant updates to these tracks in parallel.

PyTorch ports of the Keras reference at Teoroo-CMC/PiNN
(``pinn.networks.pinet`` / ``pinn.networks.pinet2``).

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from molix import config


def _activation_from_name(name: str | type[nn.Module] | None) -> type[nn.Module] | None:
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
    """Feed-forward layer applied to the last tensor dimension."""

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        activation: str | type[nn.Module] | None = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        act_cls = _activation_from_name(activation)
        layers: list[nn.Module] = []
        for width in n_nodes:
            layers.append(nn.LazyLinear(int(width), bias=use_bias, dtype=config.ftype))
            if act_cls is not None:
                layers.append(act_cls())
        self.layers = nn.Sequential(*layers)
        self.output_dim = int(n_nodes[-1]) if n_nodes else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) if len(self.layers) else x


class PILayer(nn.Module):
    """Property-to-interaction layer for scalar ``P1`` features."""

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        n_basis: int,
        activation: str | type[nn.Module] | None = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if not n_nodes:
            raise ValueError("PILayer requires at least one output width.")
        self.n_basis = int(n_basis)
        self.out_dim = int(n_nodes[-1])
        widths = [int(v) for v in n_nodes]
        widths[-1] = widths[-1] * self.n_basis
        self.ff_layer = FFLayer(widths, activation=activation, use_bias=use_bias)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        prop: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        inter = torch.cat([prop[src], prop[dst]], dim=-1)
        weights = self.ff_layer(inter).reshape(-1, self.out_dim, self.n_basis)
        return torch.einsum("ecb,eb->ec", weights, basis)


class IPLayer(nn.Module):
    """Interaction-to-property scatter sum."""

    def forward(
        self,
        src: torch.Tensor,
        prop: torch.Tensor,
        inter: torch.Tensor,
    ) -> torch.Tensor:
        out = prop.new_zeros(prop.shape[0], *inter.shape[1:])
        # Accumulate in ``out``'s dtype: under AMP autocast ``inter`` may be a
        # reduced-precision (bf16/fp16) activation while ``out`` follows the
        # fp32 ``prop`` dtype; index_add_ requires matching scalar types and
        # fp32 accumulation is the numerically correct choice for a reduction.
        out.index_add_(0, src, inter.to(out.dtype))
        return out


class ResUpdate(nn.Module):
    """Residual update with optional biasless channel projection."""

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.in_dim == self.out_dim:
            self.transform: nn.Module = nn.Identity()
        else:
            self.transform = nn.Linear(self.in_dim, self.out_dim, bias=False, dtype=config.ftype)

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return self.transform(old) + new


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
    """Dot product over equivariant components."""

    def __init__(self, *, channels: int, weighted: bool = False) -> None:
        super().__init__()
        self.weighted = bool(weighted)
        self.channels = int(channels)
        if self.weighted:
            self.wi = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)
            self.wj = nn.Linear(self.channels, self.channels, bias=False, dtype=config.ftype)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        if self.weighted:
            return torch.einsum("ixr,ixr->ir", self.wi(px), self.wj(px))
        return torch.einsum("ixr,ixr->ir", px, px)


class InvarLayer(nn.Module):
    """Scalar invariant update block: ``PI -> II -> IP -> PP``."""

    def __init__(
        self,
        *,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        self.pi_layer = PILayer(pi_nodes, n_basis=n_basis, activation=activation)
        self.ii_layer = FFLayer(ii_nodes, activation=activation, use_bias=False)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(pp_nodes, activation=activation, use_bias=False)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        p1: torch.Tensor,
        basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i1 = self.pi_layer(src, dst, p1, basis)
        i1 = self.ii_layer(i1)
        p1_new = self.ip_layer(src, p1, i1)
        p1_new = self.pp_layer(p1_new)
        return p1_new, i1


class EquivarLayer(nn.Module):
    """Equivariant update block for ``P3`` or ``P5`` features."""

    def __init__(
        self,
        *,
        channels: int,
        out_channels: int,
        weighted: bool = False,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        del activation  # unused — historical from reference impl
        self.pi_layer = PIXLayer(channels=channels, weighted=weighted)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer([out_channels], activation=None, use_bias=False)
        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer(channels=out_channels, weighted=weighted)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        px: torch.Tensor,
        i1: torch.Tensor,
        diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Equivalent to:
        #     ix = self.scale_layer(self.pi_layer(...), i1)
        #     ix = ix + self.scale_layer(diff.unsqueeze(-1), i1)
        # but with one broadcast multiply instead of two scales + add.
        ix = (self.pi_layer(src, dst, px) + diff.unsqueeze(-1)) * i1.unsqueeze(-2)
        px_new = self.ip_layer(src, px, ix)
        px_new = self.pp_layer(px_new)
        dotted_px = self.dot_layer(px_new)
        return px_new, ix, dotted_px


class GCBlock(nn.Module):
    """One PiNet graph-convolution block."""

    def __init__(
        self,
        *,
        rank: int,
        weighted: bool,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        if rank not in {1, 3, 5}:
            raise ValueError(f"rank must be 1, 3, or 5, got {rank}.")
        if not pp_nodes or not ii_nodes:
            raise ValueError("pp_nodes and ii_nodes must not be empty.")
        if int(pp_nodes[-1]) != int(ii_nodes[-1]):
            raise ValueError("pp_nodes[-1] == ii_nodes[-1] required for scalar gating of P3/P5.")
        self.rank = int(rank)
        self.n_props = int(rank // 2) + 1
        self.feature_dim = int(ii_nodes[-1])

        ii1_nodes = [int(v) for v in ii_nodes]
        ii1_nodes[-1] *= self.n_props
        self.invar_p1_layer = InvarLayer(
            pp_nodes=pp_nodes,
            pi_nodes=pi_nodes,
            ii_nodes=ii1_nodes,
            n_basis=n_basis,
            activation=activation,
        )

        if self.rank >= 3:
            self.equivar_p3_layer = EquivarLayer(
                channels=self.feature_dim,
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                activation=activation,
            )
        if self.rank >= 5:
            self.equivar_p5_layer = EquivarLayer(
                channels=self.feature_dim,
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                activation=activation,
            )
        pp1_nodes = [int(v) for v in pp_nodes]
        pp1_nodes[-1] = self.feature_dim * self.n_props
        self.pp_layer = FFLayer(pp1_nodes, activation=activation)
        self.scale3_layer = ScaleLayer()
        self.scale5_layer = ScaleLayer()

    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        edge_index = tensors["edge_index"]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        p1, i1 = self.invar_p1_layer(src, dst, tensors["p1"], basis)

        i1_chunks = torch.chunk(i1, self.n_props, dim=-1)
        px_list = [p1]
        new_tensors: dict[str, torch.Tensor] = {"i1": i1}

        if self.rank >= 3:
            p3, i3, dotted_p3 = self.equivar_p3_layer(
                src,
                dst,
                tensors["p3"],
                i1_chunks[1],
                tensors["d3"],
            )
            px_list.append(dotted_p3)
            new_tensors["i3"] = i3
            new_tensors["dotted_p3"] = dotted_p3

        if self.rank >= 5:
            p5, i5, dotted_p5 = self.equivar_p5_layer(
                src,
                dst,
                tensors["p5"],
                i1_chunks[2],
                tensors["d5"],
            )
            px_list.append(dotted_p5)
            new_tensors["i5"] = i5
            new_tensors["dotted_p5"] = dotted_p5

        p1t1 = self.pp_layer(torch.cat(px_list, dim=-1))
        pxt1 = torch.chunk(p1t1, self.n_props, dim=-1)
        new_tensors["p1"] = pxt1[0]

        if self.rank >= 3:
            new_tensors["p3"] = self.scale3_layer(p3, pxt1[1])
        if self.rank >= 5:
            new_tensors["p5"] = self.scale5_layer(p5, pxt1[2])

        return new_tensors
