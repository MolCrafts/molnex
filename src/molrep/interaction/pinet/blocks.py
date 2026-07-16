"""Composite PiNet blocks: invariant / equivariant updates, OutLayer, GCBlock."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from molix import config

from .ff import FFLayer
from .message import DotLayer, IPLayer, PILayer, PIXLayer, ScaleLayer


class InvarLayer(nn.Module):
    """Scalar invariant update block: ``PI -> II -> IP -> PP``."""

    def __init__(
        self,
        *,
        p1_in_dim: int,
        pp_nodes: Sequence[int],
        pi_nodes: Sequence[int],
        ii_nodes: Sequence[int],
        n_basis: int,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        if not pi_nodes:
            raise ValueError("pi_nodes must not be empty.")
        pi_in = 2 * int(p1_in_dim)
        self.pi_layer = PILayer(pi_nodes, in_dim=pi_in, n_basis=n_basis, activation=activation)
        self.ii_layer = FFLayer(
            ii_nodes, in_dim=int(pi_nodes[-1]), activation=activation, use_bias=False
        )
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(
            pp_nodes, in_dim=int(ii_nodes[-1]), activation=activation, use_bias=False
        )

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
        in_channels: int,
        out_channels: int,
        weighted: bool = False,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        del activation  # unused — historical from reference impl
        # PIX acts on the incoming px channel width (1 on the first block,
        # feature_dim thereafter). After ``ix = (... + d) * i1``, the channel
        # axis becomes the scalar interaction width (== out_channels under the
        # GCBlock contract pp_nodes[-1] == ii_nodes[-1]), so the PP projection
        # is always in_dim=out_channels — matching historical LazyLinear
        # materialisation on the first forward.
        self.pi_layer = PIXLayer(channels=int(in_channels), weighted=weighted)
        self.ip_layer = IPLayer()
        self.pp_layer = FFLayer(
            [out_channels], in_dim=int(out_channels), activation=None, use_bias=False
        )
        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer(channels=int(out_channels), weighted=weighted)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        px: torch.Tensor,
        i1: torch.Tensor,
        diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # One broadcast multiply instead of two ScaleLayer + add (bit-identical).
        ix = (self.pi_layer(src, dst, px) + diff.unsqueeze(-1)) * i1.unsqueeze(-2)
        px_new = self.ip_layer(src, px, ix)
        px_new = self.pp_layer(px_new)
        dotted_px = self.dot_layer(px_new)
        return px_new, ix, dotted_px


class OutLayer(nn.Module):
    """Per-block output head with residual accumulation (PiNN ``OutLayer``).

    Mirrors ``pinn.networks.pinet2.OutLayer``::

        output_i = Dense_biasless(FFLayer(p1_i)) + output_{i-1}

    The absolute energy zero-point is owned by the atomic dress, so the final
    projection carries no bias.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        out_units: int = 1,
        activation: str | type[nn.Module] | None = "tanh",
    ) -> None:
        super().__init__()
        if not n_nodes:
            raise ValueError("OutLayer requires at least one FF width.")
        self.ff_layer = FFLayer(n_nodes, in_dim=int(in_dim), activation=activation, use_bias=True)
        self.out_units = nn.Linear(int(n_nodes[-1]), int(out_units), bias=False, dtype=config.ftype)

    def forward(self, px: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        return self.out_units(self.ff_layer(px)) + prev_output


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
        p1_in_dim: int,
        p3_in_dim: int = 1,
        p5_in_dim: int = 1,
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
            p1_in_dim=int(p1_in_dim),
            pp_nodes=pp_nodes,
            pi_nodes=pi_nodes,
            ii_nodes=ii1_nodes,
            n_basis=n_basis,
            activation=activation,
        )

        if self.rank >= 3:
            self.equivar_p3_layer = EquivarLayer(
                in_channels=int(p3_in_dim),
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                activation=activation,
            )
        if self.rank >= 5:
            self.equivar_p5_layer = EquivarLayer(
                in_channels=int(p5_in_dim),
                out_channels=int(pp_nodes[-1]),
                weighted=weighted,
                activation=activation,
            )
        pp1_nodes = [int(v) for v in pp_nodes]
        pp1_nodes[-1] = self.feature_dim * self.n_props
        # cat([p1, dotted_p3, ...]) last-dim = n_props * pp_nodes[-1]
        pp_in = int(pp_nodes[-1]) * self.n_props
        self.pp_layer = FFLayer(pp1_nodes, in_dim=pp_in, activation=activation)
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
