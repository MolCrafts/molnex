"""cuEquivariance channel-wise tensor-product convolution.

The path is the official cuEquivariance indexed convolution: ``indices_1``
gathers source nodes and ``indices_out`` scatters directly into target nodes,
so no ``(E, target.dim)`` message is ever materialised.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel

from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum


class _CueqChannelwiseBlockMixin:
    """Radial heads and the fused cuEq convolution shared by the two MACE interactions."""

    conv_tp: cuet.ChannelWiseTensorProduct
    conv_tp_weights: nn.Module
    density_fn: nn.Module

    def _radial_heads(
        self,
        features: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        projected: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor-product weights and edge density from the radial features.

        With node embeddings, the radial MLPs read
        ``cat([edge_feats, source_emb[src], target_emb[tgt]])``. That
        concatenation is never built: ``projected`` carries layer 0 split by
        those three blocks, with the two node halves already multiplied
        per *node*, so only ``(E, hidden)`` is gathered per edge instead of
        ``(E, edge_dim + 2·n_scalar)``.

        Beyond the memory — 1056 channels per edge at Polar-1-M widths — the
        wide form is what made Inductor emit a ``1056 * edge`` index constant
        that leaves int32 above ~2·10⁶ edges and fails to compile.
        """
        if not projected:
            return (
                self.conv_tp_weights(features),
                torch.tanh(self.density_fn(features) ** 2),
            )
        w_conv, src_conv, tgt_conv, b_conv, w_dens, src_dens, tgt_dens, b_dens = projected
        hidden_conv = features @ w_conv + src_conv[source_index] + tgt_conv[target_index]
        hidden_dens = features @ w_dens + src_dens[source_index] + tgt_dens[target_index]
        # The first layer's bias belongs to the sum as a whole, once.
        if b_conv.numel() != 0:
            hidden_conv = hidden_conv + b_conv
        if b_dens.numel() != 0:
            hidden_dens = hidden_dens + b_dens
        return (
            self.conv_tp_weights.forward_from_first(hidden_conv),
            torch.tanh(self.density_fn.forward_from_first(hidden_dens) ** 2),
        )

    def _project_node_features(
        self,
        edge_dim: int,
        source_features: torch.Tensor | None,
        target_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Split each radial MLP's layer 0 and pre-multiply the node halves."""
        if source_features is None or target_features is None:
            return ()
        n_scalar = source_features.shape[-1]
        widths = [edge_dim, n_scalar, n_scalar]
        empty = source_features.new_empty(0)
        w_conv, ws_conv, wt_conv = self.conv_tp_weights.first_layer_parts(widths)
        w_dens, ws_dens, wt_dens = self.density_fn.first_layer_parts(widths)
        b_conv = self.conv_tp_weights.first_layer_bias()
        b_dens = self.density_fn.first_layer_bias()
        return (
            w_conv,
            source_features @ ws_conv,
            target_features @ wt_conv,
            empty if b_conv is None else b_conv,
            w_dens,
            source_features @ ws_dens,
            target_features @ wt_dens,
            empty if b_dens is None else b_dens,
        )

    def _cueq_channelwise_conv(
        self,
        node_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        *,
        source_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the official indexed cuEq convolution and its scalar density.

        cuEq is the sole owner of CG, contraction, scatter and autograd: the
        radial weights are handed to the indexed tensor product together with
        ``indices_1`` / ``indices_out``, so nothing of edge width is built
        outside the kernel.
        """
        if (source_features is None) != (target_features is None):
            raise ValueError("source_features and target_features must be supplied together")

        num_nodes = node_features.shape[0]
        source = edge_index[:, 0]
        target = edge_index[:, 1]

        projected = self._project_node_features(
            edge_features.shape[-1], source_features, target_features
        )

        weights, edge_density = self._radial_heads(edge_features, source, target, projected)
        if cutoff is not None:
            weights = weights * cutoff
            edge_density = edge_density * cutoff
        message = self.conv_tp(
            node_features,
            edge_angular,
            weights,
            indices_1=source,
            indices_out=target,
            size_out=num_nodes,
        )
        return message, _scatter_sum(edge_density, target, num_nodes)


class ConvTPSpec(BaseModel):
    r"""Specification for a channel-wise tensor product convolution."""

    in_irreps: str
    out_irreps: str
    sh_irreps: str


class ConvTP(nn.Module):
    r"""Thin wrapper over the official cuEq indexed convolution."""

    def __init__(
        self,
        *,
        in_irreps: str,
        out_irreps: str,
        sh_irreps: str,
        use_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.config = ConvTPSpec(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            sh_irreps=sh_irreps,
        )
        self.use_fallback = use_fallback

        self.cue_tp = cuet.ChannelWiseTensorProduct(
            cue.Irreps("O3", in_irreps),
            cue.Irreps("O3", sh_irreps),
            cue.Irreps("O3", out_irreps),
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            # Say the backend explicitly. ``use_fallback`` is retained only
            # because cuEq also uses it for its layout-transpose helpers.
            method="naive" if use_fallback else "uniform_1d",
            use_fallback=use_fallback,
        )
        self.weight_numel = self.cue_tp.weight_numel

    def forward(
        self,
        node_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_index: torch.Tensor,
        tp_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate channel-wise TP messages onto target nodes."""
        source = edge_index[:, 0]
        target = edge_index[:, 1]
        return self.cue_tp(
            node_features,
            edge_angular,
            tp_weights,
            indices_1=source,
            indices_out=target,
            size_out=node_features.shape[0],
        )
