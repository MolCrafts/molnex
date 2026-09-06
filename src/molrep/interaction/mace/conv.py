"""cuEquivariance channel-wise tensor-product convolution.

The default path is the official cuEquivariance indexed convolution:
``indices_1`` gathers source nodes and ``indices_out`` scatters directly into
target nodes. Large Polar systems can opt into a receiver-blocked layout.
Each fixed-size edge block then targets only a fixed-size node block, so the
official ``uniform_1d`` kernel keeps both its output workspace and the radial
weight tensor bounded without reimplementing the tensor product.

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
from torch.utils.checkpoint import checkpoint

from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum


def _checked_target_block_size(block_size: int | None) -> int | None:
    if block_size is None:
        return None
    value = int(block_size)
    if value != block_size or value < 1:
        raise ValueError(f"target block size must be an integer >= 1, got {block_size!r}")
    return value


def _block_shape(
    num_nodes: int,
    num_edges: int,
    max_block_size: int,
) -> tuple[int, int, int]:
    """Return ``(num_blocks, block_span, edges_per_block)`` for packed edges.

    The values depend only on tensor shapes, never tensor contents. Dynamo can
    therefore specialise the small Python loop without a graph break while an
    MD neighbour rebuild changes the indices in place.

    ``max_block_size`` is an upper bound.  Once the number of blocks is fixed,
    their receiver intervals are spread as evenly as integer division allows.
    This avoids paying the busiest-block edge capacity for a tiny final block;
    for example ``N=8850, max=4096`` becomes three 2950-node blocks instead of
    ``4096 + 4096 + 658``.
    """
    if num_nodes < 1:
        raise ValueError("receiver-blocked convolution needs at least one node")
    num_blocks = (num_nodes + max_block_size - 1) // max_block_size
    block_span = (num_nodes + num_blocks - 1) // num_blocks
    if num_edges % num_blocks:
        raise ValueError(
            f"receiver-blocked edge count {num_edges} is not divisible by "
            f"the {num_blocks} target blocks"
        )
    return num_blocks, block_span, num_edges // num_blocks


class _CueqChannelwiseBlockMixin:
    """Compile-safe checkpoint body shared by the two MACE interactions.

    A bound module method is deliberate. Dynamo can lift its registered
    parameters into the checkpoint higher-order op, whereas a nested closure
    over the same modules leaves parameters as free variables and fails
    fullgraph tracing.
    """

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

    def _cueq_block_forward(
        self,
        nodes: torch.Tensor,
        angular: torch.Tensor,
        features: torch.Tensor,
        source_index: torch.Tensor,
        local_target: torch.Tensor,
        global_target: torch.Tensor,
        envelope: torch.Tensor,
        output_shape: torch.Tensor,
        *projected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate radial weights and run one official indexed cuEq TP."""
        weights, edge_density = self._radial_heads(features, source_index, global_target, projected)
        if envelope.numel() != 0:
            weights = weights * envelope
            edge_density = edge_density * envelope
        size_out = output_shape.shape[0]
        message = self.conv_tp(
            nodes,
            angular,
            weights,
            indices_1=source_index,
            indices_out=local_target,
            size_out=size_out,
        )
        density = _scatter_sum(edge_density, local_target, size_out)
        return message, density

    def _cueq_channelwise_conv(
        self,
        node_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        *,
        target_block_size: int | None,
        source_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate an official cuEq convolution and its scalar density.

        With ``target_block_size=None`` this is the ordinary full indexed
        convolution. Otherwise the edge tensors must use fixed receiver-block
        layout. ``target_block_size`` is the maximum receiver count; after
        choosing ``ceil(N / target_block_size)`` blocks, their common span is
        reduced to ``ceil(N / num_blocks)``. Block ``b`` may target only its
        resulting receiver interval, and every block owns the same number of
        edge slots.

        The block path checkpoints ``radial MLP + cuEq TP`` as one unit. Its
        backward recomputes one block's weights instead of retaining all
        ``(E, weight_numel)`` tensors. cuEq remains the sole owner of CG,
        contraction, scatter and autograd.
        """
        if (source_features is None) != (target_features is None):
            raise ValueError("source_features and target_features must be supplied together")

        num_nodes = node_features.shape[0]
        source = edge_index[:, 0]
        target = edge_index[:, 1]

        projected = self._project_node_features(
            edge_features.shape[-1], source_features, target_features
        )

        if target_block_size is None:
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

        max_block_size = _checked_target_block_size(target_block_size)
        assert max_block_size is not None
        num_blocks, block_span, edges_per_block = _block_shape(
            num_nodes, edge_index.shape[0], max_block_size
        )

        empty = edge_features.new_empty(0, 0)
        output_shape = edge_features.new_empty(block_span, 0)
        messages: list[torch.Tensor] = []
        densities: list[torch.Tensor] = []
        for block in range(num_blocks):
            edge_start = block * edges_per_block
            edge_stop = edge_start + edges_per_block
            node_start = block * block_span
            edge_slice = slice(edge_start, edge_stop)
            block_source = source[edge_slice]
            block_target = target[edge_slice]
            block_cutoff = empty if cutoff is None else cutoff[edge_slice]
            block_message, block_density = checkpoint(
                self._cueq_block_forward,
                node_features,
                edge_angular[edge_slice],
                edge_features[edge_slice],
                block_source,
                block_target - node_start,
                block_target,
                block_cutoff,
                output_shape,
                *projected,
                use_reentrant=False,
                # No stochastic radial layers: RNG save/restore is wasted.
                preserve_rng_state=False,
            )
            messages.append(block_message)
            densities.append(block_density)

        return (
            torch.cat(messages, dim=0)[:num_nodes],
            torch.cat(densities, dim=0)[:num_nodes],
        )


class ConvTPSpec(BaseModel):
    r"""Specification for a channel-wise tensor product convolution."""

    in_irreps: str
    out_irreps: str
    sh_irreps: str


class ConvTP(nn.Module):
    r"""Thin wrapper over the official cuEq indexed convolution.

    :meth:`configure_target_blocks` changes only execution layout and therefore
    creates no parameters or persistent buffers. The supplied edge tensors
    must already have the matching fixed-block layout.
    """

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
        self.target_block_size: int | None = None

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

    def configure_target_blocks(self, block_size: int | None) -> None:
        """Select full convolution (``None``) or a maximum receiver-block size."""
        self.target_block_size = _checked_target_block_size(block_size)

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
        if self.target_block_size is None:
            return self.cue_tp(
                node_features,
                edge_angular,
                tp_weights,
                indices_1=source,
                indices_out=target,
                size_out=node_features.shape[0],
            )

        max_block_size = self.target_block_size
        num_nodes = node_features.shape[0]
        num_blocks, block_span, edges_per_block = _block_shape(
            num_nodes, edge_index.shape[0], max_block_size
        )
        messages: list[torch.Tensor] = []
        for block in range(num_blocks):
            edge_start = block * edges_per_block
            edge_stop = edge_start + edges_per_block
            node_start = block * block_span
            messages.append(
                self.cue_tp(
                    node_features,
                    edge_angular[edge_start:edge_stop],
                    tp_weights[edge_start:edge_stop],
                    indices_1=source[edge_start:edge_stop],
                    indices_out=target[edge_start:edge_stop] - node_start,
                    size_out=block_span,
                )
            )
        return torch.cat(messages, dim=0)[:num_nodes]
