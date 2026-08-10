"""MACE interaction block — one equivariant message-passing layer.

:class:`InteractionBlock` wires the pre-convolution equivariant linear, the
radial weight MLP, :class:`~molrep.interaction.mace.conv.ConvTP` and the
post-convolution linear into the ``avg_num_neighbors``-normalised layer used by
the plain MACE encoder. The density-normalised variants (MACE-MP / MatPES) live
in :mod:`molrep.interaction.mace.density`.

Relocated verbatim from ``molzoo/mace.py``, which continues to re-export both
names for backwards compatibility.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from molix import config
from molrep.interaction.mace.conv import ConvTP
from molrep.interaction.product import irreps_from_l_max, sh_irreps_from_l_max
from molrep.interaction.radial import RadialWeightMLP


class InteractionSpec(BaseModel):
    """Configuration for a single interaction block.

    Attributes:
        num_features: Scalar channel multiplicity.
        num_bessel: Number of Bessel radial basis functions.
        l_max: Maximum angular momentum order.
        avg_num_neighbors: Average number of neighbors for normalization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_features: int = Field(..., gt=0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    avg_num_neighbors: float = Field(1.0, gt=0.0)
    use_fallback: bool = True


class InteractionBlock(nn.Module):
    """Equivariant message passing with tensor product convolution.

    Performs geometric message passing via cuEquivariance-accelerated tensor products,
    returning updated node features and skip connection for residual updates.

    Architecture:
        node_feats → node_linear → tensor_product(edge_attrs, tp_weights)
        → aggregate → linear → (node_feats_out, skip_connection)

    Attributes:
        conv_tp: Tensor product convolution (cuEquivariance ChannelWiseTensorProduct).
        node_linear: Pre-convolution equivariant linear transformation.
        radial_mlp: MLP generating tensor product weights from edge features.
        linear: Post-convolution equivariant linear projection.
        avg_num_neighbors: Message normalization constant.

    Reference:
        https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_bessel: int = 8,
        l_max: int = 2,
        avg_num_neighbors: float = 1.0,
        use_fallback: bool = True,
    ):
        """Initialize interaction block.

        Args:
            num_features: Scalar channel multiplicity.
            num_bessel: Number of Bessel basis functions.
            l_max: Maximum angular momentum order.
            avg_num_neighbors: Average neighbor count for message normalization.
            use_fallback: Pure-torch cuEq path for the tensor product (default
                ``True``, functorch-safe); ``False`` selects the fused kernels
                for autograd-backed force paths.
        """
        super().__init__()

        self.config = InteractionSpec(
            num_features=num_features,
            num_bessel=num_bessel,
            l_max=l_max,
            avg_num_neighbors=avg_num_neighbors,
            use_fallback=use_fallback,
        )

        # Node *state* is pure scalar (l=0); the mixed-l message irreps live only
        # transiently in the tensor-product output, where they are contracted
        # back to invariant scalars by the downstream ProductHead. Keeping the
        # node state scalar makes every node-state operation (node_linear,
        # ElementUpdate, projections) equivariant by construction — fabricating
        # l>0 node components from scalars via a plain/dense linear is exactly
        # what breaks rotation invariance.
        node_irreps_str = f"{num_features}x0e"
        irreps_str = irreps_from_l_max(l_max, num_features)  # mixed-l message irreps
        sh_irreps_str = sh_irreps_from_l_max(l_max)

        # 1. Tensor product convolution (define first to get weight_numel):
        #    scalar node features ⊗ Y_l(r̂) -> mixed-l equivariant messages.
        self.conv_tp = ConvTP(
            in_irreps=node_irreps_str,
            out_irreps=irreps_str,
            sh_irreps=sh_irreps_str,
            use_fallback=use_fallback,
        )

        # Actual TP output irreps (may differ from requested out_irreps)
        tp_out_irreps = str(self.conv_tp.cue_tp.irreps_out)

        # 2. Pre-convolution equivariant linear (scalar -> scalar)
        self.node_linear = cuet.Linear(
            irreps_in=cue.Irreps("O3", node_irreps_str),
            irreps_out=cue.Irreps("O3", node_irreps_str),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        # 3. Radial MLP for TP weights
        self.radial_mlp = RadialWeightMLP(
            in_dim=num_bessel,
            hidden_dim=num_features,
            out_dim=self.conv_tp.weight_numel,
            num_layers=2,
        )

        # 4. Post-convolution equivariant linear
        self.linear = cuet.Linear(
            irreps_in=cue.Irreps("O3", tp_out_irreps),
            irreps_out=cue.Irreps("O3", irreps_str),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        self.avg_num_neighbors = avg_num_neighbors

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one interaction layer.

        Args:
            node_feats: Node features ``(n_nodes, irreps_dim)``.
            edge_attrs: Spherical harmonics ``(n_edges, sh_dim)``.
            edge_feats: Radial basis features ``(n_edges, num_bessel)``.
            edge_index: Edge indices ``(n_edges, 2)``.

        Returns:
            tuple of:
                - ``node_feats``: Updated node features ``(n_nodes, irreps_dim)``.
                - ``sc``: Skip connection (original input) ``(n_nodes, irreps_dim)``.
        """
        sc = node_feats  # skip connection for EquivariantProductBasisBlock

        # Pre-convolution linear
        node_feats_up = self.node_linear(node_feats)

        # TP weights from radial basis
        tp_weights = self.radial_mlp(edge_feats)

        # Tensor product convolution with neighbor aggregation
        messages = self.conv_tp(
            node_features=node_feats_up,
            edge_angular=edge_attrs,
            edge_index=edge_index,
            tp_weights=tp_weights,
        )

        # Normalize by average number of neighbors
        messages = messages / self.avg_num_neighbors

        # Post-convolution linear
        node_feats = self.linear(messages)

        return node_feats, sc
