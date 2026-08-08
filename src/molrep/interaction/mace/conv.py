"""Channel-wise tensor product convolution for MACE-style message passing.

:class:`ConvTP` is a thin wrapper around ``cuet.ChannelWiseTensorProduct``
(subscripts ``"uv,iu,jv,kuv+ijk"``) with gather/scatter folded in via
``indices_1``/``indices_out``/``size_out``, mirroring the signature of
``cuet.ChannelWiseTensorProduct``.

Relocated verbatim from :mod:`molrep.interaction.product`, which keeps the
generic :class:`~molrep.interaction.product.EquivariantPolynomialTP` wrapper
and the ``irreps_from_l_max`` / ``sh_irreps_from_l_max`` helpers (shared with
non-MACE encoders) and re-exports the two names below for backwards
compatibility.

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


class ConvTPSpec(BaseModel):
    r"""Specification for tensor product convolution layer.

    One-particle basis:
    $\phi_{ij} = \sum_{l_1,l_2,m_1,m_2} c_{l_3 m_3}^{l_1 m_1, l_2 m_2}
    R(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) h_j^{l_2 m_2}$

    Attributes:
        in_irreps: Input irreps.
        out_irreps: Output irreps.
        sh_irreps: Spherical harmonics irreps.
    """

    in_irreps: str
    out_irreps: str
    sh_irreps: str


class ConvTP(nn.Module):
    r"""Channelwise tensor product for equivariant message passing.

    Computes messages via tensor product:
    $$\phi_{ij} = \sum_{l_1,l_2,m_1,m_2} c_{l_3 m_3}^{l_1 m_1, l_2 m_2}
    R(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) h_j^{l_2 m_2}$$

    Attributes:
        config: ConvTPSpec configuration.
        cue_tp: ChannelWiseTensorProduct layer.
        weight_numel: Number of elements in TP weights.
    """

    def __init__(
        self,
        *,
        in_irreps: str,
        out_irreps: str,
        sh_irreps: str,
        use_fallback: bool = True,
    ):
        """Initialize channelwise tensor product layer.

        Args:
            in_irreps: Input irreps for node features.
            out_irreps: Output irreps for messages.
            sh_irreps: Irreps for spherical harmonics.
            use_fallback: If ``True`` (default), pure-torch cuEq path so
                ``ForceDerivation(method="functorch")`` can trace. Set
                ``False`` for fused kernels when forces use
                ``method="autograd"`` (e.g. MACE-OMOL).
        """
        super().__init__()

        self.config = ConvTPSpec(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            sh_irreps=sh_irreps,
        )
        self.use_fallback = use_fallback

        irreps_in = cue.Irreps("O3", in_irreps)
        irreps_sh = cue.Irreps("O3", sh_irreps)
        irreps_out = cue.Irreps("O3", out_irreps)

        self.cue_tp = cuet.ChannelWiseTensorProduct(  # type: ignore
            irreps_in,
            irreps_sh,
            irreps_out,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
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
        """Compute tensor product messages with integrated gather/scatter.

        Args:
            node_features: Node features.
            edge_angular: Spherical harmonics.
            edge_index: Edge indices ``(E, 2)``.
            tp_weights: TP weights.

        Returns:
            Computed messages (n_edges, out_irreps_dim).
        """
        indices_1 = edge_index[:, 0]
        indices_out = edge_index[:, 1]

        messages = self.cue_tp(
            node_features,
            edge_angular,
            tp_weights,
            indices_1=indices_1,
            indices_out=indices_out,
            size_out=node_features.shape[0],
        )

        return messages
