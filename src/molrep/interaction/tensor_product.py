"""Tensor product convolution for equivariant message passing.

Wraps `ChannelWiseTensorProduct` and expects precomputed edge weights.
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
    ):
        """Initialize channelwise tensor product layer.

        Args:
            in_irreps: Input irreps for node features.
            out_irreps: Output irreps for messages.
            sh_irreps: Irreps for spherical harmonics.
        """
        super().__init__()

        self.config = ConvTPSpec(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            sh_irreps=sh_irreps,
        )

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


def irreps_from_l_max(l_max: int, hidden_dim: int) -> str:
    """Generate irreps string with uniform multiplicities for optimal cuEquivariance performance.

    Args:
        l_max: Maximum angular momentum.
        hidden_dim: Uniform multiplicity across all l values.

    Returns:
        Irreps string (e.g., "32x0e + 32x1o + 32x2e").

    Note:
        Uniform multiplicities enable cuEquivariance's fast uniform_1d method (~5-10x speedup).
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"{hidden_dim}x{l}{parity}")
    return " + ".join(irreps_list)


def sh_irreps_from_l_max(l_max: int) -> str:
    """Generate spherical harmonics irreps string from l_max.

    Args:
        l_max: Maximum angular momentum.

    Returns:
        Irreps string (e.g., "1x0e + 1x1o + 1x2e").
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"1x{l}{parity}")
    return " + ".join(irreps_list)
