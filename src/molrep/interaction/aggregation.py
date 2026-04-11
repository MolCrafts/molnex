"""Message aggregation for equivariant message passing.

This module implements the aggregation operation from MACE, which sums messages
from neighboring atoms and applies an equivariant linear transformation.
"""

from __future__ import annotations

import torch
from torch import nn
from pydantic import BaseModel, Field

from molrep.interaction.linear import EquivariantLinear


class MessageAggregationSpec(BaseModel):
    """Specification for message aggregation layer.
    
    This layer implements the neighbor sum + linear operation from MACE:
        A^(s)_{i,klms} = Σ_{k≠i} W^(s)_{kk'm} Σ_{j∈N(i)} φ^(a)_{ij,k'm₁m₂}
    
    The operation consists of:
        1. Scatter sum: Aggregate messages from neighbors to target nodes
        2. Linear: Apply equivariant linear transformation to aggregated features
        3. Optional cutoff weighting: Apply smooth cutoff to messages before aggregation
    
    Attributes:
        irreps: Irreducible representations string (e.g., "64x0e + 32x1o + 16x2e").
        apply_cutoff: Whether to multiply messages by cutoff values before aggregation.
    """

    irreps: str
    apply_cutoff: bool = True


class MessageAggregation(nn.Module):
    """Message aggregation layer with neighbor sum and linear transformation.
    
    This layer performs the core aggregation operation in MACE:
        1. Optionally weight messages by cutoff: m'_ij = m_ij * f_cut(r_ij)
        2. Sum messages to target nodes: h_i = Σ_{j∈N(i)} m'_ij
        3. Apply equivariant linear: A_i = W h_i
    
    Attributes:
        config: MessageAggregationSpec configuration.
        linear: Equivariant linear transformation applied after aggregation.
    """

    def __init__(
        self,
        *,
        irreps: str,
        apply_cutoff: bool = True,
    ):
        """Initialize message aggregation layer.
        
        Args:
            irreps: Irreducible representations string (e.g., "64x0e + 32x1o").
            apply_cutoff: Whether to apply cutoff weighting to messages.
        """
        super().__init__()

        self.config = MessageAggregationSpec(
            irreps=irreps,
            apply_cutoff=apply_cutoff,
        )

        # Equivariant linear transformation after aggregation
        self.linear = EquivariantLinear(
            irreps_in=irreps,
            irreps_out=irreps,
        )

    def forward(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor | None = None,
        n_nodes: int | None = None,
    ) -> torch.Tensor:
        """Aggregate messages from neighbors.
        
        Args:
            messages: Edge messages (n_edges, irreps_dim)
            edge_index: Edge indices ``(E, 2)`` where
                ``edge_index[:, 0]`` = source, ``edge_index[:, 1]`` = target.
            edge_cutoff: Optional cutoff values (n_edges,)
            n_nodes: Optional explicit node count.

        Returns:
            Aggregated features. Output shape: (n_nodes, irreps_dim)
        """
        target_indices = edge_index[:, 1]
        
        # Apply cutoff weighting if enabled
        if self.config.apply_cutoff and edge_cutoff is not None:
            messages = messages * edge_cutoff.unsqueeze(-1)
        
        # Determine number of nodes
        if n_nodes is None:
            n_nodes = int(edge_index.max().item()) + 1
        
        # Scatter-add: sum messages to target nodes
        aggregated = torch.zeros(
            n_nodes, messages.shape[1],
            dtype=messages.dtype,
            device=messages.device,
        )
        aggregated = aggregated.scatter_add(
            0,
            target_indices.unsqueeze(-1).expand_as(messages),
            messages,
        )  # (n_nodes, irreps_dim)
        
        # Apply equivariant linear transformation
        return self.linear(aggregated)
