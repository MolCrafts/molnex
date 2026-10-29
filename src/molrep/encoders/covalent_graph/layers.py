"""Covalent-graph encoder blocks: chemical input embeddings and the edge-augmented attention layer.

Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md`` §3. Ported from the
reference implementation (``model/graph_block.py``, ``model/gnn.py``) without
torch_geometric: the message passing is an explicit gather / segment-softmax /
``index_add`` over the directed covalent edges, so the layers trace without a
graph break.

References:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
    Hussain, M. S. et al. "Global Self-Attention as a Replacement for Graph
    Convolution" (EGT). KDD 2022. https://arxiv.org/abs/2108.03348
    Reference implementation: ``ByteDance-Seed/byteff2@v1.1.0``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from molrep.chem.equivalence import Equivalence
from molrep.embedding.mlp import DenseMLP, FFLayer

__all__ = ["EdgeAttentionLayer", "ChemicalEmbedding"]


class ChemicalEmbedding(nn.Module):
    """Atom and bond input features (reference ``graph_block.py:FeatureLayer``).

    Five atom embeddings (element ``Z − 1``, degree, formal charge, ring
    connectivity, smallest ring) and two bond embeddings (ring membership,
    bond order) are concatenated, passed through an MLP that ends in GELU, and
    averaged over topological equivalence classes.

    Args:
        atom_embedding_dim: Width of the element embedding.
        channel_dim: Width of every other embedding.
        node_mlp: ``(hidden, out, layers)`` of the atom MLP.
        edge_mlp: ``(hidden, out, layers)`` of the bond MLP.
        dtype: Floating dtype of every parameter.
    """

    #: Rows of the element table: atomic numbers 1..86, indexed as ``Z − 1``.
    NUM_ELEMENTS = 86
    #: Degree and ring connectivity run 0..6 (``MAX_CONNECTIVITY``).
    MAX_CONNECTIVITY = 6
    #: Formal charge is clamped to ±2 (``MAX_FORMAL_CHARGE``).
    MAX_FORMAL_CHARGE = 2
    #: The smallest ring runs 0..8 (``MAX_RING_SIZE``).
    MAX_RING_SIZE = 8
    #: Bond orders single, double, triple, aromatic.
    NUM_BOND_ORDERS = 4

    def __init__(
        self,
        *,
        atom_embedding_dim: int,
        channel_dim: int,
        node_mlp: tuple[int, int, int],
        edge_mlp: tuple[int, int, int],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.element = nn.Embedding(self.NUM_ELEMENTS, atom_embedding_dim, dtype=dtype)
        self.degree = nn.Embedding(self.MAX_CONNECTIVITY + 1, channel_dim, dtype=dtype)
        self.formal_charge = nn.Embedding(2 * self.MAX_FORMAL_CHARGE + 1, channel_dim, dtype=dtype)
        self.ring_connectivity = nn.Embedding(self.MAX_CONNECTIVITY + 1, channel_dim, dtype=dtype)
        self.min_ring_size = nn.Embedding(self.MAX_RING_SIZE + 1, channel_dim, dtype=dtype)
        self.in_ring = nn.Embedding(2, channel_dim, dtype=dtype)
        self.bond_order = nn.Embedding(self.NUM_BOND_ORDERS, channel_dim, dtype=dtype)
        node_hidden, node_out, node_layers = node_mlp
        edge_hidden, edge_out, edge_layers = edge_mlp
        self.node_mlp = FFLayer(
            [node_hidden] * (node_layers - 1) + [node_out],
            in_dim=atom_embedding_dim + 4 * channel_dim,
            activation="gelu",
            dtype=dtype,
        )
        self.edge_mlp = FFLayer(
            [edge_hidden] * (edge_layers - 1) + [edge_out],
            in_dim=2 * channel_dim,
            activation="gelu",
            dtype=dtype,
        )
        self.node_dim = node_out
        self.edge_dim = edge_out
        self.equivalence = Equivalence()

    def atoms(
        self,
        Z: torch.Tensor,
        degree: torch.Tensor,
        formal_charge: torch.Tensor,
        ring_connectivity: torch.Tensor,
        min_ring_size: torch.Tensor,
        equivalence: torch.Tensor,
    ) -> torch.Tensor:
        """Atom features ``(N, node_dim)``; every argument is ``(N,)`` int64."""
        charge = (formal_charge + self.MAX_FORMAL_CHARGE).clamp(0, 2 * self.MAX_FORMAL_CHARGE)
        raw = torch.cat(
            (
                self.element(Z - 1),
                self.degree(degree),
                self.formal_charge(charge),
                self.ring_connectivity(ring_connectivity),
                self.min_ring_size(min_ring_size),
            ),
            dim=-1,
        )
        return self.equivalence(self.node_mlp(raw), equivalence)

    def bonds(
        self, in_ring: torch.Tensor, order: torch.Tensor, equivalence: torch.Tensor
    ) -> torch.Tensor:
        """Bond features ``(N_bonds, edge_dim)``; ``order`` is 0..3, every argument int64."""
        raw = torch.cat((self.in_ring(in_ring), self.bond_order(order)), dim=-1)
        return self.equivalence(self.edge_mlp(raw), equivalence)


class EdgeAttentionLayer(nn.Module):
    """One local edge-augmented graph-transformer layer (``gnn.py:EGTConv``).

    Node stream: pre-LN multi-head attention over the incoming covalent edges,
    the logit biased by the edge stream and the weight gated by it, then a
    pre-LN feed-forward residual. Edge stream: the raw logits are projected back
    onto the edge, then a pre-LN feed-forward residual.

    Args:
        width: Node and edge width (input = output).
        heads: Attention heads.
        head_dim: Width of one head.
        ffn: ``(hidden, layers)`` of both feed-forward MLPs.
        dtype: Floating dtype of every parameter.
    """

    def __init__(
        self, width: int, *, heads: int, head_dim: int, ffn: tuple[int, int], dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.norm1 = nn.LayerNorm(width, dtype=dtype)
        self.query = nn.Linear(width, inner, dtype=dtype)
        self.key = nn.Linear(width, inner, dtype=dtype)
        self.value = nn.Linear(width, inner, dtype=dtype)
        self.proj = nn.Linear(inner, width, dtype=dtype)
        self.norm2 = nn.LayerNorm(width, dtype=dtype)
        self.ffn = DenseMLP(width, ffn[0], width, ffn[1], dtype=dtype)
        self.edge_norm1 = nn.LayerNorm(width, dtype=dtype)
        self.gate = nn.Linear(width, heads, dtype=dtype)
        self.edge_bias = nn.Linear(width, heads, dtype=dtype)
        self.edge_proj = nn.Linear(heads, width, dtype=dtype)
        self.edge_norm2 = nn.LayerNorm(width, dtype=dtype)
        self.edge_ffn = DenseMLP(width, ffn[0], width, ffn[1], dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node features ``(N, W)`` and directed edge features ``(E, W)``.

        Args:
            x: Node features ``(N, W)``.
            e: Directed edge features ``(E, W)``.
            source: Sender of each directed edge ``(E,)``.
            target: Receiver of each directed edge ``(E,)``; the softmax runs
                over the edges entering one atom.

        Returns:
            ``(x, e)`` with the same shapes.
        """
        n_atoms = x.shape[0]
        h = self.norm1(x)
        q = self.query(h).view(n_atoms, self.heads, self.head_dim)
        k = self.key(h).view(n_atoms, self.heads, self.head_dim)
        v = self.value(h).view(n_atoms, self.heads, self.head_dim)
        g = self.edge_norm1(e)
        gate = torch.sigmoid(self.gate(g))
        logits = (q[target] * k[source]).sum(dim=-1) / math.sqrt(self.head_dim) + self.edge_bias(g)
        # Segment softmax over the edges entering each atom (PyG ``softmax``).
        peak = logits.new_zeros(n_atoms, self.heads).scatter_reduce(
            0, target[:, None].expand_as(logits), logits.detach(), "amax", include_self=False
        )
        weight = torch.exp(logits - peak[target])
        total = weight.new_zeros(n_atoms, self.heads).index_add(0, target, weight)
        weight = weight / (total[target] + 1e-16)
        message = v[source] * (gate * weight).unsqueeze(-1)
        aggregate = v.new_zeros(n_atoms, self.heads, self.head_dim).index_add(0, target, message)
        x = self.proj(aggregate.reshape(n_atoms, -1)) + x
        x = self.ffn(self.norm2(x)) + x
        e = e + self.edge_proj(logits)
        e = self.edge_ffn(self.edge_norm2(e)) + e
        return x, e
