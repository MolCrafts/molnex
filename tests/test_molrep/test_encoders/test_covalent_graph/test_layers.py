"""Unit tests for molrep.encoders.covalent_graph.layers."""

from __future__ import annotations

import torch

from molrep.encoders.covalent_graph.layers import EdgeAttentionLayer
from tests.test_molrep.test_encoders.test_covalent_graph.conftest import DTYPE


class TestEdgeAttentionLayer:
    def test_nodes_and_edges_follow_a_relabelling_of_the_atoms(self):
        torch.manual_seed(0)
        layer = EdgeAttentionLayer(8, heads=2, head_dim=4, ffn=(8, 2), dtype=DTYPE)
        x = torch.randn(4, 8, dtype=DTYPE)
        source, target = torch.tensor([0, 1, 1, 2, 2, 3]), torch.tensor([1, 0, 2, 1, 3, 2])
        e = torch.randn(6, 8, dtype=DTYPE)
        perm = torch.tensor([2, 0, 3, 1])  # new atom k is old atom perm[k]
        inverse = torch.argsort(perm)
        x1, e1 = layer(x, e, source, target)
        x2, e2 = layer(x[perm], e, inverse[source], inverse[target])
        torch.testing.assert_close(x2, x1[perm])
        torch.testing.assert_close(e2, e1)
