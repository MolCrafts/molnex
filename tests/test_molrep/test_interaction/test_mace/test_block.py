"""Tests for molrep.interaction.mace.block module."""

import pytest
import torch

from molrep.interaction.mace.block import InteractionBlock, InteractionSpec

N_NODES = 4
FEATURES = 8
NUM_BESSEL = 5
L_MAX = 1
SH_DIM = (L_MAX + 1) ** 2  # 1x0e + 1x1o
MESSAGE_DIM = FEATURES * SH_DIM  # mixed-l message irreps: 8x0e + 8x1o


@pytest.fixture
def graph():
    """A small fp64 directed graph, ``edge_index`` in ``(E, 2)`` layout."""
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]).t().contiguous()  # (E, 2)
    n_edges = edge_index.shape[0]
    return {
        "node_feats": torch.randn(N_NODES, FEATURES, dtype=torch.float64),
        "edge_attrs": torch.randn(n_edges, SH_DIM, dtype=torch.float64),
        "edge_feats": torch.randn(n_edges, NUM_BESSEL, dtype=torch.float64),
        "edge_index": edge_index,
    }


def _block(avg_num_neighbors: float) -> InteractionBlock:
    """Build an fp64 block; the seed makes two calls weight-identical."""
    torch.manual_seed(0)
    return InteractionBlock(
        num_features=FEATURES,
        num_bessel=NUM_BESSEL,
        l_max=L_MAX,
        avg_num_neighbors=avg_num_neighbors,
    ).double()


class TestInteractionSpec:
    """Test InteractionSpec configuration."""

    def test_stores_avg_num_neighbors(self):
        """The normalisation constant round-trips through the pydantic spec."""
        spec = InteractionSpec(
            num_features=FEATURES,
            num_bessel=NUM_BESSEL,
            l_max=L_MAX,
            avg_num_neighbors=2.0,
        )
        assert spec.avg_num_neighbors == 2.0


class TestInteractionBlock:
    """Test the promoted MACE interaction block."""

    def test_forward_output_shape(self, graph):
        """The message carries the mixed-l irreps ``(N, num_features * (l_max+1)**2)``."""
        node_feats, _ = _block(1.0)(**graph)
        assert node_feats.shape == (N_NODES, MESSAGE_DIM)

    def test_skip_connection_is_the_input_tensor(self, graph):
        """``sc`` is the untouched input object, not a copy — the product block needs it."""
        _, sc = _block(1.0)(**graph)
        assert sc is graph["node_feats"]

    def test_avg_num_neighbors_scales_message(self, graph):
        """Doubling ``avg_num_neighbors`` halves the message (post-conv linear is bias-free)."""
        one, _ = _block(1.0)(**graph)
        two, _ = _block(2.0)(**graph)
        assert torch.allclose(two * 2.0, one, rtol=0.0, atol=1e-12)

    def test_double_precision_roundtrip(self, graph):
        """`.double()` must work: cuEq bakes its dtype in at construction."""
        node_feats, _ = _block(1.0)(**graph)
        assert node_feats.dtype == torch.float64

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """Sub-layer names must stay put — official MACE weights load by key."""
        assert sorted(_block(1.0).state_dict().keys()) == [
            "conv_tp.cue_tp.f.m.graphs.0.graph.c0",
            "conv_tp.cue_tp.f.m.graphs.0.graph.c1",
            "linear.f.m.graphs.0.graph.c0",
            "linear.f.m.graphs.0.graph.c1",
            "linear.weight",
            "node_linear.f.m.graphs.0.graph.c0",
            "node_linear.weight",
            "radial_mlp.mlp.0.bias",
            "radial_mlp.mlp.0.weight",
            "radial_mlp.mlp.2.bias",
            "radial_mlp.mlp.2.weight",
            "radial_mlp.mlp.4.bias",
            "radial_mlp.mlp.4.weight",
        ]
