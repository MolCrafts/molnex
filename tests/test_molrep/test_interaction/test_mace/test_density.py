"""Tests for molrep.interaction.mace.density module."""

import pytest
import torch

from molrep.interaction.mace.density import DensityInteraction, DensityResidualInteraction

N_ELEMENTS = 3
N_NODES = 4
NUM_RADIAL = 5
FEATURES = 8
SH = "1x0e+1x1o"
TARGET = f"{FEATURES}x0e+{FEATURES}x1o"


@pytest.fixture
def graph():
    """A small directed graph plus the edge/node inputs the blocks consume."""
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]).t().contiguous()  # (E, 2)
    n_edges = edge_index.shape[0]
    node_attrs = torch.zeros(N_NODES, N_ELEMENTS, dtype=torch.float64)
    node_attrs[torch.arange(N_NODES), torch.tensor([0, 1, 2, 0])] = 1.0
    return {
        "node_attrs": node_attrs,
        "node_feats": torch.randn(N_NODES, FEATURES, dtype=torch.float64),
        "edge_attrs": torch.randn(n_edges, 4, dtype=torch.float64),
        "edge_feats": torch.randn(n_edges, NUM_RADIAL, dtype=torch.float64),
        "edge_index": edge_index,
    }


def _common():
    return dict(
        node_attrs_irreps=f"{N_ELEMENTS}x0e",
        edge_attrs_irreps=SH,
        edge_feats_irreps=f"{NUM_RADIAL}x0e",
        target_irreps=TARGET,
        radial_mlp=[8],
    )


@pytest.fixture
def first_layer():
    return DensityInteraction(
        node_feats_irreps=f"{FEATURES}x0e", edge_irreps=f"{FEATURES}x0e", **_common()
    ).double()


@pytest.fixture
def residual_layer():
    return DensityResidualInteraction(
        node_feats_irreps=f"{FEATURES}x0e",
        edge_irreps=f"{FEATURES}x0e",
        hidden_irreps=f"{FEATURES}x0e",
        **_common(),
    ).double()


class TestDensityInteraction:
    """Test the first-layer density-normalised interaction."""

    def test_output_shape_is_multiplet_layout(self, first_layer, graph):
        """reshape_irreps emits ``(N, ir_dim, mul)`` for the product block."""
        out, _ = first_layer(**graph)
        assert out.shape == (N_NODES, 4, FEATURES)

    def test_returns_no_skip_connection(self, first_layer, graph):
        """The first layer has no residual to carry — MACE returns ``None``."""
        assert first_layer(**graph)[1] is None

    def test_double_precision_roundtrip(self, graph):
        """`.double()` must work: cuEq bakes its dtype in at construction."""
        layer = DensityInteraction(
            node_feats_irreps=f"{FEATURES}x0e", edge_irreps=f"{FEATURES}x0e", **_common()
        ).double()
        out, _ = layer(**graph)
        assert out.dtype == torch.float64

    def test_skip_tp_weight_survives_dtype_change(self, graph):
        """Rebuilding skip_tp on `.double()` must preserve its trained weight."""
        layer = DensityInteraction(
            node_feats_irreps=f"{FEATURES}x0e", edge_irreps=f"{FEATURES}x0e", **_common()
        )
        before = layer.skip_tp.weight.detach().clone()
        layer = layer.double()
        assert torch.allclose(layer.skip_tp.weight.detach(), before.double())

    def test_density_normalisation_damps_high_coordination(self, first_layer, graph):
        """A node gathering more edges is divided by a larger density."""
        dense = dict(graph)
        # Route every edge at node 2 so its density is the largest in the graph.
        dense["edge_index"] = torch.tensor([[0, 1, 3, 0, 1], [2, 2, 2, 2, 2]]).t()  # (E, 2)
        out, _ = first_layer(**dense)
        assert torch.isfinite(out).all()

    def test_isolated_node_is_finite(self, first_layer, graph):
        """Zero density must not divide by zero — the ``+1`` guarantees it."""
        lonely = dict(graph)
        lonely["edge_index"] = torch.tensor([[0, 1], [1, 0]]).t()  # (E, 2)
        lonely["edge_attrs"] = graph["edge_attrs"][:2]
        lonely["edge_feats"] = graph["edge_feats"][:2]
        out, _ = first_layer(**lonely)
        assert torch.isfinite(out).all()
        # Nodes 2 and 3 receive nothing, so their message is exactly zero.
        assert torch.count_nonzero(out[2]) == 0
        assert torch.count_nonzero(out[3]) == 0


class TestDensityResidualInteraction:
    """Test the residual density-normalised interaction."""

    def test_output_shape_is_multiplet_layout(self, residual_layer, graph):
        """Message keeps the same multiplet layout as the first layer."""
        out, _ = residual_layer(**graph)
        assert out.shape == (N_NODES, 4, FEATURES)

    def test_returns_skip_connection(self, residual_layer, graph):
        """The residual variant hands a skip connection to the product block."""
        _, skip = residual_layer(**graph)
        assert skip.shape == (N_NODES, FEATURES)

    def test_skip_depends_only_on_node_inputs(self, residual_layer, graph):
        """``skip_tp`` reads node features and attrs — not the edges."""
        _, skip_a = residual_layer(**graph)
        rewired = dict(graph)
        rewired["edge_index"] = torch.tensor([[3, 2, 1, 0, 2], [0, 3, 2, 1, 0]]).t()  # (E, 2)
        _, skip_b = residual_layer(**rewired)
        assert torch.allclose(skip_a, skip_b)

    def test_permutation_equivariance(self, residual_layer, graph):
        """Relabelling atoms permutes the output identically."""
        perm = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(perm)
        out, skip = residual_layer(**graph)

        permuted = dict(graph)
        permuted["node_attrs"] = graph["node_attrs"][perm]
        permuted["node_feats"] = graph["node_feats"][perm]
        permuted["edge_index"] = inverse[graph["edge_index"]]
        out_p, skip_p = residual_layer(**permuted)

        assert torch.allclose(out_p, out[perm], atol=1e-12)
        assert torch.allclose(skip_p, skip[perm], atol=1e-12)

    def test_cutoff_scales_messages(self, residual_layer, graph):
        """A zero cutoff kills every message but leaves the skip untouched."""
        n_edges = graph["edge_index"].shape[0]
        zero = torch.zeros(n_edges, 1, dtype=torch.float64)
        out, skip = residual_layer(cutoff=zero, **graph)
        _, skip_ref = residual_layer(**graph)
        assert torch.count_nonzero(out) == 0
        assert torch.allclose(skip, skip_ref)
