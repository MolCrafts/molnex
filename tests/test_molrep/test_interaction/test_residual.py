"""Tests for molrep.interaction.residual module."""

import cuequivariance as cue
import pytest
import torch

from molix.F.scatter import scatter_sum
from molrep.interaction.residual import ResidualInteraction, _partition_gate

N_ELEMENTS = 3
N_NODES = 4
NUM_RADIAL = 5
FEATURES = 8
SH = "1x0e+1x1o"
TARGET = f"{FEATURES}x0e+{FEATURES}x1o"
HIDDEN = f"{FEATURES}x0e"


@pytest.fixture
def graph():
    """A small directed graph plus the edge/node inputs the block consumes."""
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


@pytest.fixture
def layer():
    return ResidualInteraction(
        node_attrs_irreps=f"{N_ELEMENTS}x0e",
        node_feats_irreps=f"{FEATURES}x0e",
        edge_attrs_irreps=SH,
        edge_feats_irreps=f"{NUM_RADIAL}x0e",
        edge_irreps=f"{FEATURES}x0e",
        target_irreps=TARGET,
        hidden_irreps=HIDDEN,
        radial_mlp=[8],
    ).double()


def _explicit_message(layer: ResidualInteraction, graph: dict) -> torch.Tensor:
    """The pre-fusion formulation: gather ``x[source]`` → per-edge TP → scatter on ``target``.

    Spelled out with the block's own sub-layers so the only thing the block
    can differ in is how the gather/scatter reaches the kernel.
    """
    source, target = graph["edge_index"][:, 0], graph["edge_index"][:, 1]
    n = graph["node_feats"].shape[0]
    up = layer.linear_up(graph["node_feats"])
    edge_feats = torch.cat(
        [
            graph["edge_feats"],
            layer.source_embedding(graph["node_attrs"])[source],
            layer.target_embedding(graph["node_attrs"])[target],
        ],
        dim=-1,
    )
    tp_weights = layer.conv_tp_weights(edge_feats)
    density = scatter_sum(torch.tanh(layer.density_fn(edge_feats) ** 2), target, dim=0, dim_size=n)
    mji = layer.conv_tp(up[source], graph["edge_attrs"], tp_weights)
    message = scatter_sum(mji, target, dim=0, dim_size=n)
    message = layer.linear_1(message) / (density * layer.beta + layer.alpha)
    message = message + layer.linear_res(up)
    message = layer.linear_2(layer.equivariant_nonlin(message))
    return layer._reshape(message)


class TestPartitionGate:
    """Test the scalar / gate / gated split feeding the gated nonlinearity."""

    def test_splits_scalars_gates_gated(self):
        """``8x0e+8x1o`` → scalars ``8x0e``, one gate per gated multiplet, gated ``8x1o``."""
        scalars, gates, gated = _partition_gate(cue.Irreps("O3", TARGET))
        assert scalars == cue.Irreps("O3", f"{FEATURES}x0e")
        assert gates == cue.Irreps("O3", f"{FEATURES}x0e")
        assert gated == cue.Irreps("O3", f"{FEATURES}x1o")


class TestResidualInteraction:
    """Test the residual non-linear interaction layer."""

    def test_output_shape_is_multiplet_layout(self, layer, graph):
        """reshape_irreps emits ``(N, ir_dim, mul)``; the skip is ``(N, hidden)``."""
        out, skip = layer(**graph)
        assert out.shape == (N_NODES, 4, FEATURES)
        assert skip.shape == (N_NODES, FEATURES)

    def test_double_precision_roundtrip(self, layer, graph):
        """`.double()` must work: cuEq bakes its dtype in at construction."""
        out, skip = layer(**graph)
        assert out.dtype == torch.float64
        assert skip.dtype == torch.float64

    def test_message_matches_explicit_gather_scatter(self, layer, graph):
        """Kernel-fused gather/scatter equals the explicit ``x[source]`` → TP → scatter route.

        Pins both the fusion and the edge convention: the message on node
        ``i`` sums the tensor products over edges whose ``target`` is ``i``.
        """
        out, _ = layer(**graph)
        assert torch.allclose(out, _explicit_message(layer, graph), atol=1e-12)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    def test_cuda_matches_explicit_gather_scatter(self, layer, graph) -> None:
        """molix scatter on GPU matches the explicit gather → TP → scatter route."""
        layer = layer.cuda()
        gpu = {k: v.cuda() if torch.is_tensor(v) else v for k, v in graph.items()}
        out, _ = layer(**gpu)
        assert torch.allclose(out, _explicit_message(layer, gpu), atol=1e-10)

    def test_skip_depends_only_on_node_inputs(self, layer, graph):
        """``skip_tp`` reads node features only — not the edges."""
        _, skip_a = layer(**graph)
        rewired = dict(graph)
        rewired["edge_index"] = torch.tensor([[3, 2, 1, 0, 2], [0, 3, 2, 1, 0]]).t()  # (E, 2)
        _, skip_b = layer(**rewired)
        assert torch.allclose(skip_a, skip_b)

    def test_permutation_equivariance(self, layer, graph):
        """Relabelling atoms permutes the output identically."""
        perm = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(perm)
        out, skip = layer(**graph)

        permuted = dict(graph)
        permuted["node_attrs"] = graph["node_attrs"][perm]
        permuted["node_feats"] = graph["node_feats"][perm]
        permuted["edge_index"] = inverse[graph["edge_index"]]
        out_p, skip_p = layer(**permuted)

        assert torch.allclose(out_p, out[perm], atol=1e-12)
        assert torch.allclose(skip_p, skip[perm], atol=1e-12)

    def test_zero_cutoff_leaves_only_the_residual_path(self, layer, graph):
        """A zero envelope kills every message; what remains cannot see the edges."""
        n_edges = graph["edge_index"].shape[0]
        zero = torch.zeros(n_edges, 1, dtype=torch.float64)
        out_a, _ = layer(cutoff=zero, **graph)
        rewired = dict(graph)
        rewired["edge_index"] = torch.tensor([[3, 2, 1, 0, 2], [0, 3, 2, 1, 0]]).t()  # (E, 2)
        out_b, _ = layer(cutoff=zero, **rewired)
        assert torch.allclose(out_a, out_b, atol=1e-12)
        # ``linear_res`` keeps the node's own features alive, so it is not zero.
        assert torch.count_nonzero(out_a) > 0
