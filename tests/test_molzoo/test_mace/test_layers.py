"""Tests for molzoo.mace.layers.

The whole ``test_mace`` suite builds at fp64 (``conftest.DTYPE``):
cuEquivariance fixes its precision at construction.
"""

from __future__ import annotations

import cuequivariance as cue
import pytest
import torch

from molix.F.scatter import scatter_sum
from molrep import cueq
from molzoo.mace.checkpoint import element_linear_weight, one_hot_weight
from molzoo.mace.layers import (
    AgnosticInteraction,
    AgnosticResidualInteraction,
    DensityInteraction,
    DensityResidualInteraction,
    ElementIndex,
    ElementLinear,
    EquivariantProductBasis,
    OneHotLinear,
    RadialMLP,
    ResidualInteraction,
    _partition_gate,
)
from tests.test_molzoo.test_mace.conftest import DTYPE

N_ELEMENTS = 3
N_NODES = 4
NUM_RADIAL = 5
FEATURES = 8
SH = "1x0e+1x1o"
TARGET = f"{FEATURES}x0e+{FEATURES}x1o"
TARGET_DIM = 4 * FEATURES  # 8x0e + 8x1o
HIDDEN = f"{FEATURES}x0e"
#: Element row of each node of the ``graph`` fixture.
ELEMENT_ROWS = torch.tensor([0, 1, 2, 0])


def _elements(index: torch.Tensor) -> ElementIndex:
    """An ``ElementIndex`` of the given rows, as the encoder builds it."""
    return ElementIndex(index=index, index32=index.to(torch.int32))


@pytest.fixture
def graph():
    """A small directed graph plus the edge/node inputs the blocks consume."""
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]).t().contiguous()  # (E, 2)
    n_edges = edge_index.shape[0]
    return {
        "elements": _elements(ELEMENT_ROWS),
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
        node_feats_irreps=f"{FEATURES}x0e",
        edge_irreps=f"{FEATURES}x0e",
        **_common(),
        dtype=DTYPE,
    )


@pytest.fixture
def residual_layer():
    return DensityResidualInteraction(
        node_feats_irreps=f"{FEATURES}x0e",
        edge_irreps=f"{FEATURES}x0e",
        hidden_irreps=f"{FEATURES}x0e",
        **_common(),
        dtype=DTYPE,
    )


def _density_message(layer, graph):
    """The pre-fusion formulation: gather ``x[source]`` → per-edge TP → scatter on ``target``.

    Spelled out with the block's own sub-layers so the only thing the block
    can differ in is how the gather/scatter reaches the kernel.
    """
    source, target = graph["edge_index"][:, 0], graph["edge_index"][:, 1]
    n = graph["node_feats"].shape[0]
    up = layer.linear_up(graph["node_feats"])
    tp_weights = layer.conv_tp_weights(graph["edge_feats"])
    density = scatter_sum(
        torch.tanh(layer.density_fn(graph["edge_feats"]) ** 2), target, dim=0, dim_size=n
    )
    mji = layer.conv_tp(up[source], graph["edge_attrs"], tp_weights)
    message = scatter_sum(mji, target, dim=0, dim_size=n)
    return layer.linear(message) / (density + 1.0)


class TestDensityInteraction:
    """Test the first-layer density-normalised interaction."""

    def test_output_is_the_target_irreps_vector(self, first_layer, graph):
        """The message is handed on as ``(N, target.dim)`` in ``ir_mul`` — no reshape."""
        out, _ = first_layer(**graph)
        assert out.shape == (N_NODES, TARGET_DIM)

    def test_message_matches_explicit_gather_scatter(self, first_layer, graph):
        """Kernel-fused gather/scatter equals the explicit ``x[source]`` → TP → scatter route.

        Pins both the fusion and the edge convention: the message on node
        ``i`` sums the tensor products over edges whose ``target`` is ``i``.
        """
        out, _ = first_layer(**graph)
        expected = first_layer.skip_tp(_density_message(first_layer, graph), graph["elements"])
        assert torch.allclose(out, expected, atol=1e-12)

    def test_returns_no_skip_connection(self, first_layer, graph):
        """The first layer has no residual to carry — MACE returns ``None``."""
        assert first_layer(**graph)[1] is None

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

    def test_output_is_the_target_irreps_vector(self, residual_layer, graph):
        """Message keeps the same ``(N, target.dim)`` layout as the first layer."""
        out, _ = residual_layer(**graph)
        assert out.shape == (N_NODES, TARGET_DIM)

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
        permuted["elements"] = _elements(ELEMENT_ROWS[perm])
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


AVG_NUM_NEIGHBORS = 2.5


def _agnostic_message(layer, graph):
    """Gather → per-edge TP → scatter on ``target``, divided by ``avg_num_neighbors``."""
    source, target = graph["edge_index"][:, 0], graph["edge_index"][:, 1]
    n = graph["node_feats"].shape[0]
    up = layer.linear_up(graph["node_feats"])
    tp_weights = layer.conv_tp_weights(graph["edge_feats"])
    mji = layer.conv_tp(up[source], graph["edge_attrs"], tp_weights)
    message = scatter_sum(mji, target, dim=0, dim_size=n)
    return layer.linear(message) / AVG_NUM_NEIGHBORS


class TestAgnosticInteraction:
    """First-layer interaction normalised by ``avg_num_neighbors`` (MACE ``RealAgnostic``)."""

    def test_message_is_the_explicit_sum_over_avg_num_neighbors(self, graph):
        layer = AgnosticInteraction(
            node_feats_irreps=f"{FEATURES}x0e",
            edge_irreps=f"{FEATURES}x0e",
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            **_common(),
            dtype=DTYPE,
        )
        out, skip = layer(**graph)
        expected = layer.skip_tp(_agnostic_message(layer, graph), graph["elements"])
        assert skip is None
        torch.testing.assert_close(out, expected)


class TestAgnosticResidualInteraction:
    """Residual interaction normalised by ``avg_num_neighbors`` (MACE ``RealAgnosticResidual``)."""

    def test_message_and_skip(self, graph):
        layer = AgnosticResidualInteraction(
            node_feats_irreps=f"{FEATURES}x0e",
            edge_irreps=f"{FEATURES}x0e",
            hidden_irreps=f"{FEATURES}x0e",
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            **_common(),
            dtype=DTYPE,
        )
        out, skip = layer(**graph)
        torch.testing.assert_close(out, _agnostic_message(layer, graph))
        torch.testing.assert_close(skip, layer.skip_tp(graph["node_feats"], graph["elements"]))


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
        dtype=DTYPE,
    )


def _residual_message(layer: ResidualInteraction, graph: dict) -> torch.Tensor:
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
            layer.source_embedding(graph["elements"].index)[source],
            layer.target_embedding(graph["elements"].index)[target],
        ],
        dim=-1,
    )
    tp_weights = layer.conv_tp_weights(edge_feats)
    density = scatter_sum(torch.tanh(layer.density_fn(edge_feats) ** 2), target, dim=0, dim_size=n)
    mji = layer.conv_tp(up[source], graph["edge_attrs"], tp_weights)
    message = scatter_sum(mji, target, dim=0, dim_size=n)
    message = layer.linear_1(message) / (density * layer.beta + layer.alpha)
    message = message + layer.linear_res(up)
    return layer.linear_2(layer.equivariant_nonlin(message))


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

    def test_output_is_the_target_irreps_vector(self, layer, graph):
        """The message is ``(N, target.dim)`` in ``ir_mul``; the skip is ``(N, hidden)``."""
        out, skip = layer(**graph)
        assert out.shape == (N_NODES, TARGET_DIM)
        assert skip.shape == (N_NODES, FEATURES)

    def test_message_matches_explicit_gather_scatter(self, layer, graph):
        """Kernel-fused gather/scatter equals the explicit ``x[source]`` → TP → scatter route.

        Pins both the fusion and the edge convention: the message on node
        ``i`` sums the tensor products over edges whose ``target`` is ``i``.
        """
        out, _ = layer(**graph)
        assert torch.allclose(out, _residual_message(layer, graph), atol=1e-12)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    def test_cuda_matches_explicit_gather_scatter(self, layer, graph) -> None:
        """molix scatter on GPU matches the explicit gather → TP → scatter route."""
        layer = layer.cuda()
        gpu = {k: v.cuda() if torch.is_tensor(v) else v for k, v in graph.items()}
        out, _ = layer(**gpu)
        assert torch.allclose(out, _residual_message(layer, gpu), atol=1e-10)

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
        permuted["elements"] = _elements(ELEMENT_ROWS[perm])
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


class TestRadialMLP:
    """Evaluating layer 0 from a partition of its inputs is the same function.

    The MACE-family interactions rely on this to avoid concatenating
    ``[edge_feats, source_emb[src], target_emb[tgt]]`` per edge — 1056 channels
    at Polar-1-M widths, which is both a multi-GiB tensor and the source of an
    Inductor ``int32`` index constant that overflows above ~2·10⁶ edges.
    """

    def test_split_parts_plus_bias_reproduce_forward(self):
        torch.manual_seed(0)
        mlp = RadialMLP([12, 16, 5], dtype=DTYPE)
        x = torch.randn(7, 12, dtype=torch.float64)
        widths = [4, 5, 3]
        weights = mlp.first_layer_parts(widths)
        hidden = sum(part @ w for part, w in zip(x.split(widths, dim=-1), weights, strict=True))
        bias = mlp.first_layer_bias()
        assert bias is not None
        got = mlp.forward_from_first(hidden + bias)
        torch.testing.assert_close(got, mlp(x), atol=1e-12, rtol=0)

    def test_single_layer_network_needs_no_activation(self):
        """``forward_from_first`` on a one-layer net is the identity."""
        mlp = RadialMLP([6, 3], dtype=DTYPE)
        hidden = torch.randn(4, 3, dtype=torch.float64)
        torch.testing.assert_close(mlp.forward_from_first(hidden), hidden, atol=0, rtol=0)

    def test_widths_must_partition_the_input(self):
        with pytest.raises(ValueError, match="do not partition"):
            RadialMLP([12, 16, 5], dtype=DTYPE).first_layer_parts([4, 4])


def _pair(irreps_in: str, num_elements: int, irreps_out: str):
    """The fully connected product and an ElementLinear holding its weight.

    The product's flat weight reaches the module the way an official
    checkpoint's does: through the boundary conversion.
    """
    i1, i3 = cue.Irreps("O3", irreps_in), cue.Irreps("O3", irreps_out)
    attrs = cue.Irreps("O3", f"{num_elements}x0e")
    product = cueq.FullyConnectedTensorProduct(
        i1, attrs, i3, layout=cue.ir_mul, dtype=torch.float64
    )
    linear = ElementLinear(i1, attrs, i3, dtype=torch.float64)
    with torch.no_grad():
        linear.weight.copy_(element_linear_weight(linear, product.weight.detach()))
    return product, linear, i1


class TestElementLinear:
    @pytest.mark.parametrize(
        ("irreps_in", "num_elements", "irreps_out"),
        [
            ("4x0e+4x1o", 3, "5x0e+5x1o"),
            ("8x0e+8x1o+8x2e", 7, "8x0e+8x1o"),  # an input irrep with no output
            ("6x0e", 5, "6x0e+6x1o"),  # an output irrep no path reaches
        ],
    )
    def test_equals_the_product_against_a_one_hot(self, irreps_in, num_elements, irreps_out):
        torch.manual_seed(0)
        product, linear, i1 = _pair(irreps_in, num_elements, irreps_out)
        x = torch.randn(11, i1.dim, dtype=torch.float64)
        element = torch.randint(0, num_elements, (11,))
        one_hot = torch.nn.functional.one_hot(element, num_elements).double()
        torch.testing.assert_close(
            linear(x, _elements(element)),
            product(x, one_hot),
            atol=1e-13,
            rtol=0.0,
        )

    def test_weight_is_one_row_per_element(self):
        """``(C, Σ_p u_p·w_p)``: element ``e``'s ``(u, w)`` block of every path is row ``e``."""
        _, linear, _ = _pair("4x0e+4x1o", 3, "5x0e+5x1o")
        assert linear.weight.shape == (3, 2 * 4 * 5)

    def test_attributes_must_be_scalars(self):
        with pytest.raises(ValueError, match="Cx0e"):
            ElementLinear(
                cue.Irreps("O3", "4x0e"),
                cue.Irreps("O3", "2x0e+1x1o"),
                cue.Irreps("O3", "4x0e"),
                dtype=torch.float64,
            )


class TestOneHotLinear:
    def test_equals_the_linear_over_a_one_hot(self):
        """The row lookup is ``cuet.Linear(one_hot)`` with the linear's own weight."""
        torch.manual_seed(0)
        attrs, out = cue.Irreps("O3", "7x0e"), cue.Irreps("O3", "5x0e")
        linear = cueq.Linear(attrs, out, layout=cue.ir_mul, dtype=torch.float64)
        lookup = OneHotLinear(attrs, out, dtype=torch.float64)
        with torch.no_grad():
            lookup.weight.copy_(one_hot_weight(lookup, linear.weight.detach()))
        element = torch.randint(0, 7, (13,))
        one_hot = torch.nn.functional.one_hot(element, 7).double()
        torch.testing.assert_close(lookup(element), linear(one_hot), atol=1e-12, rtol=0.0)

    def test_weight_is_the_row_table(self):
        """``(C, F)``: the lookup indexes the weight directly, no per-call view."""
        lookup = OneHotLinear(
            cue.Irreps("O3", "7x0e"), cue.Irreps("O3", "5x0e"), dtype=torch.float64
        )
        assert lookup.weight.shape == (7, 5)

    @pytest.mark.parametrize(
        ("irreps_in", "irreps_out"),
        [("3x0e+1x1o", "4x0e"), ("3x0e", "4x0e+4x1o")],
    )
    def test_rejects_anything_but_scalars_to_scalars(self, irreps_in, irreps_out):
        with pytest.raises(ValueError, match="x0e"):
            OneHotLinear(
                cue.Irreps("O3", irreps_in), cue.Irreps("O3", irreps_out), dtype=torch.float64
            )


MUL = 4
IRREPS_IN = f"{MUL}x0e+{MUL}x1o"
IRREPS_OUT = f"{MUL}x0e+{MUL}x1o"
N = 5


@pytest.fixture
def feats() -> torch.Tensor:
    """``(N, irreps_in.dim)`` = ``(5, 4 + 12)`` features in ``ir_mul``."""
    torch.manual_seed(0)
    return torch.randn(N, 4 * MUL, dtype=DTYPE)


def _block(**kw) -> EquivariantProductBasis:
    torch.manual_seed(1)
    return EquivariantProductBasis(
        node_feats_irreps=IRREPS_IN,
        target_irreps=IRREPS_OUT,
        correlation=2,
        **kw,
        dtype=DTYPE,
    )


class TestEquivariantProductBasis:
    def test_output_has_the_target_width(self, feats):
        out = _block()(feats, None, _elements(torch.zeros(N, dtype=torch.long)))
        assert out.shape == (N, 4 * MUL)

    def test_skip_connection_is_added(self, feats):
        block = _block()
        sc = torch.randn(N, 4 * MUL, dtype=DTYPE)
        elements = _elements(torch.zeros(N, dtype=torch.long))
        torch.testing.assert_close(block(feats, sc, elements) - block(feats, None, elements), sc)

    def test_use_sc_false_ignores_the_skip(self, feats):
        block = _block(use_sc=False)
        sc = torch.randn(N, 4 * MUL, dtype=DTYPE)
        elements = _elements(torch.zeros(N, dtype=torch.long))
        torch.testing.assert_close(block(feats, sc, elements), block(feats, None, elements))

    def test_shared_product_ignores_the_element(self, feats):
        """``num_elements=1``: every atom reads row 0 whatever its element row is."""
        block = _block()
        mixed = _elements(torch.tensor([0, 1, 2, 1, 0]))
        zeros = _elements(torch.zeros(N, dtype=torch.long))
        torch.testing.assert_close(block(feats, None, mixed), block(feats, None, zeros))

    def test_per_element_weights_depend_on_the_element(self, feats):
        block = _block(num_elements=2)
        same = feats[:1].expand(2, -1)
        out = block(same, None, _elements(torch.tensor([0, 1])))
        assert not torch.allclose(out[0], out[1])
