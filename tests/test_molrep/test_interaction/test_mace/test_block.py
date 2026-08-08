"""Tests for molrep.interaction.mace.block module."""

import cuequivariance_torch as cuet
import pytest
import torch
import torch.nn as nn

from molrep.interaction.mace.block import InteractionBlock, InteractionSpec
from molrep.interaction.mace.conv import ConvTP
from molrep.interaction.radial import RadialWeightMLP

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

    # -- migrated from tests/test_molzoo/test_mace.py by
    #    mace-subpackage-restructure-02-core (module/package name clash forced
    #    that file's removal; these cases have no equivalent above) ----------

    @pytest.fixture
    def interaction_config(self):
        """Wider configuration inherited with the migrated cases."""
        return {
            "num_features": 64,
            "num_bessel": 8,
            "l_max": 2,
            "avg_num_neighbors": 10.0,
        }

    @pytest.fixture
    def interaction_block(self, interaction_config):
        """Create an InteractionBlock instance."""
        return InteractionBlock(**interaction_config)

    def test_initialization_wires_components(self, interaction_block, interaction_config):
        """Every sub-block is present and of the expected type."""
        # conv_tp must be created first to provide weight_numel
        assert isinstance(interaction_block.conv_tp, ConvTP)
        assert isinstance(interaction_block.node_linear, cuet.Linear)
        assert isinstance(interaction_block.radial_mlp, RadialWeightMLP)
        assert isinstance(interaction_block.linear, cuet.Linear)
        assert interaction_block.avg_num_neighbors == interaction_config["avg_num_neighbors"]

    def test_initialization_order_fix(self, interaction_config):
        """conv_tp is initialized before radial_mlp (regression guard).

        Building the radial MLP first raised ``AttributeError: self.conv_tp``,
        because the MLP's output width is ``conv_tp.weight_numel``.
        """
        block = InteractionBlock(**interaction_config)
        assert hasattr(block.conv_tp, "weight_numel")
        assert block.radial_mlp.mlp[-1].out_features == block.conv_tp.weight_numel

    def test_config_storage(self, interaction_block, interaction_config):
        """Constructor kwargs round-trip through the pydantic ``config``."""
        config = interaction_block.config
        assert config.num_features == interaction_config["num_features"]
        assert config.num_bessel == interaction_config["num_bessel"]
        assert config.l_max == interaction_config["l_max"]
        assert config.avg_num_neighbors == interaction_config["avg_num_neighbors"]

    def test_radial_mlp_architecture(self, interaction_block, interaction_config):
        """``Linear → SiLU → Linear → SiLU → Linear`` with MACE's widths."""
        mlp = interaction_block.radial_mlp.mlp
        num_features = interaction_config["num_features"]
        num_bessel = interaction_config["num_bessel"]
        weight_numel = interaction_block.conv_tp.weight_numel

        assert len(mlp) == 5
        assert isinstance(mlp[0], nn.Linear)
        assert isinstance(mlp[1], nn.SiLU)
        assert isinstance(mlp[2], nn.Linear)
        assert isinstance(mlp[3], nn.SiLU)
        assert isinstance(mlp[4], nn.Linear)

        assert mlp[0].in_features == num_bessel
        assert mlp[0].out_features == num_features
        assert mlp[2].in_features == num_features
        assert mlp[2].out_features == num_features
        assert mlp[4].in_features == num_features
        assert mlp[4].out_features == weight_numel

    def test_cuequivariance_integration(self, interaction_block):
        """The convolution is a cuEq ``ChannelWiseTensorProduct``, not a port."""
        cue_tp = interaction_block.conv_tp.cue_tp
        assert isinstance(cue_tp, cuet.ChannelWiseTensorProduct)
        assert hasattr(cue_tp, "irreps_in1")
        assert hasattr(cue_tp, "irreps_in2")
        assert hasattr(cue_tp, "irreps_out")

    @pytest.mark.parametrize("l_max", [1, 2, 3])
    def test_different_l_max_values(self, interaction_config, l_max):
        """Construction succeeds for every angular order MACE ships."""
        block = InteractionBlock(
            num_features=interaction_config["num_features"],
            num_bessel=interaction_config["num_bessel"],
            l_max=l_max,
            avg_num_neighbors=interaction_config["avg_num_neighbors"],
        )
        assert block.config.l_max == l_max
        assert hasattr(block.conv_tp, "weight_numel")

    @pytest.mark.parametrize("num_features", [32, 64, 128])
    def test_different_num_features(self, interaction_config, num_features):
        """The radial MLP hidden width tracks ``num_features``."""
        block = InteractionBlock(
            num_features=num_features,
            num_bessel=interaction_config["num_bessel"],
            l_max=interaction_config["l_max"],
            avg_num_neighbors=interaction_config["avg_num_neighbors"],
        )
        assert block.radial_mlp.mlp[0].out_features == num_features
        assert block.radial_mlp.mlp[2].in_features == num_features
