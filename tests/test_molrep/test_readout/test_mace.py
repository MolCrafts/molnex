"""Tests for molrep.readout.mace module."""

import pytest
import torch

from molix import config
from molrep.readout.mace import (
    LinearReadout,
    NonLinearBiasReadout,
    NonLinearReadout,
    ProductHead,
    ProductHeadSpec,
)

FEATURES = 8
MLP_DIM = 4
N_NODES = 4


class TestProductHeadSpec:
    """Test ProductHeadSpec configuration."""

    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = ProductHeadSpec(
            hidden_dim=64,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )
        assert spec.hidden_dim == 64
        assert spec.out_dim == 1
        assert spec.num_radial == 8
        assert spec.l_max == 2
        assert spec.max_body_order == 2
        assert spec.num_species == 10

    def test_invalid_hidden_dim(self):
        """Test validation for hidden_dim."""
        with pytest.raises(ValueError):
            ProductHeadSpec(hidden_dim=0, out_dim=1)

    def test_invalid_out_dim(self):
        """Test validation for out_dim."""
        with pytest.raises(ValueError):
            ProductHeadSpec(hidden_dim=64, out_dim=0)


class TestProductHead:
    """Test ProductHead prediction layer."""

    def test_initialization(self):
        """Test ProductHead initialization."""
        # hidden_dim is the mixed-l message dim = num_features * (l_max+1)**2.
        # For num_features=8, l_max=2 -> 8 * 9 = 72.
        head = ProductHead(
            hidden_dim=72,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )
        assert head.config.hidden_dim == 72
        assert head.config.out_dim == 1

    def test_forward_shape(self):
        """Test output shape."""
        head = ProductHead(
            hidden_dim=72,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )

        n_nodes = 20
        node_features = torch.randn(n_nodes, 72, dtype=config.ftype)
        atom_types = torch.randint(0, 10, (n_nodes,), dtype=torch.long)

        output = head(node_features, atom_types)
        assert output.shape == (n_nodes, 1)

    def test_different_output_dims(self):
        """Test with different output dimensions."""
        # num_features=4, l_max=2 (default) -> hidden_dim = 4 * 9 = 36.
        for out_dim in [1, 3, 5]:
            head = ProductHead(
                hidden_dim=36,
                out_dim=out_dim,
                num_species=5,
            )

            node_features = torch.randn(10, 36, dtype=config.ftype)
            atom_types = torch.randint(0, 5, (10,), dtype=torch.long)

            output = head(node_features, atom_types)
            assert output.shape == (10, out_dim)

    def test_differentiable(self):
        """Test that gradients flow through head."""
        head = ProductHead(
            hidden_dim=36,
            out_dim=1,
            num_species=5,
        )

        node_features = torch.randn(10, 36, requires_grad=True, dtype=config.ftype)
        atom_types = torch.randint(0, 5, (10,), dtype=torch.long)

        output = head(node_features, atom_types)
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()


@pytest.fixture
def scalar_feats():
    """Per-atom scalar node features ``(N, FEATURES)`` matching ``FEATURES x0e``."""
    torch.manual_seed(0)
    return torch.randn(N_NODES, FEATURES, dtype=config.ftype)


class TestLinearReadout:
    """Test the MACE ``LinearReadoutBlock`` port."""

    def test_forward_shape(self, scalar_feats):
        """Projects scalar node features to one number per atom."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e")
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """Sub-layer name ``linear`` must stay put — official weights load by key."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e")
        assert sorted(head.state_dict().keys()) == [
            "linear.f.m.graphs.0.graph.c0",
            "linear.weight",
        ]


class TestNonLinearReadout:
    """Test the bias-free gated readout (MACE-MP / MatPES last layer)."""

    def test_forward_shape(self, scalar_feats):
        """Two equivariant linears with a SiLU gate give one number per atom."""
        head = NonLinearReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """``linear_1`` / ``linear_2`` and no bias entries — MACE's bias-free variant."""
        head = NonLinearReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert sorted(head.state_dict().keys()) == [
            "linear_1.f.m.graphs.0.graph.c0",
            "linear_1.weight",
            "linear_2.f.m.graphs.0.graph.c0",
            "linear_2.weight",
        ]


class TestNonLinearBiasReadout:
    """Test the biased three-layer readout (MACE-OMOL variant)."""

    def test_forward_shape(self, scalar_feats):
        """``Linear → SiLU → linear_mid → SiLU → linear_2`` gives one number per atom."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """``linear_mid`` plus the two biases distinguish this from NonLinearReadout."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert sorted(head.state_dict().keys()) == [
            "linear_1.f.m.graphs.0.graph.c0",
            "linear_1.weight",
            "linear_2.bias",
            "linear_2.weight",
            "linear_mid.bias",
            "linear_mid.weight",
        ]
