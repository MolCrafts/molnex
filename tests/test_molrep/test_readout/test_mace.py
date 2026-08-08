"""Tests for molrep.readout.mace module."""

import cuequivariance_torch as cuet
import pytest
import torch
import torch.nn as nn

from molix import config
from molrep.interaction.contraction import SymmetricContraction
from molrep.readout.mace import (
    LinearReadout,
    NonLinearBiasReadout,
    NonLinearReadout,
    ProductHead,
    ProductHeadSpec,
)
from molrep.readout.projection import BasisProjection

FEATURES = 8
MLP_DIM = 4
N_NODES = 4

#: Configuration inherited with the cases migrated from
#: ``tests/test_molzoo/test_mace.py`` (mace-subpackage-restructure-02-core).
#: ``hidden_dim`` is the mixed-l message width ``num_features * (l_max+1)**2``
#: = 64 * 9 = 576 for num_features=64, l_max=2.
MIGRATED_CONFIG = {
    "hidden_dim": 576,
    "out_dim": 64,
    "num_radial": 8,
    "l_max": 2,
    "max_body_order": 2,
    "num_species": 118,
}
MIGRATED_NUM_FEATURES = MIGRATED_CONFIG["hidden_dim"] // (MIGRATED_CONFIG["l_max"] + 1) ** 2


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

    # -- migrated from tests/test_molzoo/test_mace.py by
    #    mace-subpackage-restructure-02-core (module/package name clash forced
    #    that file's removal; these cases have no equivalent above) ----------

    @pytest.fixture
    def migrated_head(self):
        """A ProductHead on the wider migrated configuration."""
        return ProductHead(**MIGRATED_CONFIG)

    def test_initialization_wires_components(self, migrated_head):
        """Every sub-block is present and of the expected type."""
        assert isinstance(migrated_head.symmetric_contraction, SymmetricContraction)
        assert isinstance(migrated_head.basis_projection, BasisProjection)
        assert isinstance(migrated_head.linear, nn.Linear)
        # The contraction emits invariant scalars (num_features), so the readout
        # linear maps num_features -> out_dim (not the full mixed-l hidden_dim).
        assert migrated_head.linear.in_features == MIGRATED_NUM_FEATURES
        assert migrated_head.linear.out_features == MIGRATED_CONFIG["out_dim"]

    def test_symmetric_contraction_config(self, migrated_head):
        """Constructor kwargs reach the SymmetricContraction spec."""
        sc = migrated_head.symmetric_contraction
        assert sc.config.hidden_dim == MIGRATED_CONFIG["hidden_dim"]
        assert sc.config.num_species == MIGRATED_CONFIG["num_species"]
        assert sc.config.max_body_order == MIGRATED_CONFIG["max_body_order"]

    def test_basis_projection_config(self, migrated_head):
        """Constructor kwargs reach the BasisProjection spec."""
        bp = migrated_head.basis_projection
        assert bp.config.hidden_dim == MIGRATED_CONFIG["hidden_dim"]
        assert bp.config.num_radial == MIGRATED_CONFIG["num_radial"]
        assert bp.config.l_max == MIGRATED_CONFIG["l_max"]
        assert bp.config.max_body_order == MIGRATED_CONFIG["max_body_order"]

    def test_forward_preserves_dtype(self, migrated_head):
        """The head does not silently up/down-cast the working precision."""
        node_features = torch.randn(10, MIGRATED_CONFIG["hidden_dim"], dtype=config.ftype)
        atom_types = torch.randint(0, MIGRATED_CONFIG["num_species"], (10,), dtype=torch.long)

        assert migrated_head(node_features, atom_types).dtype == config.ftype

    @pytest.mark.parametrize("max_body_order", [1, 2, 3])
    def test_different_max_body_orders(self, max_body_order):
        """Body order propagates to both the contraction and the projection."""
        head = ProductHead(**{**MIGRATED_CONFIG, "max_body_order": max_body_order})
        assert head.symmetric_contraction.config.max_body_order == max_body_order
        assert head.basis_projection.config.max_body_order == max_body_order

    @pytest.mark.parametrize("num_species", [10, 50, 118])
    def test_different_num_species(self, num_species):
        """The element table size propagates to the contraction."""
        head = ProductHead(**{**MIGRATED_CONFIG, "num_species": num_species})
        assert head.symmetric_contraction.config.num_species == num_species

    def test_symmetric_contraction_component(self, migrated_head):
        """The contraction maps the mixed-l message to invariant scalars."""
        n_nodes = 15
        node_features = torch.randn(n_nodes, MIGRATED_CONFIG["hidden_dim"], dtype=config.ftype)
        atom_types = torch.randint(0, MIGRATED_CONFIG["num_species"], (n_nodes,), dtype=torch.long)

        basis = migrated_head.symmetric_contraction(node_features, atom_types)
        assert basis.shape == (n_nodes, MIGRATED_NUM_FEATURES)

    def test_basis_projection_component(self, migrated_head):
        """``BasisProjection`` is an identity in the current implementation."""
        basis = torch.randn(15, MIGRATED_CONFIG["hidden_dim"], dtype=config.ftype)
        assert torch.equal(migrated_head.basis_projection(basis), basis)

    def test_linear_component(self, migrated_head):
        """The readout linear consumes the contracted scalars."""
        features = torch.randn(15, MIGRATED_NUM_FEATURES, dtype=config.ftype)
        assert migrated_head.linear(features).shape == (15, MIGRATED_CONFIG["out_dim"])

    def test_cuequivariance_integration(self, migrated_head):
        """The contraction is a cuEq ``SymmetricContraction``, not a port."""
        cue_sc = migrated_head.symmetric_contraction.symmetric_contraction
        assert isinstance(cue_sc, cuet.SymmetricContraction)
        assert hasattr(cue_sc, "contraction_degree")
        assert hasattr(cue_sc, "num_elements")


class TestProductHeadEquivariance:
    """Test equivariance properties of ProductHead.

    Migrated from ``tests/test_molzoo/test_mace.py`` by
    mace-subpackage-restructure-02-core.
    """

    @pytest.fixture
    def product_head(self):
        """A small ProductHead (num_features=32, l_max=1 -> hidden_dim=128)."""
        return ProductHead(
            hidden_dim=128,
            out_dim=32,
            num_radial=8,
            l_max=1,
            max_body_order=2,
            num_species=10,
        )

    def test_permutation_equivariance(self, product_head):
        """Relabelling atoms permutes the per-atom output the same way."""
        n_nodes = 15
        torch.manual_seed(0)
        node_features = torch.randn(n_nodes, 128, dtype=config.ftype)
        atom_types = torch.randint(0, 10, (n_nodes,), dtype=torch.long)

        output1 = product_head(node_features, atom_types)

        perm = torch.randperm(n_nodes)
        output2 = product_head(node_features[perm], atom_types[perm])

        assert torch.allclose(output1[perm], output2, rtol=1e-5, atol=1e-5)


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
