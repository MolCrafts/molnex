"""Tests for molzoo.mace.heads — the scalar readouts and the joint charge/spin embedding.

The whole ``test_mace`` suite builds at fp64 (``conftest._build_at_fp64``).
The readouts' sub-layer names are the official-checkpoint contract the key
tables in ``molzoo.mace.checkpoint`` rely on, so they are pinned here once.
"""

from __future__ import annotations

import pytest
import torch

from molix import config
from molzoo.mace.heads import (
    JointFeatureEmbedding,
    JointFeatureSpec,
    LinearReadout,
    NonLinearBiasReadout,
    NonLinearReadout,
)

FEATURES = 8
MLP_DIM = 4
N_NODES = 4


@pytest.fixture
def scalar_feats():
    """Per-atom scalar node features ``(N, FEATURES)`` matching ``FEATURES x0e``."""
    torch.manual_seed(0)
    return torch.randn(N_NODES, FEATURES, dtype=config.ftype)


class TestJointFeatureSpec:
    """Test JointFeatureSpec configuration."""

    def test_invalid_emb_dim(self):
        """Test validation for emb_dim."""
        with pytest.raises(ValueError):
            JointFeatureSpec(name="total_spin", kind="continuous", emb_dim=0, in_dim=1)


class TestJointFeatureEmbedding:
    """Test JointFeatureEmbedding module."""

    #: One spec per ``kind``; ``per="graph"`` matches the OMOL charge/spin use.
    SPECS = {
        "categorical": JointFeatureSpec(
            name="total_charge", kind="categorical", emb_dim=4, num_classes=5, offset=2
        ),
        "continuous": JointFeatureSpec(name="total_spin", kind="continuous", emb_dim=4, in_dim=1),
    }

    def test_builds_and_runs_at_the_configured_precision(self):
        """Both kinds' parameters are fp64 under fp64; per-graph features fuse per atom."""
        embedding = JointFeatureEmbedding(
            feature_specs=[self.SPECS["categorical"], self.SPECS["continuous"]],
            out_dim=8,
        )
        assert {p.dtype for p in embedding.parameters()} == {torch.float64}

        n_atoms = 3
        output = embedding(
            batch=torch.zeros(n_atoms, dtype=torch.long),
            total_charge=torch.zeros(1, dtype=torch.long),
            total_spin=torch.ones(1, dtype=torch.float64),
        )

        assert output.shape == (n_atoms, 8)
        assert output.dtype == torch.float64

    def test_empty_specs_error(self):
        """Test error when no feature specs are provided."""
        with pytest.raises(ValueError, match="feature_specs must be non-empty."):
            JointFeatureEmbedding(feature_specs=[], out_dim=8)


class TestLinearReadout:
    """Test the MACE ``LinearReadoutBlock`` port."""

    def test_forward_shape(self, scalar_feats):
        """Projects scalar node features to one number per atom."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e")
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """Sub-layer name ``linear`` must stay put — official weights load by key."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e")
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
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
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
            "linear_1.weight",
            "linear_2.weight",
        ]


class TestNonLinearBiasReadout:
    """Test the biased three-layer readout (MACE-OMOL variant)."""

    def test_forward_shape(self, scalar_feats):
        """``Linear → SiLU → linear_mid → SiLU → linear_2`` gives one number per atom."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_untrained_output_is_not_constant_across_atoms(self, scalar_feats):
        """An *untrained* readout must still resolve two different atoms.

        A constant output here means the model's interaction energy is
        position-independent and its forces are identically zero, which
        silently voids every force / parity assertion built on a fresh model.
        """
        torch.manual_seed(0)
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)

        energies = head(scalar_feats)

        assert not torch.allclose(energies[0], energies[1])

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """``linear_mid`` plus the two biases distinguish this from NonLinearReadout."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM)
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
            "linear_1.weight",
            "linear_2.bias",
            "linear_2.weight",
            "linear_mid.bias",
            "linear_mid.weight",
        ]
