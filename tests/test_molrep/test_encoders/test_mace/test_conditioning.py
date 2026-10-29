"""Tests for molrep.encoders.mace.conditioning — the joint charge/spin embedding."""

from __future__ import annotations

import pytest
import torch

from molrep.encoders.mace.conditioning import JointFeatureEmbedding, JointFeatureSpec
from tests.test_molzoo.test_mace.conftest import DTYPE


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

    def test_builds_and_runs_at_the_given_dtype(self):
        """Both kinds' parameters are built at ``DTYPE``; per-graph features fuse per atom."""
        embedding = JointFeatureEmbedding(
            feature_specs=[self.SPECS["categorical"], self.SPECS["continuous"]],
            out_dim=8,
            dtype=DTYPE,
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
            JointFeatureEmbedding(feature_specs=[], out_dim=8, dtype=DTYPE)
