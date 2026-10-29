"""Tests for molzoo.mace.heads — the scalar readouts.

The whole ``test_mace`` suite builds at fp64 (``conftest.DTYPE``).
The readouts' sub-layer names are the official-checkpoint contract the key
tables in ``molzoo.mace.checkpoint`` rely on, so they are pinned here once.
"""

from __future__ import annotations

import pytest
import torch

from molzoo.mace.heads import (
    LinearReadout,
    NonLinearBiasReadout,
    NonLinearReadout,
)
from tests.test_molzoo.test_mace.conftest import DTYPE

FEATURES = 8
MLP_DIM = 4
N_NODES = 4


@pytest.fixture
def scalar_feats():
    """Per-atom scalar node features ``(N, FEATURES)`` matching ``FEATURES x0e``."""
    torch.manual_seed(0)
    return torch.randn(N_NODES, FEATURES, dtype=DTYPE)


class TestLinearReadout:
    """Test the MACE ``LinearReadoutBlock`` port."""

    def test_forward_shape(self, scalar_feats):
        """Projects scalar node features to one number per atom."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e", dtype=DTYPE)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """Sub-layer name ``linear`` must stay put — official weights load by key."""
        head = LinearReadout(irreps_in=f"{FEATURES}x0e", dtype=DTYPE)
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
            "linear.weight",
        ]


class TestNonLinearReadout:
    """Test the bias-free gated readout (MACE-MP / MatPES last layer)."""

    def test_forward_shape(self, scalar_feats):
        """Two equivariant linears with a SiLU gate give one number per atom."""
        head = NonLinearReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM, dtype=DTYPE)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """``linear_1`` / ``linear_2`` and no bias entries — MACE's bias-free variant."""
        head = NonLinearReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM, dtype=DTYPE)
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
            "linear_1.weight",
            "linear_2.weight",
        ]


class TestNonLinearBiasReadout:
    """Test the biased three-layer readout (MACE-OMOL variant)."""

    def test_forward_shape(self, scalar_feats):
        """``Linear → SiLU → linear_mid → SiLU → linear_2`` gives one number per atom."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM, dtype=DTYPE)
        assert head(scalar_feats).shape == (N_NODES, 1)

    def test_untrained_output_is_not_constant_across_atoms(self, scalar_feats):
        """An *untrained* readout must still resolve two different atoms.

        A constant output here means the model's interaction energy is
        position-independent and its forces are identically zero, which
        silently voids every force / parity assertion built on a fresh model.
        """
        torch.manual_seed(0)
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM, dtype=DTYPE)

        energies = head(scalar_feats)

        assert not torch.allclose(energies[0], energies[1])

    def test_state_dict_keys_are_the_weight_transfer_contract(self):
        """``linear_mid`` plus the two biases distinguish this from NonLinearReadout."""
        head = NonLinearBiasReadout(irreps_in=f"{FEATURES}x0e", mlp_dim=MLP_DIM, dtype=DTYPE)
        assert sorted(k for k in head.state_dict() if ".graph.c" not in k) == [
            "linear_1.weight",
            "linear_2.bias",
            "linear_2.weight",
            "linear_mid.bias",
            "linear_mid.weight",
        ]
