"""Unit tests for molzoo.byteff.potential.ByteFF."""

from __future__ import annotations

import pytest
import torch

from molpot.classical import BondChargeCorrection, LJCutDirect
from molzoo.byteff import ByteFF
from tests.test_molzoo.test_byteff.conftest import WEIGHTS_DIR

DTYPE = torch.float64


class TestByteFF:
    def test_the_pair_styles_take_the_bond_charge_and_direct_lj_heads(self, forcefield):
        model = ByteFF(forcefield, dtype=DTYPE)
        assert isinstance(model.heads["bcc"], BondChargeCorrection)
        assert isinstance(model.heads["lj_direct"], LJCutDirect)

    @pytest.mark.skipif(not WEIGHTS_DIR.is_dir(), reason="MOLNEX_BYTEFF_WEIGHTS_DIR not set")
    def test_load_fills_every_tensor_of_the_model(self, forcefield):
        path = WEIGHTS_DIR / "member-01" / "optimal.pt"
        model = ByteFF.load(path, forcefield, dtype=DTYPE)
        official = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        assert len(model.state_dict()) == len(official)
        jk = official["graph_block.gnn_layer.gnn.lin.weight"].to(DTYPE)
        torch.testing.assert_close(model.graph.jk.weight, jk, rtol=0, atol=0)
