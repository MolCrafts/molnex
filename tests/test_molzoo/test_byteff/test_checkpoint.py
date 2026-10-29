"""Unit tests for molzoo.byteff.checkpoint.ByteFFKeys."""

from __future__ import annotations

from molzoo.byteff.checkpoint import ByteFFKeys


class TestByteFFKeys:
    def test_a_slot_the_target_does_not_carry_is_dropped(self):
        table = ByteFFKeys(
            n_layers=1, head_layers=2, out_layers=2, pair_layers={}, slots=("bonds",)
        ).table()
        assert table["preff_block.pre_layers.PreMMBonded.pre_mlp.angle."] is None
        assert table["preff_block.pre_layers.PreLJEs.sigma_mlp."] is None
        assert (
            table["preff_block.pre_layers.PreMMBonded.out_mlp.bond_k.lins.1"]
            == "heads.bonds.out.k.1"
        )
