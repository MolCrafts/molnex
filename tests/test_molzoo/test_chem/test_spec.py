"""Every :class:`ChemPerceptionSpec` field reaches the encoder it configures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molzoo.chem import ChemPerception, ChemPerceptionSpec


class TestChemPerceptionSpec:
    @pytest.mark.parametrize("n_blocks", [0, 1, 3])
    def test_depth_reaches_the_encoder(self, n_blocks):
        # A spec field that never reaches the constructor is a knob that
        # silently does nothing — the receptive field would stay at the
        # parent's default no matter what the caller asked for.
        model = ChemPerception(ChemPerceptionSpec(atom_dim=8, bond_dim=8, n_blocks=n_blocks))
        assert len(model.blocks) == n_blocks

    def test_local_attention_reaches_the_blocks(self):
        model = ChemPerception(
            ChemPerceptionSpec(atom_dim=8, bond_dim=8, n_blocks=1, local_attention=True)
        )
        assert all(block.local_attention for block in model.blocks)

    def test_rejects_a_negative_depth(self):
        with pytest.raises(ValidationError):
            ChemPerceptionSpec(n_blocks=-1)
