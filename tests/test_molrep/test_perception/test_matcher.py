"""SmartsMatcher Protocol + Fake / Molpy matchers (learnable-classical-ff-07)."""

from __future__ import annotations

import pytest
import torch

from molrep.perception import (
    FakeSmartsMatcher,
    MolpySmartsMatcher,
    SymbolicPattern,
)


class TestFakeSmartsMatcher:
    def test_returns_configured_hits_shape(self):
        hits = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # [2, 2]
        matcher = FakeSmartsMatcher({"[#6]-[#6]": hits})
        pat = SymbolicPattern("[#6]-[#6]", arity=2)
        out = matcher.match(mol=None, pattern=pat)
        assert out.shape == (2, 2)
        assert torch.equal(out, hits)
        assert out.dtype == torch.long

    def test_unknown_pattern_returns_empty(self):
        matcher = FakeSmartsMatcher()
        pat = SymbolicPattern("[#8]", arity=1)
        out = matcher.match(mol=object(), pattern=pat)
        assert out.shape == (1, 0)
        assert out.dtype == torch.long

    def test_arity_mismatch_raises(self):
        matcher = FakeSmartsMatcher({"[#6]-[#6]": torch.tensor([[0], [1], [2]], dtype=torch.long)})
        pat = SymbolicPattern("[#6]-[#6]", arity=2)
        with pytest.raises(ValueError, match="arity"):
            matcher.match(None, pat)


class TestMolpySmartsMatcher:
    def test_matches_the_carbons_of_ethanol(self):
        import molpy as mp

        mol = mp.io.smiles.read_smiles("CCO").to_atomistic()
        out = MolpySmartsMatcher().match(mol, SymbolicPattern("[#6]", arity=1))
        assert out.dtype == torch.long
        assert out.shape == (1, 2)  # heavy-atom graph: C0, C1
