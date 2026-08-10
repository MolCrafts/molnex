"""SmartsMatcher Protocol + Fake / Molpy matchers (learnable-classical-ff-07)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from molrep.perception import (
    FakeSmartsMatcher,
    MolpySmartsMatcher,
    SmartsMatcher,
    SymbolicPattern,
)

PERCEPTION_ROOT = Path(__file__).resolve().parents[3] / "src" / "molrep" / "perception"


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

    def test_satisfies_protocol(self):
        matcher = FakeSmartsMatcher()
        assert isinstance(matcher, SmartsMatcher)


class TestMolpySmartsMatcher:
    def test_satisfies_protocol(self):
        try:
            matcher = MolpySmartsMatcher()
        except ImportError:
            pytest.skip("molpy.SmartsPattern unavailable")
        assert isinstance(matcher, SmartsMatcher)

    def test_import_path_is_molpy_only(self):
        """Source of matcher.py must not reference bare molrs imports."""
        src = (PERCEPTION_ROOT / "matcher.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molrs"), alias.name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith("molrs"), mod
        assert "from molpy" in src or "import molpy" in src

    def test_functional_match_if_available(self):
        """Optional smoke: match carbon atoms in ethanol when molpy works."""
        try:
            import molpy as mp
            from molpy import SmartsPattern  # noqa: F401

            matcher = MolpySmartsMatcher()
            mol = mp.io.read_smiles("CCO")
        except Exception as exc:  # pragma: no cover - env-dependent
            pytest.skip(f"molpy SMARTS smoke unavailable: {exc}")

        pat = SymbolicPattern("[#6]", arity=1)
        out = matcher.match(mol, pat)
        assert out.ndim == 2
        assert out.shape[0] == 1
        assert out.shape[1] >= 1  # ethanol has ≥1 carbon (heavy-only graph)
        assert out.dtype == torch.long


class TestPerceptionMolrsForbidden:
    def test_no_molrs_imports_in_package(self):
        py_files = sorted(PERCEPTION_ROOT.glob("*.py"))
        assert py_files, "perception package missing"
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molrs"), (
                            f"{path.name} imports {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert not mod.startswith("molrs"), f"{path.name} imports from {mod}"
