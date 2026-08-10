"""Public-API regression for symbolic SMARTS perception + SymbolicForceField.

Spec: `learnable-classical-ff-07-smarts`.

Hard-coded goldens only — no third-party oracle. Pins:

1. SymbolicPattern validates non-empty pattern + arity in {1,2,3,4}
2. FakeSmartsMatcher returns configured [arity, K] hits
3. ClassPatternRegistry bind/get + conflict detection + reverse lookup
4. SymbolicForceField.records pairs prototypes with SMARTS
5. match_molecule assigns type ids via FakeSmartsMatcher (no energy)
6. SmartsMatcher Protocol satisfied by Fake (and Molpy when available)
7. molrep.perception has zero molrs / molpot imports

Provenance
----------
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-07-smarts.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = ROOT / "src" / "molrep" / "perception"


def _assert_no_forbidden_imports(package_dir: Path, forbidden: tuple[str, ...]) -> None:
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden:
                        assert not alias.name.startswith(bad), (
                            f"{path.name} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for bad in forbidden:
                    assert not mod.startswith(bad), f"{path.name} imports from {mod}"


def main() -> int:
    from molrep.condensation import InteractionClass, TypeRecord, TypeSystem
    from molrep.perception import (
        ClassPatternRegistry,
        DiscreteClassRecord,
        FakeSmartsMatcher,
        SmartsMatcher,
        SymbolicForceField,
        SymbolicPattern,
    )

    # --- 1. SymbolicPattern validation ---
    ok = SymbolicPattern("[#6]-[#6]", arity=2)
    assert ok.arity == 2
    try:
        SymbolicPattern("", arity=2)
        raise AssertionError("empty pattern should raise")
    except ValueError:
        pass
    try:
        SymbolicPattern("[#6]", arity=0)
        raise AssertionError("arity 0 should raise")
    except ValueError:
        pass
    try:
        SymbolicPattern("[#6]", arity=5)
        raise AssertionError("arity 5 should raise")
    except ValueError:
        pass

    # --- 2. FakeSmartsMatcher ---
    hits = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    matcher = FakeSmartsMatcher({"[#6]-[#6]": hits})
    assert isinstance(matcher, SmartsMatcher)
    out = matcher.match(None, ok)
    assert out.shape == (2, 2)
    assert torch.equal(out, hits)
    empty = matcher.match(None, SymbolicPattern("[#8]", arity=1))
    assert empty.shape == (1, 0)

    # --- 3. Registry ---
    reg = ClassPatternRegistry()
    pat_cc = SymbolicPattern("[#6]-[#6]", arity=2)
    pat_co = SymbolicPattern("[#6]-[#8]", arity=2)
    reg.bind(InteractionClass.BOND, 0, pat_cc)
    reg.bind(InteractionClass.BOND, 1, pat_co)
    assert reg.get(InteractionClass.BOND, 0).pattern == "[#6]-[#6]"
    assert reg.reverse_lookup("[#6]-[#8]") == (InteractionClass.BOND, 1)
    try:
        reg.bind(InteractionClass.BOND, 0, SymbolicPattern("[#7]-[#7]", arity=2))
        raise AssertionError("conflicting bind should raise")
    except ValueError:
        pass

    # --- 4–5. SymbolicForceField ---
    ts = TypeSystem(
        InteractionClass.BOND,
        [
            TypeRecord(type_id=0, prototype={"k": 300.0, "r0": 1.09}, label="CT-CT"),
            TypeRecord(type_id=1, prototype={"k": 320.0, "r0": 1.41}, label="CT-OH"),
        ],
    )
    ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)
    recs = ff.records()
    assert len(recs) == 2
    assert all(isinstance(r, DiscreteClassRecord) for r in recs)
    by_id = {r.type_id: r for r in recs}
    assert by_id[0].smarts == "[#6]-[#6]"
    assert by_id[0].prototype == {"k": 300.0, "r0": 1.09}
    assert by_id[1].smarts == "[#6]-[#8]"

    fake = FakeSmartsMatcher(
        {
            "[#6]-[#6]": torch.tensor([[0], [1]], dtype=torch.long),
            "[#6]-[#8]": torch.tensor([[1], [2]], dtype=torch.long),
        }
    )
    assigned = ff.match_molecule(mol=object(), matcher=fake)
    bond = assigned[InteractionClass.BOND]
    assert bond["matches"].shape == (2, 2)
    assert bond["type_ids"].tolist() == [0, 1]
    assert bond["matches"][:, 0].tolist() == [0, 1]
    assert bond["matches"][:, 1].tolist() == [1, 2]

    # --- 6. Molpy matcher protocol (optional smoke) ---
    try:
        from molrep.perception import MolpySmartsMatcher

        molpy_matcher = MolpySmartsMatcher()
        assert isinstance(molpy_matcher, SmartsMatcher)
    except ImportError:
        pass

    # --- 7. Import hard rules ---
    _assert_no_forbidden_imports(PERCEPTION, ("molrs", "molpot"))

    print("learnable-classical-ff-07-smarts: all hard-coded goldens OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
