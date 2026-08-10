"""Public-API regression for continuous MM heads + ClassicalMMComposer.

Spec: `learnable-classical-ff-03-mm-heads`.

Hard-coded goldens only — no third-party oracle. Pins:

1. BondParamHead / AngleParamHead positivity + theta0 ∈ (0, π)
2. ProperTorsionParamHead multi-term shapes (k ≥ 0)
3. MultiHead(LJ + Charge) merge + neutrality
4. ClassicalMMComposer.parameterize → CLASS_I PotentialIR
5. Bond harmonic energy golden: k=2, r=1.5, r0=1 → E = 0.25 kcal/mol
6. No molzoo / molrep.chem imports in classical_mm / mm_heads

Provenance
----------
    formula          : E = ½ k (r − r₀)²  (OpenMM §19 / Class-I)
    geometry         : two atoms on x-axis at distance r
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-03-mm-heads.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import torch
from tensordict import TensorDict


def main() -> int:
    from molpot import (
        AngleParamHead,
        BondParamHead,
        ChargeHead,
        ClassicalMMComposer,
        ImproperParamHead,
        LJParameterHead,
        MultiHead,
        PotentialIR,
        ProperTorsionParamHead,
    )
    from molpot.ir import CLASS_I_CANONICAL

    # --- 1. Head positivity ---
    bond_head = BondParamHead(feature_dim=8, hidden_dim=16)
    bout = bond_head(torch.randn(4, 8))
    assert bout["k"].shape == (4,) and torch.all(bout["k"] > 0)
    assert bout["r0"].shape == (4,) and torch.all(bout["r0"] > 0)

    angle_head = AngleParamHead(feature_dim=8, hidden_dim=16)
    aout = angle_head(torch.randn(3, 8))
    assert torch.all(aout["k"] > 0)
    assert torch.all(aout["theta0"] > 0) and torch.all(aout["theta0"] < math.pi)

    proper_head = ProperTorsionParamHead(
        feature_dim=8, n_terms=2, periodicity=(1, 2)
    )
    pout = proper_head(torch.randn(2, 8))
    assert pout["k"].shape == (2, 2) and torch.all(pout["k"] >= 0)
    assert pout["phase"].shape == (2, 2)

    _ = ImproperParamHead(feature_dim=8, include_harmonic=True, include_periodic=False)

    # --- 2. MultiHead reuse ---
    multi = MultiHead(
        {
            "lj": LJParameterHead(feature_dim=8, hidden_dim=16),
            "q": ChargeHead(feature_dim=8, hidden_dim=16, total_charge=0.0),
        }
    )
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mout = multi(torch.randn(4, 8), batch=batch_idx)
    assert set(mout) == {"epsilon", "sigma", "charge"}
    assert mout["charge"][:2].sum().abs() < 1e-5

    # --- 3. Composer IR + golden energy ---
    class _ConstBond(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            n = features.shape[0]
            return {
                "k": torch.full((n,), 2.0, dtype=features.dtype),
                "r0": torch.full((n,), 1.0, dtype=features.dtype),
            }

    composer = ClassicalMMComposer(bond_head=_ConstBond())
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1]),
                    "batch": torch.zeros(2, dtype=torch.long),
                },
                batch_size=[2],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    features = {"bonds": torch.zeros(1, 1, dtype=torch.float64)}
    ir = composer.parameterize(features, batch)
    assert isinstance(ir, PotentialIR)
    assert ir.unit_system == "class_i_canonical"
    assert CLASS_I_CANONICAL["energy"] == "kcal/mol"
    e = float(composer.energy(ir, batch, pos=pos))
    # 0.5 * 2.0 * (1.5 - 1.0)^2 = 0.25
    assert math.isclose(e, 0.25, abs_tol=1e-8), f"bond golden E={e}"

    # --- 4. Import boundary ---
    root = Path(__file__).resolve().parents[1] / "src/molpot/composition"
    for name in ("classical_mm.py", "mm_heads.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molzoo")
                    assert not alias.name.startswith("molrep.chem")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molzoo")
                assert not node.module.startswith("molrep.chem")

    print("learnable-classical-ff-03-mm-heads: all hard-coded goldens OK")
    print(f"  bond harmonic E = {e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
