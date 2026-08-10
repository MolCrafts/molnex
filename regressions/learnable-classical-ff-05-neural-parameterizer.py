"""Public-API regression for ClassicalMMParameterizer (encoder → IR → E/F).

Spec: `learnable-classical-ff-05-neural-parameterizer`.

Hard-coded goldens only — no third-party oracle. Pins:

1. FakeEncoder satisfies Protocol; no molzoo import in parameterizer
2. parameterize → CLASS_I IR (kcal/mol)
3. Bond harmonic energy golden: k=2, r=1.5, r0=1 → E = 0.25 kcal/mol
4. compute_forces=True → forces (N,3) matching analytic F = -∂E/∂r
5. Optional eV conversion at boundary only (IR stays class_i_canonical)

Provenance
----------
    formula          : E = ½ k (r − r₀)²  (OpenMM §19 / Class-I)
    geometry         : two atoms on x-axis at distance r
    capture command  : PYTHONPATH=src python \
                      regressions/learnable-classical-ff-05-neural-parameterizer.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict


class _FakeEmbeddings:
    def __init__(self, features: Mapping[str, torch.Tensor]) -> None:
        self._features = dict(features)

    def interaction_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._features)


class FakeEncoder(nn.Module):
    def __init__(self, features: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = dict(features)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> _FakeEmbeddings:
        return _FakeEmbeddings(self._features)


class _ConstBond(nn.Module):
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), 2.0, dtype=features.dtype),
            "r0": torch.full((n,), 1.0, dtype=features.dtype),
        }


def main() -> int:
    from molpot import (
        KCAL_MOL_TO_EV,
        ClassicalMMComposer,
        ClassicalMMParameterizer,
        PotentialIR,
        energy_kcal_to_ev,
    )
    from molpot.composition.parameterizer import ChemEncoderProtocol
    from molpot.ir import CLASS_I_CANONICAL

    torch.manual_seed(0)

    # --- 1. Protocol + import boundary ---
    features = {"bonds": torch.zeros(1, 4, dtype=torch.float64)}
    encoder = FakeEncoder(features)
    assert isinstance(encoder, ChemEncoderProtocol)

    path = Path(__file__).resolve().parents[1] / "src/molpot/composition/parameterizer.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("molzoo")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("molzoo")

    # --- 2. Parameterize IR ---
    composer = ClassicalMMComposer(bond_head=_ConstBond())
    param = ClassicalMMParameterizer(encoder=encoder, composer=composer)
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
    ir = param.parameterize(batch)
    assert isinstance(ir, PotentialIR)
    assert ir.unit_system == "class_i_canonical"
    assert CLASS_I_CANONICAL["energy"] == "kcal/mol"

    # --- 3. Energy golden ---
    e = float(param.energy(batch, ir=ir, pos=pos))
    assert math.isclose(e, 0.25, abs_tol=1e-8), f"bond golden E={e}"

    # --- 4. Forces ---
    out = param.forward(batch, compute_forces=True)
    forces = out["forces"]
    assert forces.shape == (2, 3)
    # F0_x = +k*(r-r0) = 1.0, F1_x = -1.0
    assert math.isclose(float(forces[0, 0]), 1.0, abs_tol=1e-5)
    assert math.isclose(float(forces[1, 0]), -1.0, abs_tol=1e-5)
    assert "energy" in batch["graphs"]
    assert "forces" in batch["atoms"]

    # --- 5. Units boundary ---
    e_ev = float(energy_kcal_to_ev(torch.tensor(e, dtype=torch.float64)))
    assert math.isclose(e_ev, e * KCAL_MOL_TO_EV, abs_tol=1e-9)
    assert ir.unit_system == "class_i_canonical"

    print("learnable-classical-ff-05-neural-parameterizer: all hard-coded goldens OK")
    print(f"  bond harmonic E = {e} kcal/mol")
    print(f"  F0_x = {float(forces[0, 0])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
