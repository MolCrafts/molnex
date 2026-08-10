"""Public-API regression for Class-I Potential IR + torsion kernels.

Spec: `learnable-classical-ff-01-ir-kernels`.

Hard-coded goldens only — no OpenMM / third-party oracle. Pins:

1. CLASS_I_CANONICAL units (kcal/mol, angstrom, e, radian)
2. NonbondedScaling 1-2/1-3/1-4 defaults (AMBER/GAFF Class-I)
3. ProperTorsionPeriodic cis identity:
       E = (k/s)[1 + cos(n*phi - gamma)]
       k=1, s=1, n=1, gamma=0, phi=0 → E = 2.0 kcal/mol
4. Multi-term sum at cis: k=(1, 0.5), n=(1, 2) → E = 3.0
5. ImproperHarmonic: k=2, chi=pi/6, chi0=0 → E = (pi/6)^2
6. BondHarmonic sanity reuse: k=2, r=1.5, r0=1 → E = 0.25

Provenance
----------
    formula          : OpenMM User Guide §19.4 / SMIRNOFF proper torsion;
                       E = (k/idivf)[1 + cos(n*phi - gamma)]
    geometry         : analytic cis fixture
                       i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,1,0) → phi=0
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-01-ir-kernels.py
    date             : 2026-08-10
    note             : Spec domain-basis text said k=2 → E=2.0, which contradicts
                       the written formula (2k/s = 4 for k=2). Goldens follow the
                       formula with k=1 → E=2.0 at cis.
"""

from __future__ import annotations

import math
import sys

import torch


def main() -> int:
    from molpot.ir import CLASS_I_CANONICAL, NonbondedScaling, PotentialIR
    from molpot.potentials import (
        BondHarmonic,
        ImproperHarmonic,
        ImproperPeriodic,
        ProperTorsionPeriodic,
    )

    # --- 1. Units ---
    assert CLASS_I_CANONICAL["energy"] == "kcal/mol"
    assert CLASS_I_CANONICAL["length"] in ("angstrom", "Å", "A")
    assert CLASS_I_CANONICAL["charge"] == "e"
    assert CLASS_I_CANONICAL["angle"] in ("radian", "rad")

    # --- 2. Nonbonded scaling defaults ---
    scaling = NonbondedScaling()
    assert float(scaling.scale_q_12) == 0.0
    assert float(scaling.scale_q_13) == 0.0
    assert math.isclose(float(scaling.scale_q_14), 5.0 / 6.0, abs_tol=1e-12)
    assert float(scaling.scale_lj_12) == 0.0
    assert float(scaling.scale_lj_13) == 0.0
    assert math.isclose(float(scaling.scale_lj_14), 0.5, abs_tol=1e-12)

    # Empty IR is valid
    _ = PotentialIR()

    # --- Geometry: cis phi=0 ---
    cis = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    # proper torsion i-j-k-l on atoms 0-1-2-3
    proper_idx = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    # improper molrs center-first [center, i, j, k] with center = atom 1
    improper_idx = torch.tensor([[1], [0], [2], [3]], dtype=torch.long)
    types0 = torch.tensor([0], dtype=torch.long)

    # --- 3. Proper torsion cis golden E=2.0 (k=1) ---
    proper = ProperTorsionPeriodic(
        k=torch.tensor([[1.0]], dtype=torch.float64),
        periodicity=torch.tensor([1], dtype=torch.long),
        phase=torch.tensor([[0.0]], dtype=torch.float64),
        idivf=torch.tensor([1.0], dtype=torch.float64),
    )
    e_proper = float(proper(pos=cis, proper_index=proper_idx, proper_types=types0))
    assert math.isclose(e_proper, 2.0, abs_tol=1e-10), f"proper cis E={e_proper}"

    # --- 4. Multi-term sum ---
    proper_mt = ProperTorsionPeriodic(
        k=torch.tensor([[1.0, 0.5]], dtype=torch.float64),
        periodicity=torch.tensor([1, 2], dtype=torch.long),
        phase=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        idivf=torch.tensor([1.0], dtype=torch.float64),
    )
    e_mt = float(proper_mt(pos=cis, proper_index=proper_idx, proper_types=types0))
    assert math.isclose(e_mt, 3.0, abs_tol=1e-10), f"multi-term E={e_mt}"

    # --- Improper periodic same cis golden (center-first layout) ---
    improper = ImproperPeriodic(
        k=torch.tensor([[1.0]], dtype=torch.float64),
        periodicity=torch.tensor([1], dtype=torch.long),
        phase=torch.tensor([[0.0]], dtype=torch.float64),
        idivf=torch.tensor([1.0], dtype=torch.float64),
    )
    e_imp = float(improper(pos=cis, improper_index=improper_idx, improper_types=types0))
    assert math.isclose(e_imp, 2.0, abs_tol=1e-10), f"improper cis E={e_imp}"

    # --- 5. Improper harmonic chi=pi/6 ---
    pi6_pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)],
        ],
        dtype=torch.float64,
    )
    imp_h = ImproperHarmonic(
        k=torch.tensor([2.0], dtype=torch.float64),
        chi0=torch.tensor([0.0], dtype=torch.float64),
    )
    e_ih = float(imp_h(pos=pi6_pos, improper_index=improper_idx, improper_types=types0))
    expected_ih = (math.pi / 6.0) ** 2
    assert math.isclose(e_ih, expected_ih, abs_tol=1e-8), f"improper harmonic E={e_ih}"

    # --- 6. BondHarmonic reuse sanity ---
    bond = BondHarmonic(k=torch.tensor([2.0]), r0=torch.tensor([1.0]))
    e_bond = float(
        bond(
            pos=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            bond_index=torch.tensor([[0], [1]], dtype=torch.long),
            bond_types=torch.tensor([0], dtype=torch.long),
        )
    )
    # 0.5 * 2.0 * (1.5 - 1.0)^2 = 0.25
    assert math.isclose(e_bond, 0.25, abs_tol=1e-10), f"bond E={e_bond}"

    print("learnable-classical-ff-01-ir-kernels: all hard-coded goldens OK")
    print(f"  proper cis E          = {e_proper}")
    print(f"  proper multi-term E   = {e_mt}")
    print(f"  improper cis E        = {e_imp}")
    print(f"  improper harmonic E   = {e_ih}")
    print(f"  bond harmonic E       = {e_bond}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
