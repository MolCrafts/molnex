"""Public-API regression for Potential IR → OpenMM force-spec export.

Spec: `learnable-classical-ff-08-ff-export`.

Hard-coded goldens only — no live OpenMM / third-party oracle. Pins:

1. Bond force constant: k=100 kcal mol⁻¹ Å⁻² → 41840 kJ mol⁻¹ nm⁻²
2. AMBER torsion Vn=2 kcal/mol → OpenMM PeriodicTorsion k=4.184 kJ/mol
3. Class-I NonbondedScaling 1–4 defaults flow into ForceSpec.scaling
4. Unsupported improper_harmonic raises UnsupportedTermError (no silent drop)
5. ForceSpec.to_dict is JSON-serializable

Provenance
----------
    formula          : OpenMM User Guide §19; k_omm_bond = k_ir * 4.184 / 0.01;
                       AMBER E=(Vn/2)[1+cos] → OpenMM k=(Vn/2)*4.184
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-08-ff-export.py
    date             : 2026-08-10
    note             : Pure unit/form translation; no dynamics / energy compare.
"""

from __future__ import annotations

import json
import math
import sys

import torch


def main() -> int:
    from molix.ff_export import (
        ForceFieldCompiler,
        TranslationCase,
        UnsupportedTermError,
        scale_amber_vn,
        scale_bond_k,
    )
    from molpot.ir import (
        BondBag,
        ChargeBag,
        ImproperHarmonicBag,
        LJBag,
        NonbondedScaling,
        PotentialIR,
        ProperTorsionBag,
    )

    # --- 1. Bond k golden ---
    assert scale_bond_k(100.0) == 41840.0, f"bond k golden: {scale_bond_k(100.0)}"

    # --- 2. Torsion Vn=2 → k_omm=4.184 ---
    assert scale_amber_vn(2.0) == 4.184, f"Vn golden: {scale_amber_vn(2.0)}"

    ir = PotentialIR(
        bonds=BondBag(
            k=torch.tensor([100.0], dtype=torch.float64),
            r0=torch.tensor([1.5], dtype=torch.float64),
        ),
        propers=ProperTorsionBag(
            # IR half-barrier for AMBER Vn=2 (E=(k/s)[1+cos], s=1, k=1)
            k=torch.tensor([[1.0]], dtype=torch.float64),
            periodicity=torch.tensor([2], dtype=torch.long),
            phase=torch.tensor([[0.0]], dtype=torch.float64),
            idivf=torch.tensor([1.0], dtype=torch.float64),
        ),
        lj=LJBag(
            epsilon=torch.tensor([0.1], dtype=torch.float64),
            sigma=torch.tensor([3.5], dtype=torch.float64),
        ),
        charges=ChargeBag(q=torch.tensor([0.5, -0.5], dtype=torch.float64)),
        scaling=NonbondedScaling(),
    )

    compiler = ForceFieldCompiler("openmm")
    spec = compiler.compile(ir)

    bond = next(f for f in spec.forces if f["type"] == "HarmonicBondForce")
    assert bond["parameters"][0]["k"] == 41840.0
    assert math.isclose(bond["parameters"][0]["r0"], 0.15, abs_tol=1e-15)
    assert bond["case"] == TranslationCase.DIRECT_UNIT_SCALE.value

    torsion = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
    assert torsion["parameters"][0]["k"] == 4.184
    assert torsion["parameters"][0]["periodicity"] == 2

    # --- 3. 1–4 scales ---
    assert spec.scaling is not None
    assert math.isclose(spec.scaling["scale_q_14"], 5.0 / 6.0, abs_tol=1e-15)
    assert math.isclose(spec.scaling["scale_lj_14"], 0.5, abs_tol=1e-15)

    # --- 4. Unsupported improper harmonic ---
    bad = PotentialIR(
        impropers_harmonic=ImproperHarmonicBag(
            k=torch.tensor([1.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
    )
    try:
        compiler.compile(bad)
    except UnsupportedTermError as err:
        assert err.term == "improper_harmonic"
        assert err.case is TranslationCase.UNSUPPORTED
    else:
        raise AssertionError("expected UnsupportedTermError for improper_harmonic")

    # --- 5. JSON serializable ---
    payload = json.dumps(spec.to_dict())
    assert "HarmonicBondForce" in payload
    assert "41840" in payload

    print("OK learnable-classical-ff-08-ff-export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
