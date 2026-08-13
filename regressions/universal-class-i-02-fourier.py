"""Public-API regression for Cartesian proper-torsion (a_n, b_n).

Spec: ``universal-class-i-02-fourier`` (ac-008).

Hard-coded goldens (OpenMM §19.4 identity + math.atan2, 2026-08-13):

* ``(k,γ)=(2,0) → (a,b)=(2.0, 0.0)``
* ``(k,γ)=(1.5, π) → (a,b)=(-1.5, 0.0)``
* ``(a,b)=(0.6, 0.8) → (k,γ)=(1.0, 0.9272952180016123)`` at atol 1e-12
* ``project`` of ``(a,b)=(0, 0.4)`` n=1 raises ``ValueError``

    capture: PYTHONPATH=src python regressions/universal-class-i-02-fourier.py
    date: 2026-08-13
"""

from __future__ import annotations

import math
import sys

import torch

from molpot.forcefield import ClassicalIMM, ClassIForceField, OPLSForceField
from molpot.ir.bags import ProperTorsionBag


def main() -> int:
    bag = ProperTorsionBag.from_cartesian(
        a=torch.tensor([[2.0], [-1.5], [0.6]], dtype=torch.float64),
        b=torch.tensor([[0.0], [0.0], [0.8]], dtype=torch.float64),
        periodicity=torch.tensor([1], dtype=torch.long),
    )
    k_ref = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float64)
    g_ref = torch.tensor([0.0, math.pi, 0.9272952180016123], dtype=torch.float64)
    if not torch.allclose(bag.k.reshape(-1), k_ref, atol=1e-12):
        raise AssertionError(f"k mismatch: {bag.k}")
    if not torch.allclose(bag.phase.reshape(-1), g_ref, atol=1e-12):
        raise AssertionError(f"phase mismatch: {bag.phase}")

    source = ClassIForceField().with_params(
        bonds={"k": torch.ones(1), "r0": torch.ones(1)},
        angles={"k": torch.ones(1), "theta0": torch.ones(1)},
        propers={
            "a": torch.zeros(1, 1),
            "b": torch.full((1, 1), 0.4),
            "periodicity": torch.tensor([1], dtype=torch.long),
            "idivf": torch.ones(1),
        },
    )
    model = ClassicalIMM(torch.nn.Identity(), forcefield=ClassIForceField())
    try:
        model.project(OPLSForceField(), source=source)
    except ValueError as exc:
        msg = str(exc).lower()
        if "sin" not in msg and "phase" not in msg:
            raise AssertionError(f"unexpected project error: {exc}") from exc
    else:
        raise AssertionError("project should fail on sine residual")
    print("universal-class-i-02-fourier: mapping goldens + sine fail-loud OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
