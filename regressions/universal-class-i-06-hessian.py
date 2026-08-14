"""Public-API regression for the Cartesian Hessian producer and its consumer.

Spec: ``universal-class-i-06-hessian``.

Equilibrium harmonic dimer ``E = ½k(r − r₀)²`` with ``k = 2.0`` energy/Å²,
``r₀ = 1.5`` Å, atoms at ``(0,0,0)`` and ``(1.5,0,0)`` Å, float64.

All goldens below are **analytic** — the Wilson closed form

    ∂²E/∂d∂d = k u uᵀ + k (r − r₀)/r · (I − u uᵀ),   H₁₁ = H₂₂ = −H₁₂

hand-derived on 2026-08-14; at ``r = r₀`` the transverse term vanishes and
``H = k BᵀB`` with the Wilson stretch row ``B = (−u, u)``. No third-party
oracle is imported, subprocessed, or consulted at runtime.

    H (6×6, x0,y0,z0,x1,y1,z1): H[0,0] = H[3,3] = 2.0,
                                H[0,3] = H[3,0] = −2.0, all others 0
    H_q = project(H, B) = [[2.0]]        (B = [[−1,0,0,1,0,0]])
    ProjectedInternalHessianLoss vs diagonal [2.0] = 0

    capture: PYTHONPATH=src python regressions/universal-class-i-06-hessian.py
    date: 2026-08-14
"""

from __future__ import annotations

import sys

import torch

from molix.core.losses import ProjectedInternalHessianLoss
from molpot import BondHarmonic, HessianDerivation

_K = 2.0
_R0 = 1.5

_POS = torch.tensor([[0.0, 0.0, 0.0], [_R0, 0.0, 0.0]], dtype=torch.float64)

_HESSIAN_GOLDEN = torch.tensor(
    [
        [2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)

_WILSON_B = torch.tensor([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)


def main() -> int:
    bond = BondHarmonic(
        k=torch.tensor([_K], dtype=torch.float64),
        r0=torch.tensor([_R0], dtype=torch.float64),
    )
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)  # COO [2, num_bonds]
    bond_types = torch.zeros(1, dtype=torch.long)

    def energy_fn(pos: torch.Tensor) -> torch.Tensor:
        return bond(pos=pos, bond_index=bond_index, bond_types=bond_types)

    hessian = HessianDerivation().forward(energy_fn, _POS)
    if tuple(hessian.shape) != (6, 6):
        raise AssertionError(f"hessian must be (6, 6); got {tuple(hessian.shape)}")
    if not torch.allclose(hessian, _HESSIAN_GOLDEN, atol=1e-10, rtol=0.0):
        raise AssertionError(f"Cartesian Hessian drift:\n{hessian}")

    h_q = ProjectedInternalHessianLoss.project(hessian, _WILSON_B)
    if not torch.allclose(h_q, torch.tensor([[2.0]], dtype=torch.float64), atol=1e-6):
        raise AssertionError(f"H_q={h_q}")

    loss = ProjectedInternalHessianLoss()
    zero = loss(
        {"hessian": hessian},
        {"B": _WILSON_B, "diagonal": torch.tensor([_K], dtype=torch.float64)},
    )
    # ``hessian`` still carries its autograd graph (grad is enabled here, which is
    # what lets a Hessian loss reach a parameter head) — detach for the scalar read.
    if float(zero.detach()) > 1e-12:
        raise AssertionError(f"diagonal MSE vs k=2 should be 0, got {zero}")

    print("universal-class-i-06-hessian: dimer H = k B^T B, H_q = 2.0, diagonal MSE = 0 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
