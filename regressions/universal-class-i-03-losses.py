"""Public-API regression for Universal Class-I multi-task losses.

Spec: ``universal-class-i-03-losses`` (ac-008).

Wilson GF toy (2026-08-13, analytic, no third-party):

    B = [[-1, 0, 0, 1, 0, 0]]
    H_x = k BᵀB, k=2.0 → H_q = [[2.0]] (atol 1e-6)
    ScaleNormalizedLoss(inner=4, s=2) → 1.0

    capture: PYTHONPATH=src python regressions/universal-class-i-03-losses.py
"""

from __future__ import annotations

import sys

import torch

from molix.core.losses import ProjectedInternalHessianLoss, ScaleNormalizedLoss


class _Const(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, pred, target):
        return pred.new_tensor(self.value)


def main() -> int:
    b = torch.tensor([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    k = 2.0
    h_x = k * (b.T @ b)
    h_q = ProjectedInternalHessianLoss.project(h_x, b)
    if not torch.allclose(h_q, torch.tensor([[2.0]], dtype=torch.float64), atol=1e-6):
        raise AssertionError(f"H_q={h_q}")
    loss = ProjectedInternalHessianLoss()
    zero = loss(
        {"hessian": h_x},
        {"B": b, "diagonal": torch.tensor([2.0], dtype=torch.float64)},
    )
    if float(zero) > 1e-12:
        raise AssertionError(f"diagonal MSE vs k=2 should be 0, got {zero}")
    scaled = ScaleNormalizedLoss(_Const(4.0), scale=2.0)
    out = scaled(torch.zeros(1), torch.zeros(1))
    if not torch.allclose(out, torch.tensor(1.0), atol=1e-6):
        raise AssertionError(f"scale golden {out}")
    print("universal-class-i-03-losses: Wilson H_q=2.0 and L/s²=1.0 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
