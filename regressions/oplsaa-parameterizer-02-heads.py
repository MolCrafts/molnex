"""Public-API regression for ProperTorsionParamHead form=opls (k1..k4).

Spec: ``oplsaa-parameterizer-02-heads``.

Hard-coded goldens only — no third-party oracle. Pins:

1. Zero-init MLP → ``k1=k2=k3=k4=0`` (atol 1e-12): outputs are LAMMPS
   ``Kn``, not softplus-scaled or Class-I half-amplitudes ``k_m = Kn/2``.
2. Random features → each ``k1..k4`` has shape ``(N,)``.
3. Sum of ``k1..k4`` is differentiable wrt features.

Provenance
----------
    formula          : identity MLP (all weights/biases zero) → Kn = 0
    tool             : hand / zero linear layer (no external FF)
    capture command  : PYTHONPATH=src python regressions/oplsaa-parameterizer-02-heads.py
    date             : 2026-08-20
    tolerance        : atol 1e-12 on zero-init Kn
"""

from __future__ import annotations

import sys

import torch

from molpot.composition.mm_heads import ProperTorsionParamHead

_ZERO_ATOL = 1e-12
_N = 4
_D = 8


def main() -> int:
    head = ProperTorsionParamHead(feature_dim=_D, hidden_dim=16, form="opls")
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()

    features = torch.tensor(
        [
            [1.0, -0.5, 2.0, 0.0, 3.0, -1.0, 0.25, 4.0],
            [-2.0, 1.0, 0.5, -3.0, 0.0, 2.0, -0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ]
    )
    out = head(features)
    if set(out) != {"k1", "k2", "k3", "k4"}:
        raise AssertionError(f"unexpected keys {set(out)}")
    zeros = torch.zeros(_N, dtype=features.dtype)
    for key in ("k1", "k2", "k3", "k4"):
        if out[key].shape != (_N,):
            raise AssertionError(f"{key} shape {tuple(out[key].shape)}, want ({_N},)")
        if not torch.allclose(out[key], zeros, atol=_ZERO_ATOL):
            raise AssertionError(f"{key} zero-init golden failed: {out[key]}")

    torch.manual_seed(0)
    feats = torch.randn(_N, _D, requires_grad=True)
    head2 = ProperTorsionParamHead(feature_dim=_D, hidden_dim=16, form="opls")
    out2 = head2(feats)
    for key in ("k1", "k2", "k3", "k4"):
        if out2[key].shape != (_N,):
            raise AssertionError(f"{key} shape {tuple(out2[key].shape)}")
    loss = out2["k1"].sum() + out2["k2"].sum() + out2["k3"].sum() + out2["k4"].sum()
    loss.backward()
    if feats.grad is None or not torch.isfinite(feats.grad).all():
        raise AssertionError("features.grad missing or non-finite")
    if feats.grad.shape != (_N, _D):
        raise AssertionError(f"grad shape {tuple(feats.grad.shape)}")

    print("oplsaa-parameterizer-02-heads ProperTorsionParamHead form=opls OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
