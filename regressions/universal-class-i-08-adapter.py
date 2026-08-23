"""Public-API regression for ForceFieldPredictions (model → task predictions).

Spec: ``universal-class-i-08-adapter``.

One :class:`~molpot.Parameterizer` over a two-atom bonded topology (constant
``k`` / ``r0`` head plus a constant χ/η charge path) feeds two adapters built
from the *same* model: one declaring ``("energy", "forces", "hessian")``, one
declaring ``("q",)``. The second must return charges and **no** ``energy`` key —
the anti-façade contract, checked here through the public API only.

All goldens below are **analytic**, hand-derived on 2026-08-14; no third-party
oracle is imported, subprocessed, or consulted at runtime.

Harmonic bond ``E = ½k(r − r₀)²`` with ``k = 100`` kcal mol⁻¹ Å⁻²,
``r₀ = 1.0`` Å, atoms at ``(0,0,0)`` and ``(1.5,0,0)`` Å (``r = 1.5`` Å,
``û = x̂``), float64::

    E   = ½·100·(0.5)² = 12.5 kcal/mol
    F   = −∂E/∂x,  ∂E/∂x₁ = k(r−r₀) û  ⇒  F₀ = (+50,0,0), F₁ = (−50,0,0)
                                             kcal mol⁻¹ Å⁻¹
    H   = k ûûᵀ ⊕ (k(r−r₀)/r)(I − ûûᵀ)   (Wilson 1941, doi:10.1063/1.1750829)
          H[0,0] = k              = 100.0
          H[0,3] = −k             = −100.0
          H[1,1] = k(r−r₀)/r      = 100·0.5/1.5 = 33.333333333333336
                                             kcal mol⁻¹ Å⁻²
    and ``H = Hᵀ`` because it is a second derivative of one scalar energy.

The ForceField declares a bond style only, so the assembled form has no
LJ and no Coulomb term: the energy stays exactly 12.5 kcal/mol even though the
charge path wrote ``atoms.q``.

EEM/QEq KKT solve (``DiagonalHardnessKernel``, ``jitter = 0``), χ = (0, 1),
η = (2, 2), total charge Q = 0::

    [diag(η)  1] [q]   [−χ]
    [  1ᵀ     0] [λ] = [ Q]      ⇒  q_i = −(χ_i + λ)/η_i,  Σ_i q_i = Q
    −(0+λ)/2 − (1+λ)/2 = 0       ⇒  λ = −0.5
    q = (+0.25, −0.25) e         (the more electronegative atom is negative)

With a diagonal hardness kernel the solve reads no positions (``dq/dx = 0``),
which is what makes the fixed-parameter force above exact.

    capture: PYTHONPATH=src python regressions/universal-class-i-08-adapter.py
    date: 2026-08-14
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot import (
    AtomicChargeModel,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
    ForceField,
    ForceFieldPredictions,
    HessianDerivation,
    Parameterizer,
)

_DTYPE = torch.float64

#: fp64 budget for the analytic goldens (spec ac-011).
_ATOL = 1e-10

_BOND_K = 100.0  # kcal mol⁻¹ Å⁻²
_BOND_R0 = 1.0  # Å
_BOND_R = 1.5  # Å

_ENERGY_GOLDEN = 12.5  # kcal/mol
# kcal mol⁻¹ Å⁻¹
_FORCES_GOLDEN = torch.tensor([[50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]], dtype=_DTYPE)
_H_LONGITUDINAL_GOLDEN = 100.0  # H[0,0] = k
_H_CROSS_GOLDEN = -100.0  # H[0,3] = −k
_H_TRANSVERSE_GOLDEN = 33.333333333333336  # H[1,1] = k(r−r₀)/r

_CHI = (0.0, 1.0)
_ETA = (2.0, 2.0)
_Q_GOLDEN = torch.tensor([0.25, -0.25], dtype=_DTYPE)  # e


class _ConstantBondHead(nn.Module):
    """Constant-but-learnable ``k`` / ``r0``, so goldens are analytic."""

    def __init__(self) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.tensor(_BOND_K, dtype=_DTYPE))
        self.r0 = nn.Parameter(torch.tensor(_BOND_R0, dtype=_DTYPE))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(features.shape[0])
        return {"k": self.k.expand(n), "r0": self.r0.expand(n)}


class _ConstantChiEtaHead(nn.Module):
    """``ChiEtaHead``-shaped stub emitting the analytic χ / η of the KKT case."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("chi", torch.tensor(list(_CHI), dtype=_DTYPE))
        self.register_buffer("eta", torch.tensor(list(_ETA), dtype=_DTYPE))

    def forward(self, atom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        del atom_features
        return {"chi": self.chi, "eta": self.eta}


class _ConstantEncoder(nn.Module):
    """Deterministic per-namespace features; geometry-independent by construction."""

    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = features

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> dict[str, torch.Tensor]:
        del td
        return dict(self._features)


def _two_atom_bond_batch() -> TensorDict:
    """One molecule: two bonded atoms ``_BOND_R`` Å apart along x̂."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [_BOND_R, 0.0, 0.0]], dtype=_DTYPE)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([1, 1], dtype=torch.long),
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


def _bond_parameterizer() -> Parameterizer:
    """Bonded-only Parameterizer with the constant heads and the EEM charge path."""
    features = {
        "bonds": torch.zeros(1, 2, dtype=_DTYPE),
        "atoms": torch.zeros(2, 2, dtype=_DTYPE),
    }
    return Parameterizer(
        _ConstantEncoder(features),
        ForceField(name="class-i-adapter", units="real").def_bondstyle("harmonic"),
        bond_head=_ConstantBondHead(),
        charge_model=AtomicChargeModel(
            _ConstantChiEtaHead(),
            ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0),
        ),
    )


def main() -> int:
    model = _bond_parameterizer()
    batch = _two_atom_bond_batch()

    derivatives = ForceFieldPredictions(
        model,
        keys=("energy", "forces", "hessian"),
        hessian_derivation=HessianDerivation(),
    )
    out = derivatives(batch)
    if set(out) != {"energy", "forces", "hessian"}:
        raise AssertionError(f"declared keys not honoured; got {sorted(out)}")

    energy = out["energy"].detach()
    if abs(float(energy) - _ENERGY_GOLDEN) > _ATOL:
        raise AssertionError(f"energy must be {_ENERGY_GOLDEN} kcal/mol; got {float(energy)}")

    forces = out["forces"].detach()
    if tuple(forces.shape) != (2, 3):
        raise AssertionError(f"forces must be (2, 3); got {tuple(forces.shape)}")
    if not torch.allclose(forces, _FORCES_GOLDEN, atol=_ATOL, rtol=0.0):
        raise AssertionError(f"force drift:\n{forces}\nvs\n{_FORCES_GOLDEN}")

    hessian = out["hessian"].detach()
    if tuple(hessian.shape) != (6, 6):
        raise AssertionError(f"hessian must be (6, 6); got {tuple(hessian.shape)}")
    if abs(float(hessian[0, 0]) - _H_LONGITUDINAL_GOLDEN) > _ATOL:
        raise AssertionError(f"H[0,0] must be {_H_LONGITUDINAL_GOLDEN}; got {float(hessian[0, 0])}")
    if abs(float(hessian[0, 3]) - _H_CROSS_GOLDEN) > _ATOL:
        raise AssertionError(f"H[0,3] must be {_H_CROSS_GOLDEN}; got {float(hessian[0, 3])}")
    if abs(float(hessian[1, 1]) - _H_TRANSVERSE_GOLDEN) > _ATOL:
        raise AssertionError(f"H[1,1] must be {_H_TRANSVERSE_GOLDEN}; got {float(hessian[1, 1])}")
    if not torch.allclose(hessian, hessian.T, atol=_ATOL, rtol=0.0):
        raise AssertionError(f"hessian is not symmetric:\n{hessian}")

    charges_only = ForceFieldPredictions(model, keys=("q",))(batch)
    if set(charges_only) != {"q"}:
        raise AssertionError(f'keys=("q",) must return only "q"; got {sorted(charges_only)}')
    if "energy" in charges_only:
        raise AssertionError('a q-only adapter must not compute an "energy" prediction')

    q = charges_only["q"].detach()
    if tuple(q.shape) != (2,):
        raise AssertionError(f"q must be (2,); got {tuple(q.shape)}")
    if not torch.allclose(q, _Q_GOLDEN, atol=_ATOL, rtol=0.0):
        raise AssertionError(f"EEM charge drift: {q} vs {_Q_GOLDEN}")

    print(
        "universal-class-i-08-adapter: E = 12.5 kcal/mol, F = (±50,0,0) kcal/mol/A, "
        "H[0,0] = 100.0 / H[0,3] = -100.0 / H[1,1] = 33.333333333333336 kcal/mol/A^2 "
        "(symmetric), q = (+0.25,-0.25) e, q-only dict has no energy key OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
