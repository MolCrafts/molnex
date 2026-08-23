"""Public-API regression for Parameterizer + OPLSAAForceField fill path.

Spec: ``oplsaa-parameterizer-03-parameterizer``.

Hard-coded goldens only — no OpenMM / third-party oracle. Pins:

1. ``Parameterizer(encoder, OPLSAAForceField(), …heads…)`` forward fills
   bonds/angles/propers/impropers/atoms; propers carry ``k1..k4``
2. Original template is not mutated; no ``project`` attribute
3. Bond-only analytic energy via ``to_potentials``:
       E = ½ k (r − r0)² with k=100, r0=1.0 Å, r=1.5 Å → E = 12.5 kcal/mol

Provenance
----------
    formula          : Class-I harmonic bond E = ½k(r−r0)²
    geometry         : two H atoms on x̂, separation 1.5 Å
    citation         : Jorgensen et al. JACS 1996 DOI 10.1021/ja9621760
                       (OPLS-AA container); Wang et al. Chem. Sci. 2022
                       DOI 10.1039/D2SC02739A (learned MM parameterizer)
    capture command  : PYTHONPATH=src python regressions/oplsaa-parameterizer-03-parameterizer.py
    date             : 2026-08-20
    tolerance        : energy atol 1e-10
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.forcefield import ForceField, ForceFieldPredictions, OPLSAAForceField, Parameterizer

_D = 4
_GOLDEN_E = 12.5  # kcal/mol; ½·100·0.5²
_ENERGY_ATOL = 1e-10


class _FakeEncoder(nn.Module):
    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = features

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> dict[str, torch.Tensor]:
        del td
        return dict(self._features)


class _ConstantBondHead(nn.Module):
    def __init__(self, k: float = 100.0, r0: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


def main() -> int:
    # --- bond-only analytic pin via Parameterizer + to_potentials ---
    bond_ff = ForceField().def_bondstyle("harmonic")
    bond_model = Parameterizer(
        _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
        bond_ff,
        bond_head=_ConstantBondHead(k=100.0, r0=1.0),
    )
    assert not hasattr(bond_model, "project")
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    batch = TensorDict(
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
    assert bond_ff.params("bonds") is None
    filled_bond = bond_model(batch)
    assert bond_ff.params("bonds") is None
    energy = float(filled_bond.to_potentials().calc_energy(batch))
    if not math.isclose(energy, _GOLDEN_E, abs_tol=_ENERGY_ATOL):
        raise AssertionError(f"bond E={energy}, want {_GOLDEN_E}")

    preds = ForceFieldPredictions(bond_model, keys=("energy", "forces"))
    out = preds(batch)
    e_pred = float(out["energy"].detach())
    if not math.isclose(e_pred, _GOLDEN_E, abs_tol=_ENERGY_ATOL):
        raise AssertionError(f"predictions energy={e_pred}, want {_GOLDEN_E}")

    # --- OPLSAA schema-gated fill (params present; energy not asserted) ---
    features = {
        "bonds": torch.zeros(1, _D),
        "angles": torch.zeros(1, _D),
        "propers": torch.zeros(1, _D),
        "impropers": torch.zeros(1, _D),
        "atoms": torch.zeros(2, _D),
    }
    template = OPLSAAForceField()
    opls_model = Parameterizer(
        _FakeEncoder(features),
        template,
        bond_head=BondParamHead(feature_dim=_D, hidden_dim=8),
        angle_head=AngleParamHead(feature_dim=_D, hidden_dim=8),
        proper_head=ProperTorsionParamHead(feature_dim=_D, hidden_dim=8, form="opls"),
        improper_head=ImproperParamHead(feature_dim=_D, hidden_dim=8),
        atom_head=LJParameterHead(feature_dim=_D, hidden_dim=8),
        charge_model=AtomicChargeModel(
            ChiEtaHead(feature_dim=_D, hidden_dim=8),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        ),
    )
    assert not hasattr(opls_model, "project")
    opls_batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.zeros(2, 3),
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
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([1], dtype=torch.long),
                    "atoml": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
            "impropers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([1], dtype=torch.long),
                    "atoml": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    filled = opls_model(opls_batch)
    assert filled is not template
    for ns in ("bonds", "angles", "propers", "impropers", "atoms"):
        if filled.params(ns) is None:
            raise AssertionError(f"missing params for {ns}")
    pp = filled.params("propers")
    assert pp is not None
    for key in ("k1", "k2", "k3", "k4"):
        if key not in pp:
            raise AssertionError(f"propers missing {key}")

    print("oplsaa-parameterizer-03-parameterizer: Parameterizer + OPLSAA OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
