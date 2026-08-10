"""Tests for ClassicalMMComposer (learnable-classical-ff-03)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch
from tensordict import TensorDict

from molpot.composition.classical_mm import ClassicalMMComposer
from molpot.composition.heads import ChargeHead, LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead
from molpot.derivation import ForceDerivation
from molpot.ir import CLASS_I_CANONICAL, BondBag, PotentialIR


def _two_atom_bond_batch(
    *,
    r: float = 1.5,
    dtype: torch.dtype = torch.float64,
) -> TensorDict:
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=dtype)
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


class _ConstantBondHead(torch.nn.Module):
    """Mock head returning fixed k, r0 (ignores features)."""

    def __init__(self, k: float, r0: float):
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


# ---------------------------------------------------------------------------
# parameterize → PotentialIR (ac-006)
# ---------------------------------------------------------------------------


class TestClassicalMMComposerParameterize:
    def test_parameterize_builds_class_i_ir(self):
        bond_head = BondParamHead(feature_dim=4, hidden_dim=8)
        angle_head = AngleParamHead(feature_dim=4, hidden_dim=8)
        atom_head = MultiHead(
            {
                "lj": LJParameterHead(feature_dim=4, hidden_dim=8),
                "q": ChargeHead(feature_dim=4, hidden_dim=8),
            }
        )
        composer = ClassicalMMComposer(
            bond_head=bond_head,
            angle_head=angle_head,
            atom_head=atom_head,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(3, 3),
                        "Z": torch.tensor([6, 1, 1]),
                        "batch": torch.zeros(3, dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 0]),
                        "atomj": torch.tensor([1, 2]),
                    },
                    batch_size=[2],
                ),
                "angles": TensorDict(
                    {
                        "atomi": torch.tensor([1]),
                        "atomj": torch.tensor([0]),
                        "atomk": torch.tensor([2]),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        features = {
            "bonds": torch.randn(2, 4),
            "angles": torch.randn(1, 4),
            "atoms": torch.randn(3, 4),
        }
        ir = composer.parameterize(features, batch)
        assert isinstance(ir, PotentialIR)
        assert ir.unit_system == "class_i_canonical"
        assert ir.bonds is not None
        assert ir.bonds.k.shape == (2,)
        assert ir.bonds.r0.shape == (2,)
        assert ir.angles is not None
        assert ir.angles.k.shape == (1,)
        assert ir.lj is not None
        assert ir.lj.epsilon.shape == (3,)
        assert ir.charges is not None
        assert ir.charges.q.shape == (3,)
        assert ir.scaling is not None
        # Units tag consistency
        assert CLASS_I_CANONICAL["energy"] == "kcal/mol"


# ---------------------------------------------------------------------------
# Energy golden (ac-007)
# ---------------------------------------------------------------------------


class TestClassicalMMComposerEnergy:
    def test_bond_harmonic_analytic_golden(self):
        k, r0, r = 2.0, 1.0, 1.5
        expected = 0.5 * k * (r - r0) ** 2  # 0.25
        composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=k, r0=r0))
        batch = _two_atom_bond_batch(r=r)
        features = {"bonds": torch.zeros(1, 1, dtype=torch.float64)}
        ir = composer.parameterize(features, batch)
        energy = composer.energy(ir, batch, pos=batch["atoms", "pos"])
        assert energy.shape == ()
        assert math.isclose(float(energy), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_forward_chains_parameterize_and_energy(self):
        k, r0, r = 4.0, 1.2, 1.4
        expected = 0.5 * k * (r - r0) ** 2
        composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=k, r0=r0))
        batch = _two_atom_bond_batch(r=r)
        features = {"bonds": torch.zeros(1, 1, dtype=torch.float64)}
        out = composer(batch, features)
        assert math.isclose(float(out["energy"]), expected, rel_tol=1e-5, abs_tol=1e-5)
        assert isinstance(out["ir"], PotentialIR)

    def test_optional_injected_evaluator(self):
        """Injected evaluator receives IR + batch and returns energy sum."""
        bag = BondBag(k=torch.tensor([1.0]), r0=torch.tensor([1.0]))

        def evaluator(ir: PotentialIR, batch, *, pos=None) -> torch.Tensor:
            assert ir.bonds is not None
            return torch.tensor(42.0, dtype=torch.float64)

        composer = ClassicalMMComposer(
            bond_head=_ConstantBondHead(k=1.0, r0=1.0),
            evaluator=evaluator,
        )
        batch = _two_atom_bond_batch()
        ir = PotentialIR(bonds=bag, unit_system="class_i_canonical")
        e = composer.energy(ir, batch, pos=batch["atoms", "pos"])
        assert float(e) == 42.0


# ---------------------------------------------------------------------------
# Forces via ForceDerivation only (ac-008)
# ---------------------------------------------------------------------------


class TestClassicalMMComposerForces:
    def test_energy_differentiable_wrt_pos_via_force_derivation(self):
        composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=2.0, r0=1.0))
        batch = _two_atom_bond_batch(r=1.5)
        features = {"bonds": torch.zeros(1, 1, dtype=torch.float64)}
        ir = composer.parameterize(features, batch)
        pos = batch["atoms", "pos"].clone()

        def energy_fn(p: torch.Tensor) -> torch.Tensor:
            return composer.energy(ir, batch, pos=p)

        forces = ForceDerivation(method="autograd")(energy_fn, pos)
        assert forces.shape == (2, 3)
        # Along bond axis: atom0 pulls toward equilibrium, atom1 opposite (Newton 3)
        assert forces[0, 0] > 0  # r > r0 → force on atom0 toward +x? F = -dE/dx
        # E = 0.5*k*(r-r0)^2, r = x1-x0, dE/dx0 = k*(r-r0)*(-1) → F0 = +k*(r-r0)
        assert math.isclose(float(forces[0, 0]), 2.0 * (1.5 - 1.0), abs_tol=1e-5)
        assert math.isclose(float(forces[1, 0]), -2.0 * (1.5 - 1.0), abs_tol=1e-5)

    def test_source_has_no_hand_rolled_force_formula(self):
        src = Path(__file__).resolve().parents[3] / "src/molpot/composition/classical_mm.py"
        text = src.read_text()
        # No analytic force kernels beyond ForceDerivation usage.
        forbidden = [
            "force = -k *",
            "forces = -",
            "def calc_forces",
            "def forces(",
        ]
        for needle in forbidden:
            assert needle not in text, f"hand-rolled force pattern found: {needle!r}"


# ---------------------------------------------------------------------------
# Import boundary (ac-009)
# ---------------------------------------------------------------------------


class TestNoEncoderImports:
    def test_classical_mm_and_mm_heads_have_no_molzoo_or_molrep_chem(self):
        root = Path(__file__).resolve().parents[3] / "src/molpot/composition"
        for name in ("classical_mm.py", "mm_heads.py"):
            path = root / name
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molzoo")
                        assert not alias.name.startswith("molrep.chem")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("molzoo")
                    assert not node.module.startswith("molrep.chem")


# ---------------------------------------------------------------------------
# Docstrings (ac-010 partial)
# ---------------------------------------------------------------------------


class TestDocstrings:
    def test_public_symbols_have_google_docstrings(self):
        for cls in (
            BondParamHead,
            AngleParamHead,
            ProperTorsionParamHead,
            ImproperParamHead,
            ClassicalMMComposer,
        ):
            doc = cls.__doc__ or ""
            assert len(doc.strip()) > 20, f"{cls.__name__} missing docstring"
