"""Tests for ForceField + ClassicalIMM (Class-I learned MM)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition.heads import ChargeHead, LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead
from molpot.forcefield import ClassicalIMM, ClassIForceField, ForceField


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


class _ConstantBondHead(nn.Module):
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


class _FakeEncoder(nn.Module):
    def __init__(self, features: dict[str, torch.Tensor]):
        super().__init__()
        self._features = features
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict):
        class _E:
            def __init__(self, f):
                self._f = f

            def interaction_dict(self):
                return dict(self._f)

        return _E(self._features)


class TestForceFieldParams:
    def test_default_template_is_class_i(self):
        model = ClassicalIMM(_FakeEncoder({}))
        assert type(model.forcefield) is ClassIForceField

    def test_forward_writes_continuous_params(self):
        bond_head = BondParamHead(feature_dim=4, hidden_dim=8)
        angle_head = AngleParamHead(feature_dim=4, hidden_dim=8)
        atom_head = MultiHead(
            {
                "lj": LJParameterHead(feature_dim=4, hidden_dim=8),
                "q": ChargeHead(feature_dim=4, hidden_dim=8),
            }
        )
        ff = (
            ForceField(name="abs", units="real")
            .def_bondstyle("harmonic")
            .def_anglestyle("harmonic")
            .def_pairstyle("lj/cut")
        )
        features = {
            "bonds": torch.randn(2, 4),
            "angles": torch.randn(1, 4),
            "atoms": torch.randn(3, 4),
        }
        model = ClassicalIMM(
            _FakeEncoder(features),
            forcefield=ff,
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
        filled = model(batch)
        assert isinstance(filled, ForceField)
        assert filled.params("bonds") is not None
        assert filled.params("bonds")["k"].shape == (2,)
        assert filled.params("angles") is not None
        assert filled.params("atoms") is not None
        schema = filled.to_potentials().get_schema()
        assert "bond:harmonic" in schema
        assert "angle:harmonic" in schema


class TestEnergy:
    def test_bond_harmonic_via_to_potentials(self):
        k, r0, r = 2.0, 1.0, 1.5
        expected = 0.5 * k * (r - r0) ** 2
        ff = ForceField().def_bondstyle("harmonic")
        model = ClassicalIMM(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            forcefield=ff,
            bond_head=_ConstantBondHead(k=k, r0=r0),
        )
        batch = _two_atom_bond_batch(r=r)
        energy = model(batch).to_potentials().calc_energy(batch)
        assert energy.shape == ()
        assert math.isclose(float(energy), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_potentials_is_embeddable_module(self):
        from molpot.derivation.protocol import ENERGY_KEY, FORCES_KEY
        from molpot.potentials.base import BasePotential

        ff = (
            ForceField()
            .def_bondstyle("harmonic")
            .with_params(bonds={"k": torch.tensor([2.0]), "r0": torch.tensor([1.0])})
        )
        batch = _two_atom_bond_batch(r=1.5)
        pots = ff.to_potentials()
        assert isinstance(pots, BasePotential)
        assert isinstance(pots, nn.Module)
        out = pots(batch)
        assert out is batch
        assert ENERGY_KEY in out.keys(include_nested=True)
        assert FORCES_KEY in out.keys(include_nested=True)
        assert out[FORCES_KEY].shape == (2, 3)

    def test_forward_returns_only_forcefield(self):
        ff = ForceField().def_bondstyle("harmonic")
        model = ClassicalIMM(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            forcefield=ff,
            bond_head=_ConstantBondHead(k=4.0, r0=1.2),
        )
        batch = _two_atom_bond_batch(r=1.4)
        out = model(batch)
        assert isinstance(out, ForceField)
        assert not isinstance(out, dict)


class TestForces:
    def test_calc_forces_on_potentials(self):
        ff = ForceField().def_bondstyle("harmonic")
        model = ClassicalIMM(
            _FakeEncoder({"bonds": torch.zeros(1, 1, dtype=torch.float64)}),
            forcefield=ff,
            bond_head=_ConstantBondHead(k=2.0, r0=1.0),
        )
        batch = _two_atom_bond_batch(r=1.5)
        pots = model(batch).to_potentials()
        forces = pots.calc_forces(batch)
        assert forces.shape == (2, 3)
        assert math.isclose(float(forces[0, 0].detach()), 2.0 * 0.5, abs_tol=1e-5)
        assert math.isclose(float(forces[1, 0].detach()), -2.0 * 0.5, abs_tol=1e-5)


class _OplsLikeProperHead(nn.Module):
    """Fixed 4-term cosine with OPLS phases (γ = 0,π,0,π), k_m = K_m/2."""

    def __init__(self, K: tuple[float, float, float, float] = (1.0, 2.0, 0.5, 0.25)):
        super().__init__()
        self.K = K

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        device, dtype = features.device, features.dtype
        K = torch.tensor(self.K, device=device, dtype=dtype)
        k = (0.5 * K).unsqueeze(0).expand(n, 4).contiguous()
        phase = torch.zeros(n, 4, device=device, dtype=dtype)
        phase[:, 1] = math.pi
        phase[:, 3] = math.pi
        return {
            "k": k,
            "phase": phase,
            "periodicity": torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long),
            "idivf": torch.ones(n, device=device, dtype=dtype),
        }


def _butane_like_batch(*, dtype: torch.dtype = torch.float64) -> TensorDict:
    """Four atoms with one proper torsion (not planar; non-zero φ)."""
    pos = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.3],
        ],
        dtype=dtype,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([6, 6, 6, 6]),
                    "batch": torch.zeros(4, dtype=torch.long),
                },
                batch_size=[4],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 1, 2], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                },
                batch_size=[3],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


class TestProject:
    def test_project_fourier_to_opls_energy_and_forces_match(self):
        """Representable Fourier (n=1..4, OPLS phases) → OPLS: E/F exact."""
        src_ff = ClassIForceField()
        model = ClassicalIMM(
            _FakeEncoder(
                {
                    "bonds": torch.zeros(3, 2, dtype=torch.float64),
                    "propers": torch.zeros(1, 4, dtype=torch.float64),
                }
            ),
            forcefield=src_ff,
            bond_head=_ConstantBondHead(2.0, 1.5),
            proper_head=_OplsLikeProperHead(),
        )
        batch = _butane_like_batch()
        filled = model(batch)
        # Target form = OPLS dihedral + harmonic bonds (no angles in this fixture)
        tgt = (
            ForceField(name="OPLS", units="real")
            .def_bondstyle("harmonic")
            .def_dihedralstyle("opls")
        )
        projected = model.project(tgt, source=filled)

        assert type(filled) is ClassIForceField
        assert type(projected) is ForceField
        assert projected.dihedral_style() == "opls"
        pp = projected.params("propers")
        assert pp is not None
        # K = 2 * k for OPLS phases
        assert torch.allclose(pp["k1"], torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(pp["k2"], torch.tensor([2.0], dtype=torch.float64))

        e_src = filled.to_potentials().calc_energy(batch)
        e_tgt = projected.to_potentials().calc_energy(batch)
        assert torch.allclose(e_src, e_tgt, atol=1e-10, rtol=0)

        f_src = filled.to_potentials().calc_forces(batch)
        f_tgt = projected.to_potentials().calc_forces(batch)
        assert torch.allclose(f_src, f_tgt, atol=1e-8, rtol=0)

    def test_project_raises_on_extra_fourier_terms(self):
        src = ClassIForceField().with_params(
            bonds={
                "k": torch.tensor([1.0]),
                "r0": torch.tensor([1.0]),
            },
            propers={
                "k": torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.5]]),  # n=5 non-zero
                "phase": torch.zeros(1, 5),
                "periodicity": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "idivf": torch.ones(1),
            },
        )
        model = ClassicalIMM(_FakeEncoder({"bonds": torch.zeros(1, 1)}), forcefield=src)
        tgt = ForceField().def_bondstyle("harmonic").def_dihedralstyle("opls")
        with pytest.raises(ValueError, match="n=5"):
            model.project(tgt, source=src)

    def test_project_raises_when_target_bond_missing_in_source(self):
        src = (
            ForceField(name="src")
            .def_anglestyle("harmonic")
            .with_params(angles={"k": torch.tensor([1.0]), "theta0": torch.tensor([1.0])})
        )
        model = ClassicalIMM(_FakeEncoder({}), forcefield=src)
        tgt = ForceField().def_bondstyle("harmonic").def_anglestyle("harmonic")
        with pytest.raises(ValueError, match="no bonds params"):
            model.project(tgt, source=src)

    def test_project_cartesian_b_zero_n1_to_n4_gives_opls_k(self):
        a = torch.tensor([[0.5, 0.3, 0.2, 0.1]], dtype=torch.float64)
        src = ClassIForceField().with_params(
            propers={
                "a": a,
                "b": torch.zeros(1, 4, dtype=torch.float64),
                "periodicity": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "idivf": torch.ones(1, dtype=torch.float64),
            }
        )
        model = ClassicalIMM(_FakeEncoder({}), forcefield=src)
        tgt = ForceField().def_dihedralstyle("opls")
        projected = model.project(tgt, source=src)
        pp = projected.params("propers")
        assert pp is not None
        # K_odd = 2a, K_even = -2a
        assert torch.allclose(pp["k1"], torch.tensor([1.0], dtype=torch.float64), atol=1e-12)
        assert torch.allclose(pp["k2"], torch.tensor([-0.6], dtype=torch.float64), atol=1e-12)
        assert torch.allclose(pp["k3"], torch.tensor([0.4], dtype=torch.float64), atol=1e-12)
        assert torch.allclose(pp["k4"], torch.tensor([-0.2], dtype=torch.float64), atol=1e-12)

    def test_project_raises_on_n6_cartesian_term(self):
        src = ClassIForceField().with_params(
            propers={
                "a": torch.tensor([[0.5]], dtype=torch.float64),
                "b": torch.zeros(1, 1, dtype=torch.float64),
                "periodicity": torch.tensor([6], dtype=torch.long),
                "idivf": torch.ones(1, dtype=torch.float64),
            }
        )
        model = ClassicalIMM(_FakeEncoder({}), forcefield=src)
        tgt = ForceField().def_dihedralstyle("opls")
        with pytest.raises(ValueError, match="n=6"):
            model.project(tgt, source=src)

    def test_project_raises_on_sine_residual_cartesian(self):
        src = ClassIForceField().with_params(
            propers={
                "a": torch.tensor([[0.0]], dtype=torch.float64),
                "b": torch.tensor([[0.4]], dtype=torch.float64),
                "periodicity": torch.tensor([1], dtype=torch.long),
                "idivf": torch.ones(1, dtype=torch.float64),
            }
        )
        model = ClassicalIMM(_FakeEncoder({}), forcefield=src)
        tgt = ForceField().def_dihedralstyle("opls")
        with pytest.raises(ValueError, match="sin|phase"):
            model.project(tgt, source=src)


class TestImportBoundary:
    def test_forcefield_modules_have_no_molzoo(self):
        root = Path(__file__).resolve().parents[3] / "src/molpot/forcefield"
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molzoo")
                        assert not alias.name.startswith("molrep.chem")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("molzoo")
                    assert not node.module.startswith("molrep.chem")


class TestDocstrings:
    def test_public_symbols_have_docstrings(self):
        for cls in (
            ForceField,
            ClassicalIMM,
            BondParamHead,
            AngleParamHead,
            ProperTorsionParamHead,
            ImproperParamHead,
        ):
            doc = cls.__doc__ or ""
            assert len(doc.strip()) > 20, f"{cls.__name__} missing docstring"
