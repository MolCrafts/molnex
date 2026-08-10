"""Tests for ClassicalMMParameterizer (learnable-classical-ff-05)."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition.classical_mm import ClassicalMMComposer
from molpot.composition.parameterizer import (
    KCAL_MOL_TO_EV,
    ChemEmbeddingsLike,
    ChemEncoderProtocol,
    ClassicalMMParameterizer,
    energy_kcal_to_ev,
)
from molpot.derivation import ForceDerivation
from molpot.ir import CLASS_I_CANONICAL, PotentialIR

# ---------------------------------------------------------------------------
# Fixtures / fakes (no molzoo)
# ---------------------------------------------------------------------------


class _FakeEmbeddings:
    """Structural ChemEmbeddingsLike: interaction_dict only."""

    def __init__(self, features: Mapping[str, torch.Tensor]) -> None:
        self._features = dict(features)

    def interaction_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._features)


class FakeEncoder(nn.Module):
    """Minimal encoder satisfying ChemEncoderProtocol without molzoo/molrep.chem.

    Returns fixed feature tensors (constant heads ignore feature content).
    """

    def __init__(self, features: Mapping[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self._features = dict(features) if features is not None else {}
        # Ensure at least one trainable param so encoder registers as a submodule.
        self._dummy = nn.Parameter(torch.zeros(1))

    def set_features(self, features: Mapping[str, torch.Tensor]) -> None:
        self._features = dict(features)

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict) -> _FakeEmbeddings:
        return _FakeEmbeddings(self._features)


class _ConstantBondHead(nn.Module):
    """Mock head returning fixed k, r0 (ignores features)."""

    def __init__(self, k: float, r0: float) -> None:
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


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


def _bond_features(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    return {"bonds": torch.zeros(1, 4, dtype=dtype)}


def _make_parameterizer(
    *,
    k: float = 2.0,
    r0: float = 1.0,
    features: Mapping[str, torch.Tensor] | None = None,
) -> ClassicalMMParameterizer:
    feat = dict(features) if features is not None else _bond_features()
    encoder = FakeEncoder(feat)
    composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=k, r0=r0))
    return ClassicalMMParameterizer(encoder=encoder, composer=composer)


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


class TestChemEncoderProtocol:
    def test_fake_encoder_satisfies_protocol(self):
        enc = FakeEncoder(_bond_features())
        assert isinstance(enc, ChemEncoderProtocol)

    def test_fake_embeddings_satisfies_chem_embeddings_like(self):
        emb = _FakeEmbeddings(_bond_features())
        assert isinstance(emb, ChemEmbeddingsLike)

    def test_encode_returns_interaction_feature_dict(self):
        param = _make_parameterizer()
        batch = _two_atom_bond_batch()
        features = param.encode(batch)
        assert "bonds" in features
        assert features["bonds"].shape == (1, 4)


# ---------------------------------------------------------------------------
# parameterize → IR units kcal/mol + energy smoke
# ---------------------------------------------------------------------------


class TestParameterizeAndEnergy:
    def test_parameterize_class_i_ir_kcal_mol(self):
        param = _make_parameterizer(k=2.0, r0=1.0)
        batch = _two_atom_bond_batch(r=1.5)
        ir = param.parameterize(batch)
        assert isinstance(ir, PotentialIR)
        assert ir.unit_system == "class_i_canonical"
        assert CLASS_I_CANONICAL["energy"] == "kcal/mol"
        assert ir.bonds is not None
        assert ir.bonds.k.shape == (1,)
        assert math.isclose(float(ir.bonds.k[0]), 2.0, abs_tol=1e-8)
        assert math.isclose(float(ir.bonds.r0[0]), 1.0, abs_tol=1e-8)

    def test_energy_matches_hand_built_bond_harmonic(self):
        k, r0, r = 2.0, 1.0, 1.5
        expected = 0.5 * k * (r - r0) ** 2  # 0.25 kcal/mol
        param = _make_parameterizer(k=k, r0=r0)
        batch = _two_atom_bond_batch(r=r)
        energy = param.energy(batch)
        assert energy.shape == ()
        assert math.isclose(float(energy), expected, rel_tol=1e-5, abs_tol=1e-5)

    def test_energy_with_precomputed_ir(self):
        param = _make_parameterizer(k=4.0, r0=1.2)
        batch = _two_atom_bond_batch(r=1.4)
        ir = param.parameterize(batch)
        e1 = param.energy(batch, ir=ir)
        e2 = param.energy(batch)
        assert math.isclose(float(e1), float(e2), abs_tol=1e-10)

    def test_parameterize_accepts_explicit_features(self):
        """Skip encoder when features= is provided."""
        encoder = FakeEncoder({"bonds": torch.ones(1, 4, dtype=torch.float64)})
        composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=2.0, r0=1.0))
        param = ClassicalMMParameterizer(encoder=encoder, composer=composer)
        batch = _two_atom_bond_batch(r=1.5)
        override = {"bonds": torch.zeros(1, 8, dtype=torch.float64)}
        ir = param.parameterize(batch, features=override)
        assert ir.bonds is not None
        # Constant head ignores feature width; energy still golden.
        e = param.energy(batch, ir=ir)
        assert math.isclose(float(e), 0.25, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Forces via ForceDerivation only
# ---------------------------------------------------------------------------


class TestForces:
    def test_forward_compute_forces_shape(self):
        param = _make_parameterizer(k=2.0, r0=1.0)
        batch = _two_atom_bond_batch(r=1.5)
        out = param.forward(batch, compute_forces=True)
        assert "energy" in out
        assert "forces" in out
        forces = out["forces"]
        assert forces.shape == (2, 3)
        # Analytic: F0_x = +k*(r-r0), F1_x = -k*(r-r0)
        assert math.isclose(float(forces[0, 0]), 2.0 * (1.5 - 1.0), abs_tol=1e-5)
        assert math.isclose(float(forces[1, 0]), -2.0 * (1.5 - 1.0), abs_tol=1e-5)

    def test_forward_writes_batch_keys(self):
        param = _make_parameterizer(k=2.0, r0=1.0)
        batch = _two_atom_bond_batch(r=1.5)
        out = param.forward(batch, compute_forces=True)
        # graphs.energy / atoms.forces convention
        assert "graphs" in batch
        assert "energy" in batch["graphs"]
        assert "forces" in batch["atoms"]
        assert torch.allclose(out["energy"], batch["graphs", "energy"])
        assert torch.allclose(out["forces"], batch["atoms", "forces"])

    def test_forces_via_injected_force_derivation(self):
        enc = FakeEncoder(_bond_features())
        composer = ClassicalMMComposer(bond_head=_ConstantBondHead(k=2.0, r0=1.0))
        fd = ForceDerivation(method="autograd")
        param = ClassicalMMParameterizer(encoder=enc, composer=composer, force_derivation=fd)
        batch = _two_atom_bond_batch(r=1.5)
        out = param(batch, compute_forces=True)
        assert out["forces"].shape == (2, 3)

    def test_source_has_no_hand_rolled_force_formula(self):
        src = Path(__file__).resolve().parents[3] / "src/molpot/composition/parameterizer.py"
        text = src.read_text()
        forbidden = [
            "force = -k *",
            "forces = -grad",
            "def calc_forces",
        ]
        for needle in forbidden:
            assert needle not in text, f"hand-rolled force pattern found: {needle!r}"
        assert "ForceDerivation" in text


# ---------------------------------------------------------------------------
# Units boundary
# ---------------------------------------------------------------------------


class TestUnitsBoundary:
    def test_ir_stays_kcal_not_mutated_to_ev(self):
        param = _make_parameterizer()
        batch = _two_atom_bond_batch()
        ir = param.parameterize(batch)
        assert ir.unit_system == "class_i_canonical"
        # Conversion helper does not mutate IR
        e_kcal = param.energy(batch, ir=ir)
        e_ev = energy_kcal_to_ev(e_kcal)
        assert ir.unit_system == "class_i_canonical"
        assert not math.isclose(float(e_kcal), float(e_ev), rel_tol=0.1)
        assert math.isclose(
            float(e_ev),
            float(e_kcal) * KCAL_MOL_TO_EV,
            abs_tol=1e-12,
        )

    def test_kcal_to_ev_factor_matches_documented_constant(self):
        # 1 eV ≈ 23.060547830619026 kcal/mol → KCAL_MOL_TO_EV ≈ 0.0433641
        assert math.isclose(KCAL_MOL_TO_EV, 1.0 / 23.060547830619026, rel_tol=1e-12)
        x = torch.tensor(23.060547830619026)
        assert math.isclose(float(energy_kcal_to_ev(x)), 1.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Import boundary + submodule registration
# ---------------------------------------------------------------------------


class TestImportBoundaryAndRegistration:
    def test_parameterizer_module_has_no_molzoo_import(self):
        path = Path(__file__).resolve().parents[3] / "src/molpot/composition/parameterizer.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molzoo")
                    assert not alias.name.startswith("molrep.chem")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molzoo")
                assert not node.module.startswith("molrep.chem")

    def test_encoder_registered_as_submodule(self):
        param = _make_parameterizer()
        names = {name for name, _ in param.named_modules()}
        assert "encoder" in names
        assert "composer" in names
        # Dummy param from FakeEncoder is in state_dict
        assert any(k.startswith("encoder.") for k in param.state_dict())


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


class TestDocstrings:
    def test_public_symbols_have_google_docstrings(self):
        for obj in (
            ClassicalMMParameterizer,
            ClassicalMMParameterizer.encode,
            ClassicalMMParameterizer.parameterize,
            ClassicalMMParameterizer.energy,
            ClassicalMMParameterizer.forward,
            energy_kcal_to_ev,
        ):
            doc = obj.__doc__ or ""
            assert len(doc.strip()) > 20, f"{obj} missing docstring"
