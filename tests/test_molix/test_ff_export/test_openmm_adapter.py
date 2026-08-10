"""OpenMMAdapter force-spec emission tests (no live openmm)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.openmm_adapter import OpenMMAdapter
from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    LJBag,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
)


def _class_i_ir() -> PotentialIR:
    return PotentialIR(
        bonds=BondBag(
            k=torch.tensor([100.0]),
            r0=torch.tensor([1.5]),  # Å
        ),
        angles=AngleBag(
            k=torch.tensor([50.0]),
            theta0=torch.tensor([1.9106332362490186]),  # ~109.47°
        ),
        propers=ProperTorsionBag(
            # AMBER Vn=2 → IR half-barrier k=1 with idivf=1
            k=torch.tensor([[1.0]]),
            periodicity=torch.tensor([2], dtype=torch.long),
            phase=torch.tensor([[0.0]]),
            idivf=torch.tensor([1.0]),
        ),
        lj=LJBag(
            epsilon=torch.tensor([0.1]),  # kcal/mol
            sigma=torch.tensor([3.5]),  # Å
        ),
        charges=ChargeBag(q=torch.tensor([0.5, -0.5])),
        scaling=NonbondedScaling(),
    )


class TestOpenMMAdapterRegistration:
    def test_registered_as_openmm(self):
        assert "openmm" in BackendAdapter.names()
        adapter = BackendAdapter.from_name("openmm")
        assert isinstance(adapter, OpenMMAdapter)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown adapter"):
            BackendAdapter.from_name("not-a-backend")


class TestOpenMMAdapterTranslate:
    def test_bond_k_and_r0_units(self):
        adapter = OpenMMAdapter()
        spec = adapter.translate(_class_i_ir())
        bond_force = next(f for f in spec.forces if f["type"] == "HarmonicBondForce")
        p0 = bond_force["parameters"][0]
        assert p0["k"] == 41840.0
        assert p0["r0"] == pytest.approx(0.15)  # 1.5 Å → 0.15 nm
        assert bond_force["case"] == TranslationCase.DIRECT_UNIT_SCALE.value

    def test_torsion_vn2_golden_via_ir_k(self):
        """IR k=1 (half of AMBER Vn=2) → OpenMM k=4.184 kJ/mol."""
        adapter = OpenMMAdapter()
        spec = adapter.translate(_class_i_ir())
        torsion = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
        p0 = torsion["parameters"][0]
        assert p0["k"] == 4.184
        assert p0["periodicity"] == 2
        assert p0["phase"] == 0.0

    def test_idivf_absorbed_into_k(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[2.0]]),
                periodicity=torch.tensor([1], dtype=torch.long),
                phase=torch.tensor([[0.0]]),
                idivf=torch.tensor([2.0]),
            )
        )
        spec = OpenMMAdapter().translate(ir)
        torsion = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
        # (k/idivf)*4.184 = 1*4.184
        assert torsion["parameters"][0]["k"] == 4.184
        assert torsion["case"] == TranslationCase.FORM_REPARAMETERIZE.value

    def test_multi_term_proper_decomposes(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[1.0, 0.5]]),
                periodicity=torch.tensor([1, 2], dtype=torch.long),
                phase=torch.tensor([[0.0, math.pi]]),
                idivf=torch.tensor([1.0]),
            )
        )
        spec = OpenMMAdapter().translate(ir)
        torsion = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
        assert len(torsion["parameters"]) == 2
        assert torsion["case"] == TranslationCase.DECOMPOSE.value
        assert torsion["parameters"][0]["k"] == 4.184
        assert torsion["parameters"][1]["k"] == pytest.approx(2.092)
        assert torsion["parameters"][1]["periodicity"] == 2

    def test_nonbonded_units_and_charges(self):
        spec = OpenMMAdapter().translate(_class_i_ir())
        nb = next(f for f in spec.forces if f["type"] == "NonbondedForce")
        assert nb["parameters"][0]["sigma"] == pytest.approx(0.35)  # 3.5 Å
        assert nb["parameters"][0]["epsilon"] == pytest.approx(0.4184)
        assert nb["charges"][0] == 0.5
        assert nb["charges"][1] == -0.5

    def test_scaling_14_from_ir(self):
        """ac-006: Class-I defaults scale_q_14=5/6, scale_lj_14=0.5."""
        spec = OpenMMAdapter().translate(_class_i_ir())
        assert math.isclose(spec.scaling["scale_q_14"], 5.0 / 6.0)
        assert math.isclose(spec.scaling["scale_lj_14"], 0.5)
        assert math.isclose(spec.scaling["scale_q_12"], 0.0)
        assert math.isclose(spec.scaling["scale_lj_12"], 0.0)

    def test_force_spec_json_serializable(self):
        spec = OpenMMAdapter().translate(_class_i_ir())
        payload = json.dumps(spec.to_dict())
        assert "HarmonicBondForce" in payload
        assert "PeriodicTorsionForce" in payload

    def test_unsupported_improper_harmonic_raises(self):
        ir = PotentialIR(
            impropers_harmonic=ImproperHarmonicBag(
                k=torch.tensor([1.0]),
                chi0=torch.tensor([0.0]),
            )
        )
        with pytest.raises(UnsupportedTermError, match="improper_harmonic") as ei:
            OpenMMAdapter().translate(ir)
        assert ei.value.term == "improper_harmonic"
        assert ei.value.case is TranslationCase.UNSUPPORTED
