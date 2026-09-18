"""OpenMMAdapter force-spec emission tests (no live openmm)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

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
    NonbondedMethodRecord,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
)
from tests.test_molix.test_ff_export.conftest import PME_SWITCHED_METHOD

#: No radius and no solver — a model that truncates nothing — but the
#: Lennard-Jones combining rule is still stated, because every IR below carries
#: an ``lj`` bag and a bag whose rule is unrecorded is refused on export (see
#: ``test_an_lj_bag_whose_record_names_no_rule_is_refused``). "Unset" here means
#: the *method*, which is what this record is about; the rule is model identity
#: and has no unset state while there are ``ε`` / ``σ`` to combine.
_NO_METHOD = NonbondedMethodRecord(
    lj_cutoff=None,
    lj_switch_start=None,
    lj_mixing="lorentz-berthelot",
    coulomb_cutoff=None,
    coulomb_calculator=None,
    real_space_cutoff=None,
    smearing=None,
    lr_wavelength=None,
    mesh_spacing=None,
    capacity_factor=None,
)


def _nonbonded_ir(record: NonbondedMethodRecord, *, charges: bool = True) -> PotentialIR:
    """Single-type LJ (optionally charged) IR running the method ``record`` describes.

    Without charges there is no exclusion policy to declare either, so
    ``scaling`` goes with them.
    """
    return PotentialIR(
        lj=LJBag(
            epsilon=torch.tensor([0.1], dtype=torch.float64),
            sigma=torch.tensor([3.5], dtype=torch.float64),
        ),
        charges=ChargeBag(charge=torch.tensor([0.5, -0.5], dtype=torch.float64))
        if charges
        else None,
        scaling=NonbondedScaling() if charges else None,
        nonbonded_method=record,
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
            # AMBER Vn=2 → IR half-barrier k=1
            k=torch.tensor([[1.0]]),
            periodicity=torch.tensor([2], dtype=torch.long),
            phase=torch.tensor([[0.0]]),
        ),
        lj=LJBag(
            epsilon=torch.tensor([0.1]),  # kcal/mol
            sigma=torch.tensor([3.5]),  # Å
        ),
        charges=ChargeBag(charge=torch.tensor([0.5, -0.5])),
        scaling=NonbondedScaling(),
        # An IR that carries ``ε`` / ``σ`` states the rule they were mixed
        # with, or the export refuses it — the record is otherwise unset, so
        # this stays the untruncated "no method declared" fixture it was.
        nonbonded_method=_NO_METHOD,
    )


def _charges_only_ir() -> PotentialIR:
    """Charges under PME with no Lennard-Jones bag — nothing to combine."""
    return PotentialIR(
        charges=ChargeBag(charge=torch.tensor([0.5, -0.5], dtype=torch.float64)),
        scaling=NonbondedScaling(),
        nonbonded_method=replace(
            PME_SWITCHED_METHOD, lj_cutoff=None, lj_switch_start=None, lj_mixing=None
        ),
    )


def _lj_only_ir() -> PotentialIR:
    """LJ truncated at 12 Å with no charges, no solver and no exclusions."""
    return _nonbonded_ir(replace(_NO_METHOD, lj_cutoff=12.0), charges=False)


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

    def test_half_barrier_k_scales_to_openmm_units(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[1.0]]),
                periodicity=torch.tensor([1], dtype=torch.long),
                phase=torch.tensor([[0.0]]),
            )
        )
        spec = OpenMMAdapter().translate(ir)
        torsion = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
        # k is already the half barrier: 1 * 4.184 kJ/mol
        assert torsion["parameters"][0]["k"] == 4.184
        assert torsion["case"] == TranslationCase.FORM_REPARAMETERIZE.value

    def test_multi_term_proper_decomposes(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[1.0, 0.5]]),
                periodicity=torch.tensor([1, 2], dtype=torch.long),
                phase=torch.tensor([[0.0, math.pi]]),
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

    def test_scaling_is_the_shared_flat_spelling(self):
        """The spec carries the IR's scaling through the shared flattener.

        What the six factors *are* is pinned once, on the adapter base that
        owns the flat key spelling (``test_adapter.py``); this backend only has
        to route them.
        """
        ir = _class_i_ir()
        spec = OpenMMAdapter().translate(ir)
        assert spec.scaling == BackendAdapter.scaling_dict(ir.scaling)

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

    def test_pme_nonbonded_method_in_nanometres(self):
        """A lattice solver + switching window → PME with both radii in nm."""
        nb = next(
            f
            for f in OpenMMAdapter().translate(_nonbonded_ir(PME_SWITCHED_METHOD)).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["nonbondedMethod"] == "PME"
        assert nb["cutoffDistance"] == pytest.approx(1.2)  # 12 Å
        assert nb["ewaldErrorTolerance"] == 1e-6
        assert nb["useSwitchingFunction"] is True
        assert nb["switchingDistance"] == pytest.approx(1.0)  # 10 Å

    def test_nonbonded_record_states_the_lorentz_berthelot_rule_it_can_spell(self):
        """OpenMM's ``NonbondedForce`` mixes Lorentz–Berthelot, so this is a direct map.

        The rule is on the record whichever one the model ran, because a
        consumer building the OpenMM system has to know what the per-type
        ``sigma`` / ``epsilon`` it is handed mean — the same reason the record
        carries the cutoff it truncates at rather than leaving it implied.
        """
        nb = next(
            f
            for f in OpenMMAdapter().translate(_nonbonded_ir(PME_SWITCHED_METHOD)).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["mixing"] == "lorentz-berthelot"
        assert nb["case"] == TranslationCase.DIRECT_UNIT_SCALE.value

    def test_geometric_mixing_is_flagged_as_a_decomposition_with_a_note(self):
        """``NonbondedForce`` cannot spell geometric ``σ``, so it is not a direct map.

        OpenMM's built-in force combines ``σ`` arithmetically and offers no
        switch; reproducing an OPLS-AA model needs per-pair exceptions or a
        ``CustomNonbondedForce``. Emitting the parameters under the unchanged
        ``direct_unit_scale`` case would claim a faithful translation and
        silently build a different force field, so the case moves to
        ``decompose`` and the record carries the reason in text.
        """
        record = replace(PME_SWITCHED_METHOD, lj_mixing="geometric")
        nb = next(
            f
            for f in OpenMMAdapter().translate(_nonbonded_ir(record)).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["mixing"] == "geometric"
        assert nb["case"] == TranslationCase.DECOMPOSE.value
        assert isinstance(nb["mixing_note"], str)
        assert nb["mixing_note"].strip()

    def test_an_lj_bag_whose_record_names_no_rule_is_refused(self):
        """Fail closed: ``sigma`` / ``epsilon`` mean nothing without their rule.

        OpenMM's ``NonbondedForce`` applies its own Lorentz–Berthelot mixing to
        whatever per-type parameters it is handed, so exporting a bag whose rule
        was never recorded builds a system by *OpenMM's* convention and calls it
        the model's. The two agree only when the model happened to mix
        Lorentz–Berthelot, which is exactly the assumption this whole change
        exists to stop making.
        """
        record = replace(PME_SWITCHED_METHOD, lj_mixing=None)
        with pytest.raises(UnsupportedTermError, match="mixing rule"):
            OpenMMAdapter().translate(_nonbonded_ir(record))

    def test_a_charges_only_model_states_no_combining_rule(self):
        """No ``lj`` bag, no ``mixing`` key — and no refusal either.

        The fail-closed check above is about parameters that need a rule, not
        about the field being set: a charges-only model has no ``ε`` / ``σ`` and
        so has nothing to combine, which is why its record's ``lj_mixing`` is
        ``None`` and why that ``None`` must not be reported as a rule.
        """
        nb = next(
            f
            for f in OpenMMAdapter().translate(_charges_only_ir()).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["parameters"] == []
        assert "mixing" not in nb

    def test_open_boundary_cutoff_without_switch(self):
        """An open-boundary Coulomb cutoff is CutoffNonPeriodic, no taper."""
        record = replace(
            PME_SWITCHED_METHOD,
            lj_switch_start=None,
            coulomb_cutoff=12.0,
            coulomb_calculator=None,
            real_space_cutoff=None,
            smearing=None,
        )
        nb = next(
            f
            for f in OpenMMAdapter().translate(_nonbonded_ir(record)).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["nonbondedMethod"] == "CutoffNonPeriodic"
        assert nb["useSwitchingFunction"] is False
        assert nb["switchingDistance"] is None

    def test_lj_cutoff_alone_is_a_cutoff_method(self):
        """The method follows both pair terms, not only the Coulomb one.

        A Lennard-Jones-only model still runs at a cutoff, and reporting
        ``NoCutoff`` for it would build an OpenMM system that sums every pair
        in the box — a different energy function, quadratically more expensive.
        """
        nb = next(
            f
            for f in OpenMMAdapter().translate(_lj_only_ir()).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["nonbondedMethod"] == "CutoffNonPeriodic"
        assert nb["cutoffDistance"] == pytest.approx(1.2)  # 12 Å
        assert nb["useSwitchingFunction"] is False

    def test_no_declared_method_is_nocutoff(self):
        nb = next(
            f
            for f in OpenMMAdapter().translate(_nonbonded_ir(_NO_METHOD)).forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["nonbondedMethod"] == "NoCutoff"

    def test_ewald_error_tolerance_is_configurable(self):
        nb = next(
            f
            for f in OpenMMAdapter(ewald_error_tolerance=1e-5)
            .translate(_nonbonded_ir(PME_SWITCHED_METHOD))
            .forces
            if f["type"] == "NonbondedForce"
        )
        assert nb["ewaldErrorTolerance"] == 1e-5
