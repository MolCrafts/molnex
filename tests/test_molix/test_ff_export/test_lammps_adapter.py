"""LammpsAdapter — Class-I IR → LAMMPS units real force-spec goldens."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from molix.ff_export import ForceFieldCompiler, LammpsAdapter
from molix.ff_export.exceptions import UnsupportedTermError
from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperPeriodicBag,
    LJBag,
    NonbondedMethodRecord,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
)
from tests.test_molix.test_ff_export.conftest import PME_SWITCHED_METHOD

#: Same method with no Coulomb solver of any kind — LJ hard-cut at 12 Å.
_NO_SOLVER = replace(
    PME_SWITCHED_METHOD,
    lj_switch_start=None,
    coulomb_cutoff=None,
    coulomb_calculator=None,
    real_space_cutoff=None,
    smearing=None,
    lr_wavelength=None,
    mesh_spacing=None,
    capacity_factor=None,
)


def _pair_ir(
    record: NonbondedMethodRecord,
    *,
    lj: bool = True,
    charges: bool = True,
    scaling: bool = True,
) -> PotentialIR:
    """Two-type nonbonded IR running the nonbonded method ``record`` describes."""
    return PotentialIR(
        lj=LJBag(
            epsilon=torch.tensor([0.066, 0.03], dtype=torch.float64),
            sigma=torch.tensor([3.5, 2.5], dtype=torch.float64),
        )
        if lj
        else None,
        charges=ChargeBag(charge=torch.tensor([0.5, -0.5], dtype=torch.float64))
        if charges
        else None,
        scaling=NonbondedScaling() if scaling else None,
        nonbonded_method=record,
    )


class TestLammpsAdapter:
    def test_bond_k_half_convention(self):
        """molnex E=½k dr² → LAMMPS K = k/2."""
        ir = PotentialIR(bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.5])))
        spec = LammpsAdapter().translate(ir)
        bond = next(f for f in spec.forces if f["category"] == "bond")
        assert bond["style"] == "harmonic"
        assert bond["parameters"][0]["K"] == 50.0
        assert bond["parameters"][0]["r0"] == 1.5
        assert bond["parameters"][0]["type_index"] == 1

    def test_angle_degrees(self):
        ir = PotentialIR(
            angles=AngleBag(
                k=torch.tensor([100.0]),
                theta0=torch.tensor([math.radians(109.5)]),
            )
        )
        spec = LammpsAdapter().translate(ir)
        ang = next(f for f in spec.forces if f["category"] == "angle")
        assert abs(ang["parameters"][0]["K"] - 50.0) < 1e-12
        assert abs(ang["parameters"][0]["theta0_deg"] - 109.5) < 1e-5

    def test_opls_manifold_dihedral(self):
        k = 0.5 * torch.tensor([[1.0, 2.0, 0.5, 0.25]])
        phase = torch.zeros(1, 4)
        phase[0, 1] = math.pi
        phase[0, 3] = math.pi
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=k,
                periodicity=torch.tensor([1, 2, 3, 4], dtype=torch.long),
                phase=phase,
            )
        )
        spec = LammpsAdapter().translate(ir)
        dih = next(f for f in spec.forces if f["category"] == "dihedral")
        assert dih["style"] == "opls"
        p = dih["parameters"][0]
        assert abs(p["K1"] - 1.0) < 1e-12
        assert abs(p["K2"] - 2.0) < 1e-12
        assert abs(p["K3"] - 0.5) < 1e-12
        assert abs(p["K4"] - 0.25) < 1e-12

    def test_non_opls_emits_fourier(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[0.4, 0.1]], dtype=torch.float64),
                periodicity=torch.tensor([1, 3], dtype=torch.long),
                phase=torch.zeros(1, 2, dtype=torch.float64),
            )
        )
        spec = LammpsAdapter().translate(ir)
        dih = next(f for f in spec.forces if f["category"] == "dihedral")
        assert dih["style"] == "fourier"
        assert dih["parameters"][0]["terms"][0]["periodicity"] == 1
        assert dih["parameters"][0]["terms"][0]["k"] == 0.4

    @staticmethod
    def _improper_record(k, periodicity, phase):
        ir = PotentialIR(
            impropers_periodic=ImproperPeriodicBag(
                k=torch.tensor(k, dtype=torch.float64),
                periodicity=torch.tensor(periodicity),
                phase=torch.tensor(phase, dtype=torch.float64),
            )
        )
        spec = LammpsAdapter().translate(ir)
        return next(f for f in spec.forces if f["category"] == "improper")

    def test_periodic_improper_is_cvff_with_its_phase_as_a_sign(self):
        """``k[1 + cos(nχ − γ)]`` is cvff ``K[1 + d cos(nχ)]``: γ = 0 → d = +1, γ = π → d = −1."""
        rec = self._improper_record([[1.1], [2.0]], [[2], [2]], [[math.pi], [0.0]])
        assert rec["style"] == "cvff"
        assert rec["parameters"] == [
            {"type_index": 1, "K": 1.1, "d": -1, "n": 2},
            {"type_index": 2, "K": 2.0, "d": 1, "n": 2},
        ]

    def test_a_zero_barrier_term_does_not_count(self):
        rec = self._improper_record([[1.1, 0.0]], [[2, 1]], [[0.0, 0.7]])
        assert rec["parameters"] == [{"type_index": 1, "K": 1.1, "d": 1, "n": 2}]

    def test_a_periodic_improper_cvff_cannot_hold_is_refused(self):
        """cvff is one term with a ±1 sign; anything else was written anyway."""
        with pytest.raises(UnsupportedTermError, match="improper_periodic"):
            self._improper_record([[1.1]], [[2]], [[0.5]])
        with pytest.raises(UnsupportedTermError, match="improper_periodic"):
            self._improper_record([[1.1, 0.3]], [[2, 1]], [[0.0, 0.0]])

    def test_compiler_name_lammps(self):
        ir = PotentialIR(bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.0])))
        spec = ForceFieldCompiler("lammps").compile(ir)
        assert spec.backend == "lammps"
        assert spec.metadata["units"] == "real"

    def test_periodic_switched_pair_kspace_and_exclusions(self):
        """PME + a switching window → hybrid/overlay, pppm, gewald, special_bonds.

        ``kspace_modify gewald`` states the Ewald splitting parameter
        ``alpha = 1/(smearing * sqrt(2))`` explicitly, so LAMMPS solves the
        same decomposition molnex ran instead of choosing its own from the
        requested accuracy.
        """
        spec = LammpsAdapter().translate(_pair_ir(PME_SWITCHED_METHOD))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["type"] == "pair_style hybrid/overlay lj/mdf 10 12 coul/long 15"
        assert pair["style"] == "hybrid/overlay"
        assert pair["sub_styles"] == [
            {"style": "lj/mdf", "args": [10.0, 12.0]},
            {"style": "coul/long", "args": [15.0]},
        ]
        assert pair["parameters"] == [
            {"type_index": 1, "epsilon": 0.066, "sigma": 3.5},
            {"type_index": 2, "epsilon": 0.03, "sigma": 2.5},
        ]
        assert pair["mixing"] == "lorentz-berthelot"
        kspace = next(f for f in spec.forces if f["category"] == "kspace")
        assert kspace["type"] == "kspace_style pppm 1e-06"
        assert kspace["style"] == "pppm"
        assert kspace["accuracy"] == 1e-6
        modify = next(f for f in spec.forces if f["category"] == "kspace_modify")
        assert modify["type"] == "kspace_modify gewald 0.1414214"
        assert modify["gewald"] == pytest.approx(1.0 / (5.0 * math.sqrt(2.0)))
        exclusions = next(f for f in spec.forces if f["category"] == "special_bonds")
        assert exclusions["type"] == "special_bonds lj 0 0 0.5 coul 0 0 0.8333333"

    def test_pair_record_carries_the_models_combining_rule(self):
        """The rule travels with the coefficients the writer will mix.

        ``pair_coeff i j`` lines are mixed by ``LammpsWriter`` from the
        per-type table above, so the record has to say *which* rule — the
        alternative is the writer re-deciding, which is how molnex and a
        LAMMPS deck came to describe different models on every cross pair
        without either side saying so (ledger finding **F24**). The default
        rule rides on the same field in the golden above, so a record that
        hard-coded ``"lorentz-berthelot"`` fails here.
        """
        record = replace(PME_SWITCHED_METHOD, lj_mixing="geometric")
        spec = LammpsAdapter().translate(_pair_ir(record))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["mixing"] == "geometric"

    def test_an_lj_bag_whose_record_names_no_rule_is_refused(self):
        """Fail closed: the deck cannot be mixed by a rule nobody declared.

        ``write_pair_block`` states every ``pair_coeff i j`` explicitly and
        mixes them itself, so a record with an ``lj`` bag but no rule leaves the
        writer to choose one. Choosing means exporting a force field the model
        never described — and it would be *this* export, the one whose whole
        purpose is to be compared against LAMMPS, that silently disagreed.
        Refusing names the field instead.
        """
        record = replace(PME_SWITCHED_METHOD, lj_mixing=None)
        with pytest.raises(UnsupportedTermError, match="mixing rule"):
            LammpsAdapter().translate(_pair_ir(record))

    def test_a_charges_only_model_states_no_combining_rule(self):
        """No ``lj`` bag, nothing to combine — the key is absent, not ``None``.

        A charges-only model declares no ``pair:lj/cut`` style, so its record's
        ``lj_mixing`` is ``None`` and there are no ``pair_coeff`` coefficients
        for a rule to apply to. Emitting the key anyway would put a rule (or a
        null) on a record that has no Lennard-Jones term, which the writer would
        then have to know to ignore.
        """
        record = replace(PME_SWITCHED_METHOD, lj_cutoff=None, lj_switch_start=None, lj_mixing=None)
        spec = LammpsAdapter().translate(_pair_ir(record, lj=False))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["parameters"] == []
        assert "mixing" not in pair

    def test_periodic_hard_cutoff_uses_single_style(self):
        """No switching window → the fused lj/cut/coul/long style, still with pppm."""
        record = replace(PME_SWITCHED_METHOD, lj_switch_start=None)
        spec = LammpsAdapter().translate(_pair_ir(record))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["type"] == "pair_style lj/cut/coul/long 12 15"
        assert pair["sub_styles"] == [{"style": "lj/cut/coul/long", "args": [12.0, 15.0]}]
        assert any(f["category"] == "kspace" for f in spec.forces)

    def test_open_boundary_truncates_coulomb_without_kspace(self):
        """An open-boundary Coulomb cutoff has no reciprocal-space solver."""
        record = replace(_NO_SOLVER, coulomb_cutoff=12.0)
        spec = LammpsAdapter().translate(_pair_ir(record))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["type"] == "pair_style lj/cut/coul/cut 12 12"
        assert not [f for f in spec.forces if f["category"].startswith("kspace")]

    def test_lj_only_switched_emits_no_coulomb_records(self):
        """Without charges there is no Coulomb sub-style and no exclusion policy."""
        record = replace(_NO_SOLVER, lj_switch_start=10.0)
        spec = LammpsAdapter().translate(_pair_ir(record, charges=False, scaling=False))
        pair = next(f for f in spec.forces if f["category"] == "pair")
        assert pair["type"] == "pair_style lj/mdf 10 12"
        assert not [f for f in spec.forces if f["category"] == "special_bonds"]
        assert not [f for f in spec.forces if f["category"] == "kspace"]

    def test_charges_without_solver_or_cutoff_unsupported(self):
        """No LAMMPS style is an untruncated open-boundary Coulomb sum."""
        with pytest.raises(UnsupportedTermError):
            LammpsAdapter().translate(_pair_ir(_NO_SOLVER))

    def test_lattice_solver_without_a_split_radius_is_rejected(self):
        """A lattice sum with no real-space radius has no ``coul/long`` to write.

        ``ForceField`` always fills both (the radius falls back to
        ``real_space_cutoff_factor * smearing``), so a record naming a solver
        but no radius is an inconsistent hand-built IR. Guessing a radius here
        would run LAMMPS at a different Ewald split than the model did, which
        no downstream test could attribute; the error names the missing field
        instead.
        """
        record = replace(PME_SWITCHED_METHOD, real_space_cutoff=None)
        with pytest.raises(UnsupportedTermError, match="real_space_cutoff"):
            LammpsAdapter().translate(_pair_ir(record))

    def test_kspace_accuracy_is_configurable(self):
        spec = LammpsAdapter(kspace_accuracy=1e-8).translate(_pair_ir(PME_SWITCHED_METHOD))
        kspace = next(f for f in spec.forces if f["category"] == "kspace")
        assert kspace["type"] == "kspace_style pppm 1e-08"
