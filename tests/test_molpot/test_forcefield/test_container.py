"""OPLSAAForceField inherits ForceField; product templates are named subclasses."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from molpy import ForceField as MolpyForceField
from tensordict import TensorDict

from molpot.forcefield import ForceField, OPLSAAForceField
from molpot.ir import ExclusionScales, NonbondedScaling
from molpot.manifest import ModelManifest, NumericsRecord, ParamTableRecord


def _realized(n_atoms: int) -> ForceField:
    """Force field with identity atom types realized over ``n_atoms`` atoms."""
    ff = ForceField(name="class-i", units="real").def_atomstyle("full")
    pos = torch.arange(n_atoms, dtype=torch.float64).unsqueeze(1) * torch.tensor(
        [1.53, 0.0, 0.0], dtype=torch.float64
    )
    system = TensorDict({"atoms": TensorDict({"pos": pos}, batch_size=[n_atoms])}, batch_size=[])
    return ff.realize(system)


class TestForceField:
    def test_identity_types_are_labelled_for_lammps(self):
        """Identity type names start with a letter: ``t1``, ``t2``, …

        They leave molnex as LAMMPS *type labels*, and a type label may not
        begin with a digit — ``read_data`` refuses a bare ``1``.
        """
        from molpy import AtomType

        assert ForceField.identity_type_label(0, 2) == "t1"
        assert [t.name for t in _realized(2).get_types(AtomType)] == ["t1", "t2"]

    def test_identity_labels_are_zero_padded_to_the_atom_count(self):
        """Ten atoms label as ``t01``…``t10``, so text order is atom order.

        The labels are sorted as text downstream (molrs writes the type tables
        in that order), and unpadded ``t10`` sorts before ``t2`` — which
        renumbers the types against the atoms they came from and corrupts every
        coeff line that follows.
        """
        from molpy import AtomType

        assert ForceField.identity_type_label(0, 10) == "t01"
        names = [t.name for t in _realized(10).get_types(AtomType)]
        assert names == [f"t{i:02d}" for i in range(1, 11)]
        assert sorted(names) == names

    def test_nonbonded_scaling_is_mirrored_into_molpy_special_bonds(self):
        """The exclusion policy reaches molpy, which is what writes the deck.

        molpy formats the ``special_bonds`` line from its own arrays, so a
        policy that lived only on the Python side would export an unscaled 1–4
        (``1.0``) — the most expensive silent error a Class-I deck can carry.
        """
        opls = NonbondedScaling(
            q=ExclusionScales(0.0, 0.0, 0.5),
            lj=ExclusionScales(0.0, 0.0, 0.5),
        )
        ff = ForceField(name="x", units="real", nonbonded_scaling=opls)
        assert list(ff.special_bonds_lj) == [0.0, 0.0, 0.5]
        assert list(ff.special_bonds_coul) == [0.0, 0.0, 0.5]

        ff.nonbonded_scaling = NonbondedScaling()  # AMBER: q 1–4 = 5/6
        assert ff.special_bonds_coul[2] == pytest.approx(5.0 / 6.0)
        assert ff.special_bonds_lj[2] == 0.5

        clone = ff.clone()
        assert list(clone.special_bonds_lj) == list(ff.special_bonds_lj)
        assert list(clone.special_bonds_coul) == list(ff.special_bonds_coul)

    def test_setting_molpy_special_bonds_updates_the_scaling(self):
        """The mirror runs both ways, so the two spellings cannot drift apart."""
        ff = ForceField(name="x", units="real")  # AMBER default: q 1–4 = 5/6
        ff.set_special_bonds((0.0, 0.0, 0.5), (0.0, 0.0, 0.5))
        assert ff.nonbonded_scaling.triple("q") == (0.0, 0.0, 0.5)
        assert ff.nonbonded_scaling.triple("lj") == (0.0, 0.0, 0.5)


class TestOPLSAAForceField:
    def test_default_styles_are_oplsaa_set(self):
        from molpot.forcefield import OPLSAAForceField

        ff = OPLSAAForceField()
        assert isinstance(ff, ForceField)
        assert isinstance(ff, MolpyForceField)
        for key in (
            "bond:harmonic",
            "angle:harmonic",
            "dihedral:opls",
            "improper:periodic",
            "pair:lj/cut",
            "pair:coulomb",
        ):
            assert ff.has_style(key), key

    def test_default_nonbonded_scaling_f14_is_half(self):
        from molpot.forcefield import OPLSAAForceField

        scaling = OPLSAAForceField().nonbonded_scaling
        assert scaling.q.s14 == 0.5
        assert scaling.lj.s14 == 0.5


def _filled_oplsaa() -> OPLSAAForceField:
    """OPLS-AA template with the bond and per-atom nonbonded tables filled.

    Two of the six declared styles carry parameters; the manifest still has to
    describe all six, because a declared style with no table is an
    unparameterised model, not an absent term.
    """
    return OPLSAAForceField().with_params(
        bonds={
            "k": torch.tensor([340.0], dtype=torch.float64),
            "r0": torch.tensor([1.09], dtype=torch.float64),
        },
        atoms={
            "epsilon": torch.tensor([0.066, 0.030], dtype=torch.float64),
            "sigma": torch.tensor([3.50, 2.50], dtype=torch.float64),
            "q": torch.tensor([-0.12, 0.06], dtype=torch.float64),
        },
    )


class TestForceFieldManifest:
    """``ForceField.manifest`` — the S2 manifest's owner.

    The force field is where the styles, the style options, the parameter
    tables and the exclusion scaling all already live, so it fills the
    manifest; ``Potentials`` is only its torch view and owns no second copy.
    ``representation`` and ``provenance`` are left unset on purpose — the
    caller composes them in with ``dataclasses.replace``, because composition
    is the caller's job.
    """

    def test_interactions_are_the_energy_bearing_declared_styles(self):
        """Every declared style but ``atom:*``, which is a record layout."""
        ff = _filled_oplsaa()
        manifest = ff.manifest(profile="A0")
        assert isinstance(manifest, ModelManifest)
        assert [record.style for record in manifest.interactions] == [
            key for key in ff.style_names() if not key.startswith("atom:")
        ]

    def test_nonbonded_record_is_the_one_to_ir_reports(self):
        """One nonbonded-method record, shared with ``PotentialIR`` — not two."""
        ff = _filled_oplsaa()
        assert ff.manifest(profile="A0").nonbonded == ff.to_ir().nonbonded_method

    def test_electrostatics_carries_the_permanent_exclusion_scaling(self):
        """S5: the permanent scaling is model identity, not a runtime detail."""
        ff = _filled_oplsaa()
        electrostatics = ff.manifest(profile="A0").electrostatics
        assert electrostatics is not None
        assert electrostatics.permanent_scaling == ff.nonbonded_scaling

    def test_electrostatics_reports_only_what_the_force_field_owns(self):
        """A ``pair:coulomb`` style is not a charge model, so it constrains nothing.

        The force field owns the multipole order (0 — point charges) and the
        permanent scaling. *How* the charges were obtained is the charge
        model's: ``AtomicChargeModel.electrostatics_record`` names the
        constraint (``"total-charge-kkt"``) and the solve it ran, and the
        caller composes the two with ``dataclasses.replace``. Claiming the KKT
        constraint here describes a solver this container never ran — and
        says it even for a field whose ``atoms.q`` came from a file.
        """
        electrostatics = _filled_oplsaa().manifest(profile="A0").electrostatics
        assert electrostatics is not None
        assert electrostatics.multipole_order == 0
        assert electrostatics.charge_constraint == "none"
        assert electrostatics.solver is None

    def test_a_bonded_only_field_still_carries_both_nonbonded_records(self):
        """Records are never ``None``: a consumer branches on a field, not presence.

        A field with no pair style at all gets an all-``None``
        ``NonbondedMethodRecord`` and an electrostatics record all the same,
        so reading ``manifest.nonbonded.lj_cutoff`` is always legal.
        """
        manifest = (
            ForceField(name="bonded", units="real").def_bondstyle("harmonic").manifest(profile="A0")
        )
        nonbonded = manifest.nonbonded
        assert nonbonded is not None
        assert [getattr(nonbonded, f.name) for f in dataclasses.fields(nonbonded)] == [None] * len(
            dataclasses.fields(nonbonded)
        )
        assert manifest.electrostatics is not None

    def test_profile_must_be_one_this_molnex_knows(self):
        """``A1`` is not a spelling choice — the polarization records do not exist.

        Accepted verbatim, the string would label a manifest as a model
        profile whose induced scaling and response solver this schema has no
        fields for, and the mislabelling would only surface in whatever reads
        it back. The refusal names the profile asked for and the ones there
        are.
        """
        ff = _filled_oplsaa()
        with pytest.raises(ValueError) as excinfo:
            ff.manifest(profile="A1")
        assert "A1" in str(excinfo.value)
        assert "A0" in str(excinfo.value)
        ff.manifest(profile="A0")

    def test_numerics_is_the_runtime_record_rather_than_a_second_reading(self):
        """One reader of the global precision, and the force field defers to it.

        What ``NumericsRecord.from_runtime`` reports is pinned in
        ``tests/test_molpot/test_manifest.py``; this pins only that the
        container does not take its own reading of ``molix.config["ftype"]``
        and ``torch.get_float32_matmul_precision()``.
        """
        assert _filled_oplsaa().manifest(profile="A0").numerics == NumericsRecord.from_runtime()

    def test_tables_content_hash_the_filled_parameter_tables(self):
        """Each filled category is hashed by ``ParamTableRecord``, not re-derived."""
        ff = _filled_oplsaa()
        tables = {record.category: record for record in ff.manifest(profile="A0").tables}
        assert set(tables) == {"bonds", "atoms"}
        assert tables["bonds"] == ParamTableRecord.from_tables("bonds", ff.params("bonds"))

    def test_profile_is_recorded_and_composition_is_left_to_the_caller(self):
        """No façade fills representation / provenance; the caller replaces them."""
        manifest = _filled_oplsaa().manifest(profile="A0")
        assert manifest.profile == "A0"
        assert manifest.representation is None
        assert manifest.provenance is None
