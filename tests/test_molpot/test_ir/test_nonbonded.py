"""NonbondedMethodRecord — mirrors ``src/molpot/ir/nonbonded.py``.

The record names *which* nonbonded method actually ran: the Lennard-Jones
cutoff and its switching window, the open-boundary Coulomb cutoff, and — when
a cell is present — the reciprocal-space solver and its sizing parameters.
``molpot.forcefield.PotentialMeta`` deliberately cannot carry this: its
``pair:coulomb`` contract declares ``CutoffKind.EWALD`` as a *policy* label
covering both the open-boundary truncation and the lattice sum, so the solver
that ran is per-model runtime state (``composable-a0-04``, requirement S2).

Every field is optional because a bonded-only force field declares no
nonbonded method at all, and a caller must never have to branch on the
*record* being absent — only on its fields.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table).
"""

from __future__ import annotations

import dataclasses

import pytest

from molpot.ir.nonbonded import NonbondedMethodRecord


def _record(**overrides: float | str | None) -> NonbondedMethodRecord:
    """All-``None`` record with ``overrides`` applied — every field is required."""
    fields = {
        "lj_cutoff": None,
        "lj_switch_start": None,
        "lj_mixing": None,
        "coulomb_cutoff": None,
        "coulomb_calculator": None,
        "real_space_cutoff": None,
        "smearing": None,
        "lr_wavelength": None,
        "mesh_spacing": None,
        "capacity_factor": None,
    }
    fields.update(overrides)
    return NonbondedMethodRecord(**fields)


class TestNonbondedMethodRecord:
    def test_field_set_is_the_declared_method_surface(self) -> None:
        """The ten knobs ``Potentials._assemble_pair_potentials`` reads back.

        ``lj_mixing`` is one of them: the Lennard-Jones combining rule is model
        identity (a table mixed the other way is a different force field on
        every cross pair), so it is recorded here beside the radii rather than
        left to whichever kernel happens to run — ledger finding **F24**.
        """
        assert {f.name for f in dataclasses.fields(NonbondedMethodRecord)} == {
            "lj_cutoff",
            "lj_switch_start",
            "lj_mixing",
            "coulomb_cutoff",
            "coulomb_calculator",
            "real_space_cutoff",
            "smearing",
            "lr_wavelength",
            "mesh_spacing",
            "capacity_factor",
        }

    def test_all_none_is_a_valid_record(self) -> None:
        """A bonded-only field still gets a record — an empty one, not ``None``."""
        record = NonbondedMethodRecord(
            lj_cutoff=None,
            lj_switch_start=None,
            lj_mixing=None,
            coulomb_cutoff=None,
            coulomb_calculator=None,
            real_space_cutoff=None,
            smearing=None,
            lr_wavelength=None,
            mesh_spacing=None,
            capacity_factor=None,
        )
        assert record.lj_cutoff is None
        assert record.lj_mixing is None
        assert record.coulomb_calculator is None

    def test_is_lattice_sum_tracks_the_declared_solver(self) -> None:
        """True exactly when a reciprocal-space solver ran, i.e. under a cell.

        Consumers ask this instead of re-listing the solver names: an
        open-boundary model declares none, and every backend that has to choose
        between a truncated sum and a lattice sum reads the same predicate.
        """
        for calculator in ("ewald", "pme", "p3m"):
            assert _record(coulomb_calculator=calculator).is_lattice_sum, calculator
        assert not _record(coulomb_calculator=None).is_lattice_sum
        # The Lennard-Jones combining rule says nothing about the Coulomb
        # solver, so declaring one may not move this predicate either way.
        assert not _record(lj_mixing="geometric").is_lattice_sum

    def test_is_frozen(self) -> None:
        """Manifests are hashed and compared; a mutable record would lie."""
        record = NonbondedMethodRecord(
            lj_cutoff=10.0,
            lj_switch_start=None,
            lj_mixing="lorentz-berthelot",
            coulomb_cutoff=None,
            coulomb_calculator="pme",
            real_space_cutoff=None,
            smearing=5.0,
            lr_wavelength=1.0,
            mesh_spacing=1.0,
            capacity_factor=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.lj_cutoff = 12.0  # ty: ignore[invalid-assignment]
