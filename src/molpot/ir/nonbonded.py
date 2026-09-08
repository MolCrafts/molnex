"""NonbondedMethodRecord — which nonbonded method a model actually ran (S2).

A style's :class:`~molpot.contract.InteractionContract` says what a term *is*;
it deliberately cannot say which solver ran. ``pair:coulomb`` declares
:attr:`~molpot.contract.CutoffKind.EWALD` as a *policy* label spanning both the
open-boundary truncation and the lattice sum, because the same style is
truncated at a cutoff without a cell and lattice-summed with one. The cutoff
radii, the switching window and the reciprocal-space solver with its sizing are
therefore per-model runtime state, recorded here and carried by
:attr:`~molpot.ir.PotentialIR.nonbonded_method` and by the model manifest.
``S2`` (the requirement that a model carry a versioned manifest) and ``F7`` (the
finding that a style contract cannot name the solver) are entries of the notes
file cited below; the record itself landed with the completed
``composable-a0-04-manifest`` spec.

The values are the ones
:meth:`~molpot.forcefield.potentials.Potentials._assemble_pair_potentials`
will use: a style option where one was given, and otherwise the assembly
default from :data:`~molpot.forcefield.potentials.PAIR_STYLE_DEFAULTS`. Every
field is optional, because a bonded-only force field declares no nonbonded
method at all — such a model still gets an all-``None`` record rather than no
record, so a consumer branches on a field, never on the record's presence.

References:
    Ewald, P. P. "Die Berechnung optischer und elektrostatischer
    Gitterpotentiale" Ann. Phys. 369, 253 (1921) — the lattice sum whose split
    the ``smearing`` / ``real_space_cutoff`` / reciprocal-space fields below
    size. https://doi.org/10.1002/andp.19213690304
    LAMMPS ``kspace_style`` command — how the reciprocal-space half is named on
    export.
    OpenMM User Guide §19 "Forces", ``NonbondedForce`` — the same split, with
    the cutoff and switching window attached to one force object.
    ``.claude/notes/composable-potentials.md`` §"S2 manifest" (record table).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NonbondedMethodRecord"]


@dataclass(frozen=True)
class NonbondedMethodRecord:
    """Cutoffs, switching window and Coulomb solver of one model.

    Frozen: a manifest is compared and hashed to decide whether two runs
    describe the same model, so a record that could be edited after the fact
    would lie about the run it came from.

    All nine fields are required — an omitted keyword would otherwise read as
    "unset" when it was merely forgotten, which is exactly the silent default
    this record exists to make visible.

    Several fields size an *Ewald decomposition*, the standard way to sum
    ``1/r`` electrostatics under periodic boundary conditions: each point
    charge is screened by a Gaussian of width :attr:`smearing`, leaving a
    short-ranged remainder that is summed pair-by-pair in real space out to
    :attr:`real_space_cutoff`, plus a smooth compensating charge distribution
    that is summed over reciprocal-space (Fourier) vectors. The two halves add
    back to the exact ``1/r`` sum; the split point is an accuracy/cost trade,
    not physics, which is why it is model identity and recorded here.

    The four sizing fields of that split — :attr:`smearing`,
    :attr:`lr_wavelength`, :attr:`mesh_spacing` and :attr:`capacity_factor` —
    are ``None`` exactly when no ``pair:coulomb`` style is declared.

    Attributes:
        lj_cutoff: Lennard-Jones cutoff radius in Å, or ``None`` when no
            ``pair:lj/cut`` style is declared.
        lj_switch_start: Inner radius in Å where the quintic switching taper
            begins. ``None`` means no window, i.e. a hard truncation at
            :attr:`lj_cutoff`.
        coulomb_cutoff: Open-boundary Coulomb truncation radius in Å, or
            ``None`` for an untruncated open sum.
        coulomb_calculator: Reciprocal-space algorithm used under a cell —
            ``"ewald"`` (every k-vector summed explicitly), ``"pme"`` or
            ``"p3m"`` (both on an FFT mesh). ``None`` when no ``pair:coulomb``
            style is declared.
        real_space_cutoff: Cutoff in Å of the real-space half of the Ewald
            split — the effective radius, so an undeclared one is reported as
            the derived
            ``PAIR_STYLE_DEFAULTS.real_space_cutoff_factor × smearing`` rather
            than as ``None``, which would name no radius while the term ran at
            one. ``None`` only when no ``pair:coulomb`` style is declared.
        smearing: Gaussian charge-splitting width in Å of the Ewald
            decomposition.
        lr_wavelength: Shortest reciprocal-space wavelength in Å resolved by
            the ``"ewald"`` sum; ignored by ``"pme"`` / ``"p3m"``.
        mesh_spacing: Grid spacing in Å of the ``"pme"`` / ``"p3m"`` mesh;
            ignored by ``"ewald"``.
        capacity_factor: Dimensionless head-room on the frozen reciprocal-space
            shape — the grid is sized for a cell expanded isotropically by this
            factor. It is an accuracy knob as much as a capacity one, so it is
            model identity, not a runtime convenience.
    """

    lj_cutoff: float | None
    lj_switch_start: float | None
    coulomb_cutoff: float | None
    coulomb_calculator: str | None
    real_space_cutoff: float | None
    smearing: float | None
    lr_wavelength: float | None
    mesh_spacing: float | None
    capacity_factor: float | None

    @property
    def is_lattice_sum(self) -> bool:
        """Whether the Coulomb term was summed over the lattice in reciprocal space.

        True exactly when :attr:`coulomb_calculator` is set, because every value
        it can take is a reciprocal-space algorithm: the vocabulary is
        ``ewald`` / ``pme`` / ``p3m``, and it is closed by
        :class:`~molpot.forcefield.pair_terms.CoulombPeriodicPotential`, which
        both types its ``calculator`` argument to those three literals and
        raises on anything else. There is therefore no "declared but not a
        lattice sum" state to test for — an open-boundary model leaves this
        field ``None`` and is truncated at :attr:`coulomb_cutoff` instead.

        The question is asked by every export target, each of which then needs
        its own reciprocal-space solver (LAMMPS' ``kspace_style``, OpenMM's
        ``PME`` — particle-mesh Ewald, the FFT-mesh flavour of the same
        decomposition). It is answered here, on the record that holds the evidence,
        rather than by a set of calculator names kept beside the adapters —
        which was a second place for the vocabulary to be listed, and so a
        second place for it to fall out of date with the potential.

        Returns:
            ``True`` for a periodic lattice sum, ``False`` for an open-boundary
            model or a bonded-only one.
        """
        return self.coulomb_calculator is not None
