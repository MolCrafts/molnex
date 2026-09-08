"""PotentialIR — aggregate Class-I interaction bags + unit system.

A *Class-I* force field is the classical diagonal form: harmonic bonds and
angles, a periodic proper torsion, an improper term, and the nonbonded
Lennard-Jones and Coulomb pair terms, with no cross terms coupling one internal
coordinate to another. A *bag* is this layer's plain parameter container for one
such term — a few named tensors, one row per interaction type, carrying no
kernel and no topology (:mod:`molpot.ir.bags`). ``IR`` is *intermediate
representation*: :class:`PotentialIR` is the whole model reduced to bags, which
is what the export adapters read instead of reaching into a live force field.

Missing bags are allowed (zero contribution from that term). The unit system
defaults to ``"class_i_canonical"`` — energy kcal/mol, length Å, charge in
elementary charges ``e``, angles rad — and an unknown label raises rather than
being carried through, since a bag whose units nobody agreed on cannot be
converted on the way out.

References:
    OpenMM User Guide §19 "Forces" — the force objects the OpenMM adapter maps
    these bags onto.
    Design note: ``.claude/notes/learnable-classical-ff.md`` (the
    ``learnable-classical-ff-01-ir-kernels`` spec it came from was completed and
    removed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from molpot.ir.bags import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    ProperTorsionBag,
)
from molpot.ir.nonbonded import NonbondedMethodRecord
from molpot.ir.scaling import NonbondedScaling

__all__ = ["PotentialIR", "KNOWN_UNIT_SYSTEMS", "no_nonbonded_method"]

#: Unit-system labels :class:`PotentialIR` accepts. One entry today —
#: ``"class_i_canonical"``: kcal/mol, Å, ``e``, rad. Adding a second is a format
#: decision rather than a detail, because every adapter picks its conversion
#: factors from this label alone.
KNOWN_UNIT_SYSTEMS: frozenset[str] = frozenset({"class_i_canonical"})


def no_nonbonded_method() -> NonbondedMethodRecord:
    """The record a model with no nonbonded method at all implies.

    Every field ``None``: no Lennard-Jones cutoff, no switching window, no
    combining rule, no Coulomb solver. It is the default of
    :attr:`PotentialIR.nonbonded_method`, which is why that attribute is never
    ``None`` — see its docstring for why the two spellings had to collapse into
    one.

    A fresh instance per call, though the record is frozen and could be shared:
    a module-level singleton compares equal to every other all-``None`` record
    anyway, so sharing buys nothing, while a mutable class attribute holding one
    is the kind of thing that gets subclassed.

    Returns:
        The all-``None`` record.
    """
    return NonbondedMethodRecord(
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


@dataclass
class PotentialIR:
    """Aggregate holding optional Class-I parameter bags and scaling.

    Attributes:
        bonds: Optional :class:`~molpot.ir.bags.BondBag`.
        angles: Optional :class:`~molpot.ir.bags.AngleBag`.
        propers: Optional :class:`~molpot.ir.bags.ProperTorsionBag`.
        impropers_periodic: Optional :class:`~molpot.ir.bags.ImproperPeriodicBag`.
        impropers_harmonic: Optional :class:`~molpot.ir.bags.ImproperHarmonicBag`.
        lj: Optional :class:`~molpot.ir.bags.LJBag`.
        charges: Optional :class:`~molpot.ir.bags.ChargeBag`.
        scaling: Optional :class:`~molpot.ir.scaling.NonbondedScaling`.
        nonbonded_method: Which nonbonded method the model runs — cutoffs in Å,
            switching window, Coulomb solver and its sizing
            (:class:`~molpot.ir.nonbonded.NonbondedMethodRecord`). The style
            contracts cannot carry it: ``pair:coulomb`` declares
            ``CutoffKind.EWALD`` as a policy label covering both the
            open-boundary truncation and the lattice sum, so the solver that
            ran is per-model state (ledger finding **F7** in
            ``.claude/notes/composable-potentials.md``).
            Never ``None``: an IR assembled by hand states no method, and a
            bonded-only force field has none to state, but those are the same
            thing — no cutoff, no switching window, no Coulomb solver — so both
            get the all-``None`` record :func:`no_nonbonded_method` builds.
            A consumer therefore branches on a *field* of the record and never
            on the record's presence.
        unit_system: Unit-system label the bags above are expressed in. Default
            ``"class_i_canonical"`` — kcal/mol, Å, ``e``, rad, as spelled out in
            the module docstring. Only labels in :data:`KNOWN_UNIT_SYSTEMS` are
            accepted.

    Raises:
        ValueError: ``unit_system`` is not in :data:`KNOWN_UNIT_SYSTEMS`.
            Checked at construction, because every adapter downstream scales
            these numbers by factors chosen from this label alone.
    """

    bonds: BondBag | None = None
    angles: AngleBag | None = None
    propers: ProperTorsionBag | None = None
    impropers_periodic: ImproperPeriodicBag | None = None
    impropers_harmonic: ImproperHarmonicBag | None = None
    lj: LJBag | None = None
    charges: ChargeBag | None = None
    scaling: NonbondedScaling | None = None
    nonbonded_method: NonbondedMethodRecord = field(default_factory=no_nonbonded_method)
    unit_system: str = field(default="class_i_canonical")

    def __post_init__(self) -> None:
        if self.unit_system not in KNOWN_UNIT_SYSTEMS:
            raise ValueError(
                f"Unknown unit_system {self.unit_system!r}; "
                f"known systems: {sorted(KNOWN_UNIT_SYSTEMS)}"
            )
