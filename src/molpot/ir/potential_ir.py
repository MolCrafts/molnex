"""PotentialIR — aggregate Class-I interaction bags + unit system.

Missing bags are allowed (zero contribution from that term). The unit system
defaults to ``"class_i_canonical"``; unknown labels raise.

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-01-ir-kernels
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
from molpot.ir.scaling import NonbondedScaling

__all__ = ["PotentialIR", "KNOWN_UNIT_SYSTEMS"]

KNOWN_UNIT_SYSTEMS: frozenset[str] = frozenset({"class_i_canonical"})


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
        unit_system: Unit-system label. Default ``"class_i_canonical"``.
    """

    bonds: BondBag | None = None
    angles: AngleBag | None = None
    propers: ProperTorsionBag | None = None
    impropers_periodic: ImproperPeriodicBag | None = None
    impropers_harmonic: ImproperHarmonicBag | None = None
    lj: LJBag | None = None
    charges: ChargeBag | None = None
    scaling: NonbondedScaling | None = None
    unit_system: str = field(default="class_i_canonical")

    def __post_init__(self) -> None:
        if self.unit_system not in KNOWN_UNIT_SYSTEMS:
            raise ValueError(
                f"Unknown unit_system {self.unit_system!r}; "
                f"known systems: {sorted(KNOWN_UNIT_SYSTEMS)}"
            )
