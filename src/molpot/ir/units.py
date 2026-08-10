"""Canonical unit tags for the Class-I Potential Intermediate Representation.

Internal Class-I units (SI-free): kcal/mol, angstrom, elementary charge e,
radians. Downstream export (kJ/mol·nm) is owned by a later sub-spec.

References:
    OpenMM User Guide §19 "Forces"
    Cornell et al., JACS 1995 DOI 10.1021/ja00124a002 (AMBER)
"""

from enum import Enum
from types import MappingProxyType

__all__ = ["UnitTag", "CLASS_I_CANONICAL"]


class UnitTag(Enum):
    """Physical dimensions named by the Potential IR."""

    energy = "energy"
    length = "length"
    charge = "charge"
    angle = "angle"
    force_const_bond = "force_const_bond"
    force_const_angle = "force_const_angle"
    torsion_barrier = "torsion_barrier"


CLASS_I_CANONICAL: MappingProxyType[str, str] = MappingProxyType(
    {
        "energy": "kcal/mol",
        "length": "angstrom",
        "charge": "e",
        "angle": "radian",
    }
)
