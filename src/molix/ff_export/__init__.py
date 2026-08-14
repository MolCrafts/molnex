"""Potential IR → backend force-spec, plus LAMMPS files via molpy.io.

Compile Class-I :class:`~molpot.ir.PotentialIR` into a :class:`ForceSpec`
via peer :class:`BackendAdapter` types. File export uses
:class:`LammpsWriter` → :mod:`molpy.io` (not a hand-written deck).
Unit factors come from :class:`molpy.UnitSystem`.

Load-bearing conversion goldens
-------------------------------
* Bond ``k = 100`` kcal mol⁻¹ Å⁻² → OpenMM ``41840`` kJ mol⁻¹ nm⁻²
* AMBER torsion ``Vn = 2`` kcal/mol → OpenMM PeriodicTorsion ``k = 4.184`` kJ/mol
* Bond ``k = 100`` kcal mol⁻¹ Å⁻² → LAMMPS ``K = 50`` (``E = K x²``)

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-08-ff-export
"""

from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.compiler import ForceFieldCompiler
from molix.ff_export.conventions import (
    ANGSTROM_TO_NM,
    BOND_K_IR_TO_OPENMM,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    ConventionRow,
    ConventionTable,
)
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.force_spec import ForceSpec
from molix.ff_export.lammps_adapter import LammpsAdapter
from molix.ff_export.lammps_writer import LammpsWriter
from molix.ff_export.openmm_adapter import OpenMMAdapter

# Register peer adapters on import.
_ = (LammpsAdapter, OpenMMAdapter)

__all__ = [
    "ANGSTROM_TO_NM",
    "BOND_K_IR_TO_OPENMM",
    "KCAL_PER_MOL_TO_KJ_PER_MOL",
    "BackendAdapter",
    "ConventionRow",
    "ConventionTable",
    "ForceFieldCompiler",
    "ForceSpec",
    "LammpsAdapter",
    "LammpsWriter",
    "OpenMMAdapter",
    "TranslationCase",
    "UnsupportedTermError",
]
