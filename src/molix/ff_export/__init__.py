"""Potential IR → backend force-spec export (OpenMM-first).

Compile Class-I :class:`~molpot.ir.PotentialIR` parameter bags into a
backend-neutral :class:`ForceSpec`, with peer :class:`BackendAdapter` types
for unit/form translation. No live OpenMM import is required for tests —
goldens are hard-coded numbers and JSON force-spec dicts.

Load-bearing conversion goldens
-------------------------------
* Bond ``k = 100`` kcal mol⁻¹ Å⁻² → ``41840`` kJ mol⁻¹ nm⁻²
* AMBER torsion ``Vn = 2`` kcal/mol → OpenMM PeriodicTorsion ``k = 4.184`` kJ/mol

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
    scale_amber_vn,
    scale_angle_k,
    scale_bond_k,
    scale_energy,
    scale_length,
    scale_torsion_k,
)
from molix.ff_export.exceptions import UnsupportedTermError
from molix.ff_export.force_spec import ForceSpec
from molix.ff_export.openmm_adapter import OpenMMAdapter

__all__ = [
    "ANGSTROM_TO_NM",
    "BOND_K_IR_TO_OPENMM",
    "KCAL_PER_MOL_TO_KJ_PER_MOL",
    "BackendAdapter",
    "ConventionRow",
    "ConventionTable",
    "ForceFieldCompiler",
    "ForceSpec",
    "OpenMMAdapter",
    "TranslationCase",
    "UnsupportedTermError",
    "scale_amber_vn",
    "scale_angle_k",
    "scale_bond_k",
    "scale_energy",
    "scale_length",
    "scale_torsion_k",
]
