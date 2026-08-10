"""molrep.perception — symbolic SMARTS/SMIRKS chemical-perception interface.

Binds discrete condensed classes (06) to SMARTS/SMIRKS patterns, matches them
via a :class:`SmartsMatcher` Protocol, and exposes :class:`SymbolicForceField`
for export/human inspection. Matching is pure perception — no energy evaluation.

Spec: learnable-classical-ff-07-smarts
"""

from molrep.perception.forcefield import SymbolicForceField
from molrep.perception.matcher import (
    FakeSmartsMatcher,
    MolpySmartsMatcher,
    SmartsMatcher,
)
from molrep.perception.patterns import SymbolicPattern
from molrep.perception.records import DiscreteClassRecord
from molrep.perception.registry import ClassPatternRegistry

__all__ = [
    "ClassPatternRegistry",
    "DiscreteClassRecord",
    "FakeSmartsMatcher",
    "MolpySmartsMatcher",
    "SmartsMatcher",
    "SymbolicForceField",
    "SymbolicPattern",
]
