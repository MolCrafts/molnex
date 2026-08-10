"""Physics-aware multi-system chemical class condensation.

Turn continuous per-interaction MM parameters into a discrete
:class:`TypeSystem` per :class:`InteractionClass` via greedy merge under
:class:`MergeCriterion` budgets. Optional physics gating uses an injected
``physical_eval`` callable — this package never builds molpot energy graphs
and never emits SMARTS/SMIRKS text (sub-spec 07).

Public surface:
    InteractionClass, MergeCriterion, default_criterion,
    TypeRecord, TypeSystem, UNMATCHED_TYPE_ID,
    ClassAssignment, PhysicalErrorMetrics,
    Condenser, CondensationResult,
    TypeSystemLabeler
"""

from molrep.condensation.assignment import ClassAssignment
from molrep.condensation.classes import InteractionClass
from molrep.condensation.condenser import CondensationResult, Condenser
from molrep.condensation.criterion import (
    MergeCriterion,
    angle_default_criterion,
    bond_default_criterion,
    charge_default_criterion,
    default_criterion,
    improper_default_criterion,
    lj_default_criterion,
    proper_default_criterion,
)
from molrep.condensation.labeler import TypeSystemLabeler
from molrep.condensation.metrics import PhysicalErrorMetrics
from molrep.condensation.type_system import UNMATCHED_TYPE_ID, TypeRecord, TypeSystem

__all__ = [
    "InteractionClass",
    "MergeCriterion",
    "default_criterion",
    "bond_default_criterion",
    "angle_default_criterion",
    "proper_default_criterion",
    "improper_default_criterion",
    "lj_default_criterion",
    "charge_default_criterion",
    "TypeRecord",
    "TypeSystem",
    "UNMATCHED_TYPE_ID",
    "ClassAssignment",
    "PhysicalErrorMetrics",
    "Condenser",
    "CondensationResult",
    "TypeSystemLabeler",
]
