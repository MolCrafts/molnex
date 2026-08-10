"""molrep.chem — continuous chemical perception (no energy / no molpot).

Public surface:
    AtomChemEmbedding, BondChemEmbedding, context builders, ChemEmbeddings,
    ChemEncoder.
"""

from molrep.chem.context import (
    AngleContext,
    BondContext,
    ImproperContext,
    ProperContext,
)
from molrep.chem.embed import AtomChemEmbedding, BondChemEmbedding
from molrep.chem.encoder import ChemEncoder
from molrep.chem.features import ChemEmbeddings
from molrep.chem.typing_metrics import TypingRecoveryMetrics, TypingRecoveryReport
from molrep.chem.typing_probe import AtomTypeReadout

__all__ = [
    "AtomChemEmbedding",
    "BondChemEmbedding",
    "BondContext",
    "AngleContext",
    "ProperContext",
    "ImproperContext",
    "ChemEmbeddings",
    "ChemEncoder",
    "AtomTypeReadout",
    "TypingRecoveryMetrics",
    "TypingRecoveryReport",
]
