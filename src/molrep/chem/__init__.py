"""molrep.chem — continuous chemical perception (no energy / no molpot).

Public surface:
    AtomChemEmbedding, BondChemEmbedding, ChemGraphBlock, context builders,
    ChemEmbeddings, ChemEncoder.
"""

from molrep.chem.blocks import ChemGraphBlock
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
    "AngleContext",
    "AtomChemEmbedding",
    "AtomTypeReadout",
    "BondChemEmbedding",
    "BondContext",
    "ChemEmbeddings",
    "ChemEncoder",
    "ChemGraphBlock",
    "ImproperContext",
    "ProperContext",
    "TypingRecoveryMetrics",
    "TypingRecoveryReport",
]
