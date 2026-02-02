"""MolPot: ML Potential Toolkit.

TensorDictModule-based components for molecular ML potentials.

Note: AtomTD is now in molix.data (protocol-level infrastructure).
Import from molix.data instead of molpot.data.

Loss functions have been moved to molix.core.losses.
"""

# Classic potentials
from molpot.potentials import BasePotential, LJ126, BondHarmonic, AngleHarmonic, DihedralHarmonic

# Prediction heads
from molpot.heads import EnergyHead, ForceHead, TypeHead

# Readout operations
from molpot.readout import SumPooling, MeanPooling

__all__ = [
    # Potentials
    "BasePotential",
    "LJ126",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralHarmonic",
    # Heads
    "EnergyHead",
    "ForceHead",
    "TypeHead",
    # Readout
    "SumPooling",
    "MeanPooling",
]
