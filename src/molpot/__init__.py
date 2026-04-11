"""MolPot: ML Potential Toolkit.

Pure PyTorch components for molecular ML potentials.
"""

# Potentials
from molpot.potentials import (
    BasePotential,
    LJ126,
    lorentz_berthelot,
    BondHarmonic,
    AngleHarmonic,
    DihedralHarmonic,
    RepulsionExp6,
    DispersionC6,
    ChargeTransfer,
    Polarization,
    geometric_arithmetic_mixing,
)

# Prediction heads
from molpot.heads import AtomicEnergyMLP, EnergyHead, TypeHead

# Physical derivation
from molpot.derivation import EnergyAggregation, ForceDerivation, StressDerivation

# Pooling
from molpot.pooling import (
    LayerPooling,
    EdgeToNodePooling,
    SumPooling,
    MeanPooling,
    MaxPooling,
)

# Composition
from molpot.composition import (
    LJParameterHead,
    RepulsionParameterHead,
    ChargeTransferParameterHead,
    ChargeHead,
    TSScalingHead,
    MultiHead,
    PotentialComposer,
)

__all__ = [
    # Potentials
    "BasePotential",
    "LJ126",
    "lorentz_berthelot",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralHarmonic",
    "RepulsionExp6",
    "DispersionC6",
    "ChargeTransfer",
    "Polarization",
    "geometric_arithmetic_mixing",
    # Heads
    "AtomicEnergyMLP",
    "EnergyHead",
    "TypeHead",
    # Derivation
    "EnergyAggregation",
    "ForceDerivation",
    "StressDerivation",
    # Pooling
    "LayerPooling",
    "EdgeToNodePooling",
    "SumPooling",
    "MeanPooling",
    "MaxPooling",
    # Composition
    "LJParameterHead",
    "RepulsionParameterHead",
    "ChargeTransferParameterHead",
    "ChargeHead",
    "TSScalingHead",
    "MultiHead",
    "PotentialComposer",
]
