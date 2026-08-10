"""MolPot: ML Potential Toolkit.

Pure PyTorch components for molecular ML potentials.
"""

# Potentials
# Composition
from molpot.composition import (
    ChargeHead,
    ChargeTransferParameterHead,
    LJParameterHead,
    MultiHead,
    PotentialComposer,
    RepulsionParameterHead,
    Sonata,
    SonataSpec,
    TSScalingHead,
)

# Physical derivation
from molpot.derivation import EnergyAggregation, ForceDerivation, StressDerivation

# Prediction heads
from molpot.heads import AtomicEnergyMLP, EnergyHead, TypeHead

# Class-I Potential IR
from molpot.ir import (
    CLASS_I_CANONICAL,
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
    UnitTag,
)

# Pooling
from molpot.pooling import (
    EdgeToNodePooling,
    LayerPooling,
    MaxPooling,
    MeanPooling,
    SumPooling,
)
from molpot.potentials import (
    LJ126,
    AngleHarmonic,
    BasePotential,
    BondHarmonic,
    ChargeTransfer,
    DihedralHarmonic,
    DispersionC6,
    ImproperHarmonic,
    ImproperPeriodic,
    Polarization,
    ProperTorsionPeriodic,
    RepulsionExp6,
    geometric_arithmetic_mixing,
    lorentz_berthelot,
)

__all__ = [
    # Potentials
    "BasePotential",
    "LJ126",
    "lorentz_berthelot",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralHarmonic",
    "ProperTorsionPeriodic",
    "ImproperPeriodic",
    "ImproperHarmonic",
    "RepulsionExp6",
    "DispersionC6",
    "ChargeTransfer",
    "Polarization",
    "geometric_arithmetic_mixing",
    # Potential IR
    "UnitTag",
    "CLASS_I_CANONICAL",
    "BondBag",
    "AngleBag",
    "ProperTorsionBag",
    "ImproperPeriodicBag",
    "ImproperHarmonicBag",
    "LJBag",
    "ChargeBag",
    "NonbondedScaling",
    "PotentialIR",
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
    "Sonata",
    "SonataSpec",
]
