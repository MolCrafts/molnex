"""MolPot: Componentized ML Potential Toolkit.

MolPot provides reusable building blocks for molecular machine learning potentials:
- Data structures (AtomicTD with TensorDict)
- Graph construction (pure PyTorch)
- Feature engineering (RBF, cutoffs, geometry)
- Neural network primitives (MLP, normalization)
- Readout operations (pooling, aggregation)
- Prediction heads (energy, forces)
- Loss functions (energy, force, combined)
- Model implementations (PiNet2, etc.)

Design principles:
- Component-first architecture (maximize reuse)
- Pure PyTorch (no PyTorch Geometric dependency)
- Clear boundaries (model/loss/trainer separation)
- Full testability
"""

__version__ = "0.1.0"

# Core data structures
from molpot.data.atomic_td import AtomicTD, AtomicCollator

# Graph construction
from molpot.graph.radius_graph import radius_graph

# Feature engineering
from molpot.feats.rbf import GaussianRBF
from molpot.feats.cutoff import CosineCutoff

# Neural network primitives
from molpot.nn.mlp import MLP

# Readout operations
from molpot.readout.pooling import SumPooling, MeanPooling

# Prediction heads
from molpot.heads.energy import EnergyHead

# Loss functions
from molpot.losses.energy import EnergyLoss

# Classic potentials
from molpot.potentials import (
    LJ126,
    BondHarmonic,
    AngleHarmonic,
    DihedralHarmonic,
)

# ML potentials
from molpot.models.pinet2 import PiNet

__all__ = [
    # Data
    "AtomicTD",
    "collate_atomic",
    # Graph
    "radius_graph",
    # Features
    "GaussianRBF",
    "CosineCutoff",
    # NN
    "MLP",
    # Readout
    "SumPooling",
    "MeanPooling",
    # Heads
    "EnergyHead",
    # Losses
    "EnergyLoss",
    # Potentials (Classic)
    "LJ126",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralHarmonic",
    # Potentials (ML)
    "PiNet",
]
