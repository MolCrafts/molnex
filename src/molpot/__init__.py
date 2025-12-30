"""MolPot: ML Potential Toolkit.

TensorDictModule-based components for molecular ML potentials.

Note: AtomicTD is now in molix.data (protocol-level infrastructure).
Import from molix.data instead of molpot.data.
"""

__version__ = "2.0.0"

# Prediction heads
from molpot.heads import EnergyHead, ForceHead, TypeHead

# Loss functions
from molpot.losses import EnergyLoss, ForceLoss, CombinedLoss

# Functional API (PyTorch-style)
from molpot import F

__all__ = [
    # Heads
    "EnergyHead",
    "ForceHead",
    "TypeHead",
    # Losses
    "EnergyLoss",
    "ForceLoss",
    "CombinedLoss",
    "F",  # Functional API
]
