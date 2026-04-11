"""Prediction heads.

All heads are plain PyTorch modules.
"""

from molpot.heads.heads import AtomicEnergyMLP, EnergyHead, TypeHead

__all__ = ["AtomicEnergyMLP", "EnergyHead", "TypeHead"]
