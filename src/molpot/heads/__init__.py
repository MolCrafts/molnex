"""Prediction heads.

All heads are TensorDictModules with explicit in_keys/out_keys.
"""

from molpot.heads.heads import EnergyHead, ForceHead, TypeHead

__all__ = ["EnergyHead", "ForceHead", "TypeHead"]
