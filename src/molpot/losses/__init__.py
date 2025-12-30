"""Loss functions.

All losses are TensorDictModules.
"""

from molpot.losses.losses import EnergyLoss, ForceLoss, CombinedLoss

__all__ = ["EnergyLoss", "ForceLoss", "CombinedLoss"]
