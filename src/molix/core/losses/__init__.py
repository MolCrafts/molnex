"""Generic loss functions for ML models.

Provides reusable, configurable loss functions:
- MSELoss: Mean squared error loss
- MAELoss: Mean absolute error (L1) loss
- WeightedLoss: Weighted combination of multiple losses
"""

from molix.core.losses.combined import WeightedLoss
from molix.core.losses.energy import MSELoss
from molix.core.losses.force import MAELoss

__all__ = [
    "MSELoss",
    "MAELoss",
    "WeightedLoss",
]
