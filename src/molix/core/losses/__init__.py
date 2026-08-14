"""Loss functions for ML models.

Generic tensor-level losses:
- ``MSELoss``: Mean squared error loss
- ``MAELoss``: Mean absolute error (L1) loss
- ``WeightedLoss``: Weighted combination of multiple losses

Molecular-ML presets (``TensorDict``-aware closures):
- ``energy_mse(target_key)``: MSE on graph-level ``energy`` vs
  ``batch["graphs", target_key]``.
- ``energy_force_mse(lambda_F=...)``: joint energy + forces MSE.
"""

from molix.core.losses.combined import WeightedLoss
from molix.core.losses.energy import MSELoss
from molix.core.losses.force import MAELoss
from molix.core.losses.molecular import (
    ChargeMSELoss,
    ParsleyEnergyWeight,
    ProjectedInternalHessianLoss,
    ScaleNormalizedLoss,
    TorsionProfileLoss,
    center_by_group,
    energy_force_mse,
    energy_mse,
    molecule_centered_energy_mse,
    parameter_bag_mse,
)

__all__ = [
    "ChargeMSELoss",
    "MAELoss",
    "MSELoss",
    "ParsleyEnergyWeight",
    "ProjectedInternalHessianLoss",
    "ScaleNormalizedLoss",
    "TorsionProfileLoss",
    "WeightedLoss",
    "center_by_group",
    "energy_force_mse",
    "energy_mse",
    "molecule_centered_energy_mse",
    "parameter_bag_mse",
]
