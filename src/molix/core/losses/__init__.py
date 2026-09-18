"""Loss functions for ML models.

Generic tensor-level losses:
- ``MSELoss``: Mean squared error loss
- ``MAELoss``: Mean absolute error (L1) loss
- ``WeightedLoss``: Weighted combination of multiple losses

Molecular-ML presets (``TensorDict``-aware closures):
- ``energy_mse(target_key)``: MSE on graph-level ``energy`` vs
  ``batch["graphs", target_key]``.
- ``energy_force_mse(lambda_F=...)``: joint energy + forces MSE.

Composable pieces the presets above are themselves built from — vary the
measure, the weight and the combination independently, with no molnex loss
class in the way::

    from torch import nn
    from molix.core.losses import WeightedLoss, energy_loss, force_loss

    # Note the measure: ``SmoothL1Loss(beta=d)`` is ``HuberLoss(delta=d) / d``.
    # Both are "a Huber"; they differ by a factor of ``d`` in the loss scale
    # and therefore in every gradient. State which one you mean.
    smooth = lambda beta: nn.SmoothL1Loss(beta=beta, reduction="none")
    loss = WeightedLoss([
        (1.0, energy_loss("energy_residual", pred_key="energy_mm",
                          elementwise=smooth(20.0), weight_key="mask")),
        (1.0, force_loss("forces", elementwise=smooth(20.0),
                         weight_key="force_weight")),
    ])

``atoms.force_weight`` is written by
:class:`~molix.data.ForceInlierWeight`, and ``graphs.energy_residual`` by
:class:`~molix.data.EnergyResidual` — the weight and the residual target are
batch columns, so the loss stays a pure function of what the batch carries.
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
    energy_loss,
    energy_mse,
    force_loss,
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
    "energy_loss",
    "energy_mse",
    "force_loss",
    "molecule_centered_energy_mse",
    "parameter_bag_mse",
]
