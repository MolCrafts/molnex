"""Generic MAE loss function."""

from typing import Any

import torch
import torch.nn as nn


class MAELoss(nn.Module):
    """Generic MAE (L1) loss with configurable keys.

    Computes mean absolute error between predicted and target values.
    Works with dictionaries and plain tensors.

    Args:
        pred_key: Key to extract from prediction dict/dataclass
        target_key: Key to extract from target dict/dataclass
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        reduction: str = "mean",
    ):
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        pred: Any,
        target: Any,
    ) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            pred: Predictions (dict with pred_key or tensor)
            target: Targets (dict with target_key or tensor)

        Returns:
            MAE loss
        """
        # Extract pred_val
        if isinstance(pred, dict):
            pred_val = pred[self.pred_key]
        elif hasattr(pred, self.pred_key):
            pred_val = getattr(pred, self.pred_key)
        else:
            pred_val = pred

        # Extract target_val
        if isinstance(target, dict):
            target_val = target[self.target_key]
        elif hasattr(target, self.target_key):
            target_val = getattr(target, self.target_key)
        else:
            target_val = target

        return self.l1(pred_val, target_val)
