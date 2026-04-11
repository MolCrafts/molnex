"""Generic weighted combination of multiple losses."""

from typing import Any

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Computes L = Σ w_i * L_i(pred, target) for multiple loss terms.
    Each loss term is a (weight, loss_fn) tuple.

    Args:
        losses: List of (weight, loss_fn) tuples
    """

    def __init__(
        self,
        losses: list[tuple[float, nn.Module]],
    ):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for _, loss_fn in losses])
        self.weights = torch.tensor([weight for weight, _ in losses])

    def forward(
        self,
        pred: Any,
        target: Any,
    ) -> torch.Tensor:
        """Compute weighted combination of losses.

        Args:
            pred: Predictions (dict/dataclass or tensor)
            target: Targets (dict/dataclass or tensor)

        Returns:
            Weighted sum of all losses
        """
        device = self._get_device(pred)
        total_loss = torch.tensor(0.0, device=device)
        self.weights = self.weights.to(device)

        for weight, loss_fn in zip(self.weights, self.losses):
            try:
                loss_val = loss_fn(pred, target)
                total_loss = total_loss + weight * loss_val
            except (KeyError, AttributeError):
                # Skip if keys don't exist in pred/target
                continue

        return total_loss

    def _get_device(self, pred: Any) -> torch.device:
        """Get device from first available tensor."""
        if isinstance(pred, torch.Tensor):
            return pred.device

        if isinstance(pred, dict):
            for value in pred.values():
                if isinstance(value, torch.Tensor):
                    return value.device

        # Dataclass or other object
        for attr in dir(pred):
            if not attr.startswith("_"):
                val = getattr(pred, attr)
                if isinstance(val, torch.Tensor):
                    return val.device

        return torch.device("cpu")
