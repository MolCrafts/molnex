"""Metrics system for Molix Trainer.

This module provides a flexible metrics system compatible with torchmetrics.
Users can use built-in metrics or torchmetrics metrics interchangeably.

Example:
    ```python
    from molix.core.metrics import MAE, RMSE
    from molix.core.hooks import MetricsHook

    # Use built-in metrics
    hook = MetricsHook(metrics=[MAE(), RMSE()])

    # Or mix with torchmetrics (if installed)
    from torchmetrics import R2Score
    hook = MetricsHook(metrics=[MAE(), R2Score()])
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch


class Metric(Protocol):
    """Protocol for metrics compatible with torchmetrics API.

    All metrics must implement three methods:
    - update(): Accumulate batch predictions and targets
    - compute(): Calculate final metric value from accumulated state
    - reset(): Clear internal state for new epoch

    This protocol is compatible with torchmetrics.Metric, enabling
    seamless interoperability between built-in and torchmetrics metrics.
    """

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with batch predictions and targets.

        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        ...

    def compute(self) -> float | dict[str, float]:
        """Compute final metric value from accumulated state.

        Returns:
            Metric value (float) or dict of metric values
        """
        ...

    def reset(self) -> None:
        """Reset metric state for new epoch."""
        ...


class BaseMetric(ABC):
    """Base class for built-in metrics with device handling.

    Provides common functionality for accumulating predictions/targets
    and managing device placement.

    Example:
        ```python
        class MyMetric(BaseMetric):
            def __init__(self):
                super().__init__()
                self.reset()

            def update(self, preds, targets):
                self.preds.append(preds.detach())
                self.targets.append(targets.detach())

            def compute(self):
                preds = torch.cat(self.preds)
                targets = torch.cat(self.targets)
                return my_metric_fn(preds, targets).item()

            def reset(self):
                self.preds = []
                self.targets = []
        ```
    """

    def __init__(self):
        """Initialize base metric."""
        self.device = torch.device("cpu")

    def to(self, device: str | torch.device) -> BaseMetric:
        """Move metric to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = torch.device(device)
        return self

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with batch predictions and targets."""
        ...

    @abstractmethod
    def compute(self) -> float:
        """Compute final metric value from accumulated state."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for new epoch."""
        ...


class MAE(BaseMetric):
    """Mean Absolute Error metric.

    Computes: mean(|predictions - targets|)

    Example:
        ```python
        metric = MAE()
        metric.update(preds, targets)
        mae = metric.compute()
        metric.reset()
        ```
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        """Compute MAE from accumulated predictions and targets."""
        if not self.preds:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean(torch.abs(preds - targets)).item()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class RMSE(BaseMetric):
    """Root Mean Squared Error metric.

    Computes: sqrt(mean((predictions - targets)²))

    Example:
        ```python
        metric = RMSE()
        metric.update(preds, targets)
        rmse = metric.compute()
        metric.reset()
        ```
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        """Compute RMSE from accumulated predictions and targets."""
        if not self.preds:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class MSE(BaseMetric):
    """Mean Squared Error metric.

    Computes: mean((predictions - targets)²)

    Example:
        ```python
        metric = MSE()
        metric.update(preds, targets)
        mse = metric.compute()
        metric.reset()
        ```
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        """Compute MSE from accumulated predictions and targets."""
        if not self.preds:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds - targets) ** 2).item()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class R2Score(BaseMetric):
    """R² (coefficient of determination) metric.

    Computes: 1 - (SS_res / SS_tot)
    where SS_res = sum((targets - predictions)²)
          SS_tot = sum((targets - mean(targets))²)

    Example:
        ```python
        metric = R2Score()
        metric.update(preds, targets)
        r2 = metric.compute()
        metric.reset()
        ```
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        """Compute R² from accumulated predictions and targets."""
        if not self.preds:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

        if ss_tot == 0:
            return 0.0

        return (1 - ss_res / ss_tot).item()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class Accuracy(BaseMetric):
    """Classification accuracy metric.

    Computes: mean(predictions == targets)

    For multi-class classification, predictions should be class indices
    (use argmax on logits before passing to metric).

    Example:
        ```python
        metric = Accuracy()
        # For logits, apply argmax first
        preds = logits.argmax(dim=-1)
        metric.update(preds, targets)
        acc = metric.compute()
        metric.reset()
        ```
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        """Compute accuracy from accumulated predictions and targets."""
        if not self.preds:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds == targets).float()).item()

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class MetricCollection:
    """Collection of metrics for convenient management.

    Groups multiple metrics and provides unified update/compute/reset interface.

    Example:
        ```python
        metrics = MetricCollection([MAE(), RMSE(), R2Score()])

        # Update all metrics
        metrics.update(preds, targets)

        # Compute all metrics
        results = metrics.compute()  # {"MAE": 0.5, "RMSE": 0.7, "R2Score": 0.9}

        # Reset all metrics
        metrics.reset()
        ```
    """

    def __init__(self, metrics: list[Metric] | dict[str, Metric]):
        """Initialize metric collection.

        Args:
            metrics: List of metrics or dict mapping names to metrics.
                    If list, metric names are inferred from class names.
        """
        if isinstance(metrics, dict):
            self.metrics = metrics
        else:
            self.metrics = {m.__class__.__name__: m for m in metrics}

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update all metrics with batch predictions and targets."""
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self) -> dict[str, float]:  # type: ignore[return]
        """Compute all metrics and return as dict."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: str | torch.device) -> MetricCollection:
        """Move all metrics to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        for metric in self.metrics.values():
            if hasattr(metric, "to"):
                metric.to(device)
        return self
