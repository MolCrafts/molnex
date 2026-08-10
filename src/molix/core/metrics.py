"""Metrics system for Molix Trainer.

This module provides a flexible metrics system compatible with torchmetrics.
Users can use built-in metrics or torchmetrics metrics interchangeably.

Example:
    ```python
    from molix.core.metrics import MAE, RMSE
    from molix.hooks.scalar import MetricsHook

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

    def compute(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute final metric value from accumulated state.

        Returns a 0-d :class:`torch.Tensor` left on the inputs' device (the
        torchmetrics contract). Materialising it to a Python ``float`` via
        ``.item()`` forces a CPU↔GPU sync, so callers do that only off the
        training hot path (the async journal copies the tensor instead).

        Returns:
            A 0-d metric tensor, or a dict of such tensors.
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
                return my_metric_fn(preds, targets)  # 0-d tensor, no .item()

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
    def compute(self) -> torch.Tensor:
        """Compute final metric value from accumulated state.

        Returns a 0-d tensor on the inputs' device — no ``.item()`` so the
        training hot path never blocks on a CPU↔GPU sync.
        """
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
        """Accumulate predictions and targets (kept on the original device
        — ``compute()``'s reduction + ``.item()`` is the single sync point)."""
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute MAE as a 0-d tensor on the inputs' device (no ``.item()``)."""
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean(torch.abs(preds - targets))

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
        """Accumulate predictions and targets (kept on the original device
        — ``compute()``'s reduction + ``.item()`` is the single sync point)."""
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute RMSE as a 0-d tensor on the inputs' device (no ``.item()``)."""
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.sqrt(torch.mean((preds - targets) ** 2))

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
        """Accumulate predictions and targets (kept on the original device
        — ``compute()``'s reduction + ``.item()`` is the single sync point)."""
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute MSE as a 0-d tensor on the inputs' device (no ``.item()``)."""
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds - targets) ** 2)

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
        """Accumulate predictions and targets (kept on the original device
        — ``compute()``'s reduction + ``.item()`` is the single sync point)."""
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute R² as a 0-d tensor on the inputs' device (no ``.item()``).

        The degenerate ``ss_tot == 0`` case (zero-variance targets) is
        resolved with :func:`torch.where` rather than a Python ``if`` so the
        method never forces a CPU↔GPU sync on the hot path.
        """
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

        return torch.where(ss_tot == 0, ss_tot.new_zeros(()), 1 - ss_res / ss_tot)

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
        """Accumulate predictions and targets (kept on the original device
        — ``compute()``'s reduction + ``.item()`` is the single sync point)."""
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute accuracy as a 0-d tensor on the inputs' device (no ``.item()``)."""
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds == targets).float())

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

        # Compute all metrics (0-d tensors; call float()/.item() off hot path)
        results = metrics.compute()  # {"MAE": tensor(0.5), "RMSE": tensor(0.7), ...}

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

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute all metrics, each a 0-d tensor on the inputs' device.

        No ``.item()`` is taken here — callers materialise to Python floats
        off the training hot path (the async journal copies the tensors).
        """
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


class MoleculeCenteredRMSE(BaseMetric):
    """RMSE after per-molecule mean-centering of pred and target (kcal/mol).

    Espaloma relative conformational energy metric. Call
    :meth:`update` with ``preds``, ``targets``, and integer ``group_ids``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> None:
        if group_ids is None:
            raise ValueError("MoleculeCenteredRMSE.update requires group_ids")
        from molix.core.losses.molecular import center_by_group

        p = center_by_group(preds.detach().reshape(-1), group_ids.detach().reshape(-1))
        t = center_by_group(targets.detach().reshape(-1), group_ids.detach().reshape(-1))
        self.preds.append(p)
        self.targets.append(t)

    def compute(self) -> torch.Tensor:
        if not self.preds:
            return torch.zeros(())
        p = torch.cat(self.preds)
        t = torch.cat(self.targets)
        return torch.sqrt(torch.mean((p - t) ** 2))

    def reset(self) -> None:
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []


class MoleculeCenteredMAE(BaseMetric):
    """MAE after per-molecule mean-centering of pred and target (kcal/mol)."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> None:
        if group_ids is None:
            raise ValueError("MoleculeCenteredMAE.update requires group_ids")
        from molix.core.losses.molecular import center_by_group

        p = center_by_group(preds.detach().reshape(-1), group_ids.detach().reshape(-1))
        t = center_by_group(targets.detach().reshape(-1), group_ids.detach().reshape(-1))
        self.preds.append(p)
        self.targets.append(t)

    def compute(self) -> torch.Tensor:
        if not self.preds:
            return torch.zeros(())
        p = torch.cat(self.preds)
        t = torch.cat(self.targets)
        return torch.mean((p - t).abs())

    def reset(self) -> None:
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
