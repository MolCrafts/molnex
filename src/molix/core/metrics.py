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
from typing import Any, Protocol

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

    def to(self, device: str | torch.device) -> Any:
        """Move the metric's state to *device* (``torchmetrics.Metric.to``)."""
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

    def sync(self, device: torch.device | str | None = None) -> None:
        """Reduce the accumulated state across ranks before :meth:`compute`; none by default.

        A torchmetrics metric is not a :class:`BaseMetric` and syncs itself
        inside its own ``compute``.
        """

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


def _distributed_world() -> int:
    """Process-group size, or 1 when this process is not in one."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _all_reduce_sums(*totals: torch.Tensor) -> bool:
    """Sum ``totals`` across the process group.

    Returns:
        False when there is nothing to reduce (one process). The caller
        then keeps its local buffers. True after an in-place sum.
    """
    if _distributed_world() < 2:
        return False
    import torch.distributed as dist

    for total in totals:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return True


def _mean_of(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """``total / count``, and zero when ``count`` is zero, without a host sync."""
    return torch.where(count == 0, total.new_zeros(()), total / count)


def _sync_pair(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    kind: str,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """All-reduce one ``(total, count)`` pair built from stored predictions.

    ``kind`` is ``"abs"``, ``"square"`` or ``"match"``. Returns ``None`` when
    this process is not in a group of two or more, so the caller keeps its
    lists and :meth:`compute` stays local.
    """
    if preds:
        pred = torch.cat(preds)
        target = torch.cat(targets)
        if kind == "abs":
            total = (pred - target).abs().to(torch.float64).sum()
        elif kind == "square":
            total = (pred - target).square().to(torch.float64).sum()
        elif kind == "match":
            total = (pred == target).to(torch.float64).sum()
        else:
            raise ValueError(f"kind must be 'abs', 'square' or 'match', got {kind!r}")
        count = torch.tensor(float(pred.numel()), device=total.device, dtype=torch.float64)
    else:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        total = torch.zeros((), device=dev, dtype=torch.float64)
        count = torch.zeros((), device=dev, dtype=torch.float64)
    if not _all_reduce_sums(total, count):
        return None
    return total, count


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
        reduced = self._reduced
        if reduced is not None:
            return _mean_of(*reduced)
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean(torch.abs(preds - targets))

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the absolute error so :meth:`compute` is the global mean."""
        reduced = _sync_pair(self.preds, self.targets, kind="abs", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            mean = _mean_of(*reduced)
            return torch.sqrt(mean)
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.sqrt(torch.mean((preds - targets) ** 2))

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the squared error so :meth:`compute` is the global RMSE."""
        reduced = _sync_pair(self.preds, self.targets, kind="square", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            return _mean_of(*reduced)
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds - targets) ** 2)

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the squared error so :meth:`compute` is the global MSE."""
        reduced = _sync_pair(self.preds, self.targets, kind="square", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            ss_res, sum_t, sum_t2, count = reduced
            ss_tot = sum_t2 - sum_t.square() / count.clamp(min=1)
            r2 = 1 - ss_res / ss_tot
            return torch.where((count == 0) | (ss_tot == 0), ss_res.new_zeros(()), r2)
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

        return torch.where(ss_tot == 0, ss_tot.new_zeros(()), 1 - ss_res / ss_tot)

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the R² sufficient statistics and replace the local lists."""
        if self.preds:
            preds = torch.cat(self.preds).to(torch.float64)
            targets = torch.cat(self.targets).to(torch.float64)
            ss_res = (targets - preds).square().sum()
            sum_t = targets.sum()
            sum_t2 = targets.square().sum()
            count = ss_res.new_tensor(float(targets.numel()))
        else:
            dev = torch.device(device) if device is not None else torch.device("cpu")
            ss_res = torch.zeros((), device=dev, dtype=torch.float64)
            sum_t = torch.zeros((), device=dev, dtype=torch.float64)
            sum_t2 = torch.zeros((), device=dev, dtype=torch.float64)
            count = torch.zeros((), device=dev, dtype=torch.float64)
        if not _all_reduce_sums(ss_res, sum_t, sum_t2, count):
            return
        self._reduced = (ss_res, sum_t, sum_t2, count)
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            return _mean_of(*reduced)
        if not self.preds:
            return torch.zeros(())
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds == targets).float())

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce matches so :meth:`compute` is the global accuracy."""
        reduced = _sync_pair(self.preds, self.targets, kind="match", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            return torch.sqrt(_mean_of(*reduced))
        if not self.preds:
            return torch.zeros(())
        p = torch.cat(self.preds)
        t = torch.cat(self.targets)
        return torch.sqrt(torch.mean((p - t) ** 2))

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the centered squared error."""
        reduced = _sync_pair(self.preds, self.targets, kind="square", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


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
        reduced = self._reduced
        if reduced is not None:
            return _mean_of(*reduced)
        if not self.preds:
            return torch.zeros(())
        p = torch.cat(self.preds)
        t = torch.cat(self.targets)
        return torch.mean((p - t).abs())

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce the centered absolute error."""
        reduced = _sync_pair(self.preds, self.targets, kind="abs", device=device)
        if reduced is None:
            return
        self._reduced = reduced
        self.preds = []
        self.targets = []

    def reset(self) -> None:
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self._reduced: tuple[torch.Tensor, torch.Tensor] | None = None


class _SumCount:
    """A sum and a count on one device, reduced by :meth:`PotentialReport.sync`."""

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def add(self, total: torch.Tensor, count: torch.Tensor | int) -> None:
        piece = total.detach().to(torch.float64).reshape(())
        if torch.is_tensor(count):
            seen = count.detach().to(device=piece.device, dtype=torch.float64).reshape(())
        else:
            seen = piece.new_tensor(float(count))
        if self.total is None or self.count is None:
            self.total = piece
            self.count = seen
            return
        self.total = self.total + piece
        self.count = self.count + seen

    def ensure(self, device: torch.device | str | None) -> None:
        if self.total is not None and self.count is not None:
            return
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.total = torch.zeros((), device=dev, dtype=torch.float64)
        self.count = torch.zeros((), device=dev, dtype=torch.float64)

    def mean(self) -> torch.Tensor | None:
        """The mean when :meth:`add` ran, else ``None``.

        A zero count yields zero via :func:`_mean_of`, with no host sync.
        ``float()`` at publish time is the sync.
        """
        if self.total is None or self.count is None:
            return None
        return _mean_of(self.total, self.count)


class PotentialReport:
    """Potential-error sums for one phase.

    Force MAE is the mean absolute component error. Each declared element
    gets the same mean on its atoms. Energy per atom is the mean over
    graphs of ``|E_pred - E_ref| / n_atoms``. Virial MAE is the mean
    absolute component error. ``force_outlier_count`` counts atoms whose
    force-error norm exceeds ``outlier_force``.

    Padding is ``atoms.mask`` / ``graphs.mask`` when the caller passes it.
    A loss weight (``atoms.force_weight``, an inlier mask) is not an
    argument: the report and the loss do not share that buffer.

    Args:
        elements: Atomic numbers to report separately.
        outlier_force: Force-error norm above which an atom counts as an
            outlier. ``None`` leaves the outlier sums untouched.
    """

    def __init__(
        self,
        elements: tuple[int, ...] = (),
        outlier_force: float | None = None,
    ) -> None:
        self.elements = elements
        self.outlier_force = outlier_force
        self.reset()

    def reset(self) -> None:
        """Drop every sum."""
        self.force = _SumCount()
        self.energy = _SumCount()
        self.virial = _SumCount()
        self.outliers = _SumCount()
        self.by_element = {element: _SumCount() for element in self.elements}

    def update_forces(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate component-absolute force error. ``pred`` and ``target`` are ``(N, 3)``."""
        delta = (pred.detach() - target.detach()).abs()
        live = None if mask is None else mask.detach().reshape(-1).to(torch.bool)
        if live is not None:
            weight = live.to(delta.dtype).reshape(live.shape[0], *([1] * (delta.ndim - 1)))
            delta = delta * weight
            count: torch.Tensor | int = live.to(delta.dtype).sum() * int(delta.shape[-1])
        else:
            count = int(delta.numel())
        self.force.add(delta.sum(), count)
        if z is None:
            return
        numbers = z.detach().reshape(-1)
        for element, stat in self.by_element.items():
            chosen = numbers == element
            if live is not None:
                chosen = chosen & live
            piece = (pred.detach() - target.detach()).abs()[chosen]
            stat.add(piece.sum(), int(piece.numel()))

    def update_energy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_atoms: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate ``|ΔE| / n_atoms`` over graphs."""
        per = (pred.detach() - target.detach()).abs().reshape(-1)
        scale = num_atoms.detach().reshape(-1).to(per.dtype).clamp(min=1)
        per = per / scale
        if mask is not None:
            live = mask.detach().reshape(-1).to(dtype=torch.bool)
            per = per * live.to(per.dtype)
            count: torch.Tensor | int = live.to(per.dtype).sum()
        else:
            count = int(per.numel())
        self.energy.add(per.sum(), count)

    def update_virial(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Accumulate the mean absolute virial-component error."""
        delta = (pred.detach() - target.detach()).abs()
        components = int(delta[0].numel()) if delta.ndim >= 1 and int(delta.shape[0]) else 1
        if mask is not None:
            live = mask.detach().reshape(-1).to(torch.bool)
            weight = live.to(delta.dtype)
            while weight.ndim < delta.ndim:
                weight = weight.unsqueeze(-1)
            delta = delta * weight
            count: torch.Tensor | int = live.to(delta.dtype).sum() * components
        else:
            count = int(delta.numel())
        self.virial.add(delta.sum(), count)

    def update_outliers(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Count atoms whose force-error norm exceeds ``outlier_force``."""
        if self.outlier_force is None:
            return
        err = (pred.detach() - target.detach()).norm(dim=-1).reshape(-1)
        if mask is not None:
            live = mask.detach().reshape(-1).to(torch.bool)
        else:
            live = torch.ones(err.shape[0], dtype=torch.bool, device=err.device)
        hit = (err > self.outlier_force) & live
        self.outliers.add(hit.to(torch.float64).sum(), live.to(torch.float64).sum())

    def sync(self, device: torch.device | str | None = None) -> None:
        """All-reduce every sum. A one-process call leaves the sums local."""
        if _distributed_world() < 2:
            return
        stats = [self.force, self.energy, self.virial, self.outliers, *self.by_element.values()]
        for stat in stats:
            stat.ensure(device)
        totals = [stat.total for stat in stats if stat.total is not None]
        counts = [stat.count for stat in stats if stat.count is not None]
        _all_reduce_sums(*totals, *counts)

    def scalars(self) -> dict[str, torch.Tensor]:
        """Names to 0-d tensors for the sums that saw at least one element."""
        out: dict[str, torch.Tensor] = {}
        force = self.force.mean()
        if force is not None:
            out["force_mae"] = force
        for element, stat in self.by_element.items():
            value = stat.mean()
            if value is not None:
                out[f"force_mae_Z{element}"] = value
        energy = self.energy.mean()
        if energy is not None:
            out["energy_per_atom_mae"] = energy
        virial = self.virial.mean()
        if virial is not None:
            out["virial_mae"] = virial
        outliers = self.outliers.mean()
        if outliers is not None and self.outliers.total is not None:
            out["force_outlier_fraction"] = outliers
            out["force_outlier_count"] = self.outliers.total
        return out
