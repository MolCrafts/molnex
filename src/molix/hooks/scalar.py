"""Scalar-publishing hooks: ``MetricsHook`` and ``StepSpeedHook``.

Per the io-hooks-refactor scalar-vs-training split, this module
houses hooks whose responsibility is to populate scalar values into
:class:`~molix.core.state.TrainState` namespaces. Training-loop
intervention hooks (gradient clipping, activation checkpointing) live
in :mod:`molix.hooks.training`.
"""

from __future__ import annotations

from typing import Any

from molix.core.hook import ScalarHook
from molix.core.metrics import BaseMetric
from molix.core.state import Path


class MetricsHook(ScalarHook):
    """Track train and val metrics with phase isolation.

    Invariants:

    - ``train_metrics`` and ``val_metrics`` are independent deep copies of
      the supplied metrics, so neither side can corrupt the other.
    - Train metrics are per-batch (reset + update + compute each batch),
      matching the per-batch semantics of ``train/loss``.
    - Val metrics accumulate across a whole eval phase and are published
      once in ``on_eval_step_complete``.

    Args:
        metrics: Metric instances satisfying the
            :class:`~molix.core.metrics.Metric` Protocol
            (``update`` / ``compute`` / ``reset``). Deep-copied so
            train and val accumulators never share buffers.
        pred_key: Dotted path or tuple to extract predictions from outputs.
        target_key: Dotted path or tuple to extract targets from batch.
        prefix_train: State namespace for train metrics (default ``"train"``).
        prefix_val: State namespace for val metrics (default ``"eval"``).
        name_prefix: Optional string prepended to each metric's scalar name.
            Use when registering two MetricsHook instances for different
            quantities (e.g. ``name_prefix="E_"`` for energy,
            ``name_prefix="F_"`` for forces) so their scalar keys don't
            collide.
    """

    def __init__(
        self,
        metrics: list[Any],
        pred_key: str | tuple = "predictions",
        target_key: str | tuple = "targets",
        prefix_train: str = "train",
        prefix_val: str = "eval",
        name_prefix: str = "",
        train_every_n_steps: int = 1,
    ):
        import copy

        self.train_metrics = [copy.deepcopy(m) for m in metrics]
        self.val_metrics = [copy.deepcopy(m) for m in metrics]
        self.pred_key = pred_key if isinstance(pred_key, tuple) else (pred_key,)
        self.target_key = target_key if isinstance(target_key, tuple) else (target_key,)
        self.prefix_train = prefix_train
        self.prefix_val = prefix_val
        self.name_prefix = name_prefix
        # Each train-metric ``compute()`` forces a ``.item()`` CPU↔GPU sync.
        # Computing it every step throttles a launch-bound loop; sample it
        # every ``train_every_n_steps`` batches instead (the value is only
        # read by ``Log`` on its own, usually coarser, cadence).
        self.train_every_n_steps = max(1, train_every_n_steps)
        self._train_batch_count = 0

    def _scalar_name(self, metric: Any) -> str:
        return self.name_prefix + type(metric).__name__

    @property
    def scalar_keys(self) -> tuple[Path, ...]:
        """The ``(prefix, metric_name)`` keys this hook writes for both phases.

        Returns one key per metric under each of ``prefix_train`` and
        ``prefix_val`` (the train / eval namespace sub-dicts).
        """
        names = [self._scalar_name(m) for m in self.train_metrics]
        return tuple((prefix, n) for prefix in (self.prefix_train, self.prefix_val) for n in names)

    def _extract_value(self, data: Any, keys: tuple) -> Any:
        """Extract value from nested dict/dataclass using key path."""
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            elif hasattr(value, key):
                value = getattr(value, key)
            elif hasattr(value, "__getitem__"):
                value = value[key]
            else:
                raise KeyError(f"Cannot extract key {key} from {type(value)}")
        return value

    def on_eval_phase_start(self, trainer, state):
        """Reset the eval-side accumulators at the start of *every* eval phase.

        Fired before the eval batch loop for both step-based and epoch-end
        eval, so each phase accumulates from a clean slate — independent of
        ``on_epoch_start`` (which does not run between mid-epoch step-based
        evals). This is the structural guard against the cross-phase
        accumulator bleed that produced the train-MAE-exploded-to-203 bug.
        """
        for metric in self.val_metrics:
            metric.reset()

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Compute each train metric on this batch and write it to ``state[prefix_train]``.

        Train metrics are reset every computed step, so each value is the
        per-batch metric rather than a running average. Throttled by
        ``train_every_n_steps`` so it doesn't fire on every micro-step of a
        launch-bound loop.

        The metric is stored as the **0-d device tensor** returned by
        ``compute()`` — *not* ``.item()``-ed — so this hot-path hook never
        forces a CPU↔GPU sync. Downstream consumers materialise it on their
        own throttled cadence: the async :class:`~molix.hooks.JournalHook`
        copies it through a pinned buffer + CUDA event, and TensorBoard /
        progress hooks coerce via ``_as_scalar`` when they sample.
        """
        self._train_batch_count += 1
        if self._train_batch_count % self.train_every_n_steps != 0:
            return

        preds = self._extract_value(outputs, self.pred_key)
        targets = self._extract_value(batch, self.target_key)

        train_ns = state[self.prefix_train]
        for metric in self.train_metrics:
            metric.reset()
            metric.update(preds, targets)
            train_ns[self._scalar_name(metric)] = metric.compute()

    def on_eval_batch_end(self, trainer, state, batch, outputs):
        """Accumulate this batch into the eval metrics (running over the eval set)."""
        preds = self._extract_value(outputs, self.pred_key)
        targets = self._extract_value(batch, self.target_key)
        for metric in self.val_metrics:
            metric.update(preds, targets)

    def on_eval_step_complete(self, trainer, state):
        """Finalize the accumulated eval metrics into ``state[prefix_val]``.

        Fired at the end of every eval phase so the values are published
        before the LR scheduler reads ``best_metric_name``. The reset lives
        in :meth:`on_eval_phase_start` (start-of-phase), so the published
        value here reflects exactly this phase's accumulation.

        Unlike the train path, eval metrics are materialised to Python
        ``float`` here: this is a cold path (once per eval phase, not per
        batch) and the LR scheduler / ``CheckpointHook`` compare and
        serialise ``best_metric`` as a scalar, so they need a real number
        rather than a device tensor.
        """
        device = None
        try:
            device = next(trainer.model.parameters()).device
        except (AttributeError, StopIteration, RuntimeError):
            device = None
        val_ns = state[self.prefix_val]
        for metric in self.val_metrics:
            # A torchmetrics metric syncs inside its own compute(); its sync()
            # takes a dist_sync_fn, not a device.
            if isinstance(metric, BaseMetric):
                metric.sync(device)
            val_ns[self._scalar_name(metric)] = float(metric.compute())


class StepSpeedHook(ScalarHook):
    """Publish delivered step rate, and the accelerator rate beside it.

    ``step_per_second`` is optimizer steps divided by wall time since
    :meth:`on_train_start`. Wall time is the trainer's data-wait plus
    compute, so a prefetch burst cannot report the GPU's burst rate as
    the run's throughput. ``compute_step_per_second`` divides the same
    steps by compute time only — what the accelerator can do when the
    next batch is already in hand. ``data_wait_s`` is the cumulative
    time blocked in the dataloader.

    The trainer records the two times around each micro-batch
    (``_data_wait_s``, ``_compute_s``). This hook only publishes them,
    on optimizer-step boundaries.
    """

    scalar_keys = (
        ("performance", "step_per_second"),
        ("performance", "compute_step_per_second"),
        ("performance", "data_wait_s"),
    )

    def __init__(self) -> None:
        self._optimizer_steps = 0

    def on_train_start(self, trainer, state):
        """Start the delivered-rate clock at zero steps."""
        del trainer, state
        self._optimizer_steps = 0

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Publish the cumulative rates once an optimizer step closes."""
        del batch, outputs
        accum = max(1, int(trainer.accumulate_grad_batches))
        if (int(trainer._micro_step) + 1) % accum != 0:
            return
        self._optimizer_steps += 1
        data_s = float(trainer._data_wait_s)
        compute_s = float(trainer._compute_s)
        wall = data_s + compute_s
        perf = state["performance"]
        if wall > 0.0:
            perf["step_per_second"] = self._optimizer_steps / wall
        if compute_s > 0.0:
            perf["compute_step_per_second"] = self._optimizer_steps / compute_s
        perf["data_wait_s"] = data_s
