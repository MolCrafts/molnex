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
        ``train_every_n_steps`` so the ``compute()`` sync doesn't fire on
        every micro-step of a launch-bound loop.
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
        """
        val_ns = state[self.prefix_val]
        for metric in self.val_metrics:
            val_ns[self._scalar_name(metric)] = metric.compute()


class StepSpeedHook(ScalarHook):
    """Track training step speed and write to ``state["performance"]``.

    Measures steps per second during training and writes to
    ``state["performance"]["step_per_second"]``.

    Args:
        window_size: Number of steps to average over (default: 10).
    """

    scalar_keys = (("performance", "step_per_second"),)

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._step_start_time = None
        self._steps_in_window = 0

    def on_train_start(self, trainer, state):
        """Initialize timing."""
        import time

        self._step_start_time = time.time()
        self._steps_in_window = 0

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Compute step speed and write to state."""
        import time

        self._steps_in_window += 1

        if self._steps_in_window >= self.window_size:
            if self._step_start_time is not None:
                elapsed = time.time() - self._step_start_time
                steps_per_sec = self._steps_in_window / elapsed

                state["performance"]["step_per_second"] = steps_per_sec

                self._step_start_time = time.time()
                self._steps_in_window = 0
