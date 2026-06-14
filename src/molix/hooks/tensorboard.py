"""TensorBoard event-file logging hook."""

from __future__ import annotations

from molix import logger as _logger_mod
from molix.core.hook import BaseHook
from molix.hooks._utils import _as_scalar

logger = _logger_mod.getLogger(__name__)


class TensorBoardHook(BaseHook):
    """Logs whatever scalar metrics other hooks have written into ``state``.

    The hook does not require a key list. On each train-batch end it scans
    ``state`` for keys under the ``train/*`` and ``performance/*`` namespaces;
    on each eval completion it scans ``eval/*``. Any value that is a number
    or a 0-d tensor is written to TensorBoard under its state key; non-scalar
    entries are skipped.

    Args:
        every_n_steps: Logging frequency for train-step scalars.
        log_dir: Directory to save TensorBoard event files.
        log_hparams: Log hyperparameters for the HParams dashboard.
        log_histograms: Log weight/gradient histograms each epoch.
        hparams: Hyperparameter dict (required when ``log_hparams=True``).
        histogram_freq: Log histograms every N epochs (default: 1).
        start_step: Suppress train/performance/gpu writes before this step.
            Eval scalars are still written (one eval phase point per
            evaluation is usually wanted from the beginning).
    """

    TRAIN_NAMESPACES: tuple[str, ...] = ("train", "performance", "gpu")
    EVAL_NAMESPACES: tuple[str, ...] = ("eval",)

    def __init__(
        self,
        every_n_steps: int,
        log_dir: str,
        *,
        log_hparams: bool = False,
        log_histograms: bool = False,
        hparams: dict | None = None,
        histogram_freq: int = 1,
        start_step: int = 0,
    ):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")

        from torch.utils.tensorboard import SummaryWriter

        self.SummaryWriter = SummaryWriter
        self.log_dir = log_dir
        self.every_n_steps = every_n_steps
        self.log_hparams = log_hparams
        self.log_histograms = log_histograms
        self.hparams = hparams or {}
        self.histogram_freq = histogram_freq
        self.start_step = start_step

        self.writer = None
        self._graph_logged = False

    def on_train_start(self, trainer, state):
        """Open the SummaryWriter."""
        self.writer = self.SummaryWriter(self.log_dir)
        if self.log_hparams and self.hparams:
            logger.info(f"TensorBoardHook: logging hyperparameters {self.hparams}")

    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Log every ``train/*``, ``performance/*``, ``gpu/*`` scalar."""
        if state.global_step < self.start_step:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        self._log_namespaces(state, self.TRAIN_NAMESPACES)

    def on_eval_step_complete(self, trainer, state):
        """Log every ``eval/*`` scalar."""
        self._log_namespaces(state, self.EVAL_NAMESPACES)

    def _log_namespaces(self, state, namespaces: tuple[str, ...]) -> None:
        for ns in namespaces:
            for k, value in state[ns].items():
                scalar = _as_scalar(value)
                if scalar is None:
                    continue
                self.writer.add_scalar(f"{ns}/{k}", scalar, state.global_step)

    def on_epoch_end(self, trainer, state):
        """Log weight/gradient histograms."""
        if self.log_histograms and (state.epoch + 1) % self.histogram_freq == 0:
            self._log_histograms(trainer, state)

    def on_train_end(self, trainer, state):
        """Log final metrics with hyperparameters and close writer."""
        if self.log_hparams and self.hparams:
            final_metrics = self._extract_final_metrics(trainer)
            if final_metrics:
                self.writer.add_hparams(self.hparams, final_metrics)
                logger.info(f"Logged hyperparameters with final metrics: {final_metrics}")

        if self.writer:
            self.writer.close()

    def _log_histograms(self, trainer, state):
        """Log weight and gradient histograms."""
        for name, param in trainer.model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param.data, state.epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad.data, state.epoch)

    def _extract_final_metrics(self, trainer):
        """Extract final metrics for hparams logging."""
        final_metrics = {}

        for hook in trainer.hooks:
            hook_obj = hook[0] if isinstance(hook, tuple) else hook
            if hook_obj.__class__.__name__ == "MetricsHook":
                for metric in hook_obj.metrics:
                    value = metric.compute()
                    metric_name = metric.__class__.__name__
                    final_metrics[f"final_{metric_name}"] = value

        return final_metrics
