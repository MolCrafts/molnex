"""Trainer implementation for Molix."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from molix import logger as _logger_mod
from molix.core.hooks import Hook
from molix.core.state import Stage, TrainState
from molix.core.steps import DefaultEvalStep, DefaultTrainStep, Step
from molix.data.datamodule import DataModuleProtocol

logger = _logger_mod.getLogger(__name__)


class Trainer:
    """ML training system for MolNex.

    The Trainer:
    - Owns execution and control flow
    - Owns training state
    - Executes epoch/step loops
    - Provides graph export for introspection

    Attributes:
        train_step: Training step object (for step-based training)
        eval_step: Evaluation step object (for step-based training)
        model: Neural network model (for direct training)
        loss_fn: Loss function (for direct training)
        optimizer: Optimizer instance (for direct training)
        state: Current training state
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        loss_fn: Callable | None = None,
        optimizer_factory: Callable | None = None,
        train_step: Step | None = None,
        eval_step: Step | None = None,
        hooks: list[Hook | tuple[Hook, int]] | None = None,
        eval_every_n_steps: int | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Neural network model (for direct training)
            loss_fn: Loss function (for direct training)
            optimizer_factory: Factory to create optimizer from parameters
            train_step: Training step implementing Step protocol. If None, uses
                DefaultTrainStep.
            eval_step: Evaluation step implementing Step protocol. If None, uses
                DefaultEvalStep.
            hooks: List of hooks or (hook, priority) tuples. Hooks execute in
                   registration order by default. Use tuples to override priority
                   (lower priority = earlier execution, default = 100).
            eval_every_n_steps: Run evaluation every N training steps (in addition
                   to epoch-end eval). If None (default), only epoch-end eval runs.
                   Must be > 0 if provided.

        Note:
            For training with default steps:
                trainer = Trainer(model=model, loss_fn=loss_fn, optimizer_factory=opt_factory)

            For custom steps:
                trainer = Trainer(model=model, loss_fn=loss_fn, optimizer_factory=opt_factory,
                                train_step=MyCustomStep(), eval_step=MyCustomStep())

        Raises:
            ValueError: If eval_every_n_steps is <= 0
        """
        # Validate eval_every_n_steps
        if eval_every_n_steps is not None and eval_every_n_steps <= 0:
            raise ValueError(f"eval_every_n_steps must be > 0, got {eval_every_n_steps}")
        self.eval_every_n_steps = eval_every_n_steps

        # Initialize model, loss, optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = (
            optimizer_factory(model.parameters()) if optimizer_factory and model else None
        )

        # Use default steps if not provided, or accept custom steps
        self.train_step = train_step if train_step is not None else DefaultTrainStep()
        self.eval_step = eval_step if eval_step is not None else DefaultEvalStep()

        self.state = TrainState()

        # Initialize hooks with priority sorting
        if not hooks:
            self.hooks = []
        else:
            # Normalize hooks to (hook, priority, index) tuples
            normalized = []
            for idx, item in enumerate(hooks):
                if isinstance(item, tuple):
                    hook, priority = item
                    normalized.append((hook, priority, idx))
                else:
                    normalized.append((item, 100, idx))  # Default priority 100

            # Sort by priority (ascending), then by registration index
            normalized.sort(key=lambda x: (x[1], x[2]))

            # Extract just the hooks
            self.hooks = [hook for hook, _, _ in normalized]

    def train(
        self,
        datamodule: DataModuleProtocol,
        max_epochs: int = 1,
    ) -> TrainState:
        """Execute training loop.

        Args:
            datamodule: Data module providing train/val dataloaders
            max_epochs: Maximum number of epochs to train

        Returns:
            Final training state
        """
        return self._train(datamodule, max_epochs)

    def _call_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Call a hook on all registered hooks.

        Args:
            hook_name: Name of the hook method to call
            *args: Positional arguments to pass to hook
            **kwargs: Keyword arguments to pass to hook
        """
        for hook in self.hooks:
            try:
                method = getattr(hook, hook_name, None)
                if method is not None and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in hook {hook.__class__.__name__}.{hook_name}: {e}", exc_info=True
                )

    def _train(
        self,
        datamodule: DataModuleProtocol,
        max_epochs: int,
    ) -> TrainState:
        """Execute training loop."""
        assert self.model is not None, "Trainer.model must be set before calling train()"

        # DataModule setup
        if hasattr(datamodule, "setup"):
            datamodule.setup("fit")

        # Hook: on_train_start
        if self.hooks is not None:
            self._call_hooks("on_train_start", self, self.state)

        for epoch in range(max_epochs):
            # DataModule epoch sync (DDP sampler shuffle)
            if hasattr(datamodule, "on_epoch_start"):
                datamodule.on_epoch_start(epoch)

            # Hook: on_epoch_start
            if self.hooks is not None:
                self._call_hooks("on_epoch_start", self, self.state)

            # Training phase
            self.state.set_stage(Stage.TRAIN)
            self.model.train()

            for batch in datamodule.train_dataloader():
                # Hook: on_train_batch_start
                if self.hooks is not None:
                    self._call_hooks("on_train_batch_start", self, self.state, batch)

                # Delegate computation to train_step
                outputs = self.train_step.on_train_batch(self, self.state, batch)

                # Hook: on_train_batch_end
                if self.hooks is not None:
                    self._call_hooks("on_train_batch_end", self, self.state, batch, outputs)

                self.state.increment_step()
                self.state.steps_since_last_eval += 1

                # Check if step-based eval should run
                if (
                    self.eval_every_n_steps is not None
                    and self.state.steps_since_last_eval >= self.eval_every_n_steps
                ):
                    self._run_eval_phase(datamodule)
                    self.state.steps_since_last_eval = 0

                    # Call step-based eval completion hook
                    if self.hooks is not None:
                        self._call_hooks("on_eval_step_complete", self, self.state)

            # Validation phase
            self.state.set_stage(Stage.EVAL)
            self.model.eval()

            for batch in datamodule.val_dataloader():
                # Hook: on_eval_batch_start
                if self.hooks is not None:
                    self._call_hooks("on_eval_batch_start", self, self.state, batch)

                # Delegate computation to eval_step
                outputs = self.eval_step.on_eval_batch(self, self.state, batch)

                # Hook: on_eval_batch_end
                if self.hooks is not None:
                    self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)

            # Hook: on_epoch_end
            if self.hooks is not None:
                self._call_hooks("on_epoch_end", self, self.state)

            self.state.increment_epoch()

        # Hook: on_train_end
        if self.hooks is not None:
            self._call_hooks("on_train_end", self, self.state)

        return self.state

    def _run_eval_phase(self, datamodule: DataModuleProtocol) -> None:
        """Run evaluation phase (internal helper).

        Args:
            datamodule: Data module providing validation dataloader
        """
        # Save current stage and switch to eval
        prev_stage = self.state.stage
        self.state.set_stage(Stage.EVAL)
        self.model.eval()

        for batch in datamodule.val_dataloader():
            # Hook: on_eval_batch_start
            if self.hooks is not None:
                self._call_hooks("on_eval_batch_start", self, self.state, batch)

            # Delegate computation to eval_step
            outputs = self.eval_step.on_eval_batch(self, self.state, batch)

            # Hook: on_eval_batch_end
            if self.hooks is not None:
                self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)

        # Restore training mode
        self.model.train()
        self.state.set_stage(prev_stage)
