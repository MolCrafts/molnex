"""Trainer implementation for Molix."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import torch.nn as nn

from molix.core.hooks import Hook
from molix.core.state import Stage, TrainState
from molix.steps.eval_step import EvalStep
from molix.steps.train_step import TrainStep

logger = logging.getLogger(__name__)


class DataModule:
    """Simple data module interface.
    
    Provides train and validation dataloaders.
    """
    
    def train_dataloader(self) -> Iterable[Any]:
        """Return training dataloader."""
        raise NotImplementedError
    
    def val_dataloader(self) -> Iterable[Any]:
        """Return validation dataloader."""
        raise NotImplementedError


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
        train_step: TrainStep | None = None,
        eval_step: EvalStep | None = None,
        hooks: list[Hook | tuple[Hook, int]] | None = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model (for direct training)
            loss_fn: Loss function (for direct training)
            optimizer_factory: Factory to create optimizer from parameters
            train_step: Training step (for step-based training, backward compatible)
            eval_step: Evaluation step (for step-based training, backward compatible)
            hooks: List of hooks or (hook, priority) tuples. Hooks execute in
                   registration order by default. Use tuples to override priority
                   (lower priority = earlier execution, default = 100).
            
        Note:
            Use either (model, loss_fn, optimizer_factory) for direct training
            OR (train_step, eval_step) for step-based training (legacy).
        """
        # Direct training mode
        if model is not None:
            self.model = model
            self.loss_fn = loss_fn
            self.optimizer = optimizer_factory(model.parameters()) if optimizer_factory else None
            self.train_step = None
            self.eval_step = None
            self._use_direct_training = True
        else:
            # Step-based training (backward compatible)
            self.model = None
            self.loss_fn = None
            self.optimizer = None
            self.train_step = train_step or TrainStep()
            self.eval_step = eval_step or EvalStep()
            self._use_direct_training = False
        
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
        datamodule: DataModule,
        max_epochs: int = 1,
    ) -> TrainState:
        """Execute training loop.
        
        Args:
            datamodule: Data module providing train/val dataloaders
            max_epochs: Maximum number of epochs to train
            
        Returns:
            Final training state
        """
        if self._use_direct_training:
            return self._train_direct(datamodule, max_epochs)
        else:
            return self._train_with_steps(datamodule, max_epochs)
    
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
                    f"Error in hook {hook.__class__.__name__}.{hook_name}: {e}",
                    exc_info=True
                )
    
    def _train_direct(
        self,
        datamodule: DataModule,
        max_epochs: int,
    ) -> TrainState:
        """Execute training loop with direct model/loss/optimizer."""
        # Hook: on_train_start
        if self.hooks:
            self._call_hooks("on_train_start", self, self.state)
        
        for epoch in range(max_epochs):
            # Hook: on_epoch_start
            if self.hooks:
                self._call_hooks("on_epoch_start", self, self.state)
            
            # Training phase
            self.state.set_stage(Stage.TRAIN)
            self.model.train()
            
            for batch in datamodule.train_dataloader():
                # Hook: on_train_batch_start
                if self.hooks:
                    self._call_hooks("on_train_batch_start", self, self.state, batch)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                targets = batch.get("y_energy") if isinstance(batch, dict) else batch
                loss = self.loss_fn(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Hook: on_train_batch_end
                if self.hooks:
                    outputs = {"loss": loss, "predictions": predictions}
                    self._call_hooks("on_train_batch_end", self, self.state, batch, outputs)
                
                self.state.increment_step()
            
            # Validation phase
            self.state.set_stage(Stage.EVAL)
            self.model.eval()
            
            for batch in datamodule.val_dataloader():
                # Hook: on_eval_batch_start
                if self.hooks:
                    self._call_hooks("on_eval_batch_start", self, self.state, batch)
                
                # Forward pass (no gradients)
                predictions = self.model(batch)
                
                # Compute loss (for logging)
                targets = batch.get("y_energy") if isinstance(batch, dict) else batch
                loss = self.loss_fn(predictions, targets)
                
                # Hook: on_eval_batch_end
                if self.hooks:
                    outputs = {"loss": loss, "predictions": predictions}
                    self._call_hooks("on_eval_batch_end", self, self.state, batch, outputs)
            
            # Hook: on_epoch_end
            if self.hooks:
                self._call_hooks("on_epoch_end", self, self.state)
            
            self.state.increment_epoch()
        
        # Hook: on_train_end
        if self.hooks:
            self._call_hooks("on_train_end", self, self.state)
        
        return self.state
    def _train_with_steps(
        self,
        datamodule: DataModule,
        max_epochs: int,
    ) -> TrainState:
        """Execute training loop with step objects (backward compatible)."""
        # Hook: on_train_start
        if self.hooks:
            self._call_hooks("on_train_start", self, self.state)
        
        for epoch in range(max_epochs):
            # Hook: on_epoch_start
            if self.hooks:
                self._call_hooks("on_epoch_start", self, self.state)
            
            # Training phase
            self.state.set_stage(Stage.TRAIN)
            for batch in datamodule.train_dataloader():
                # Hook: on_train_batch_start
                if self.hooks:
                    self._call_hooks("on_train_batch_start", self, self.state, batch)
                
                result = self.train_step.run(self.state, batch=batch)
                
                # Hook: on_train_batch_end
                if self.hooks:
                    self._call_hooks("on_train_batch_end", self, self.state, batch, result)
                
                self.state.increment_step()
            
            # Validation phase
            self.state.set_stage(Stage.EVAL)
            for batch in datamodule.val_dataloader():
                # Hook: on_eval_batch_start
                if self.hooks:
                    self._call_hooks("on_eval_batch_start", self, self.state, batch)
                
                result = self.eval_step.run(self.state, batch=batch)
                
                # Hook: on_eval_batch_end
                if self.hooks:
                    self._call_hooks("on_eval_batch_end", self, self.state, batch, result)
            
            # Hook: on_epoch_end
            if self.hooks:
                self._call_hooks("on_epoch_end", self, self.state)
            
            self.state.increment_epoch()
        
        # Hook: on_train_end
        if self.hooks:
            self._call_hooks("on_train_end", self, self.state)
        
        return self.state
    
    @property
    def tasks(self) -> Iterable[Any]:
        """Return sequence of task nodes (step objects)."""
        return [self.train_step, self.eval_step]
    
    @property
    def links(self) -> Iterable[dict]:
        """Return sequence of links representing flow between tasks."""
        return [
            {"source": "train_step", "target": "eval_step", "type": "stage_flow"}
        ]
    
    @property
    def metadata(self) -> dict:
        """Return metadata about the workflow (stage ordering, loop structure, etc.)."""
        return {
            "stage_order": [Stage.TRAIN.value, Stage.EVAL.value],
            "loop_structure": {
                "description": "epoch loop with train and eval stages",
                "stages": {
                    "train": {
                        "step_type": "TrainStep",
                        "iterates_over": "train_dataloader",
                    },
                    "eval": {
                        "step_type": "EvalStep",
                        "iterates_over": "val_dataloader",
                    },
                },
            },
        }
