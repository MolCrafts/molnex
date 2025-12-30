"""Hook system for Molix Trainer.

This module provides an extensible hook system that allows users to inject
custom logic at various points in the training lifecycle.

Hooks execute in registration order by default. Use (hook, priority) tuples
to override execution order (lower priority = earlier execution).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from molix.core.trainer import Trainer
    from molix.core.state import TrainState

logger = logging.getLogger(__name__)


class Hook(Protocol):
    """Protocol for training hooks.
    
    Hooks receive notifications at various points in the training lifecycle.
    All methods are optional - implement only the hooks you need.
    
    Hook execution order:
    - By default, hooks execute in registration order
    - Use (hook, priority) tuples to override execution order
    - Lower priority values execute earlier (default priority = 100)
    - Hooks with same priority execute in registration order
    - If a hook raises an exception, it is logged but training continues
    
    Example:
        ```python
        class MyHook:
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")
        
        # Registration order
        trainer = Trainer(hooks=[MyHook(), OtherHook()])
        
        # With priority
        trainer = Trainer(hooks=[(MyHook(), 10), OtherHook()])
        ```
    """
    
    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training.
        
        Args:
            trainer: The trainer instance
            state: Current training state (epoch=0, global_step=0)
        """
        ...
    
    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training.
        
        Args:
            trainer: The trainer instance
            state: Final training state
        """
        ...
    
    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch.
        
        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...
    
    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation).
        
        Args:
            trainer: The trainer instance
            state: Current training state
        """
        ...
    
    def on_train_batch_start(
        self, 
        trainer: "Trainer", 
        state: "TrainState", 
        batch: Any
    ) -> None:
        """Called before processing each training batch.
        
        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...
    
    def on_train_batch_end(
        self, 
        trainer: "Trainer", 
        state: "TrainState", 
        batch: Any, 
        outputs: Any
    ) -> None:
        """Called after processing each training batch.
        
        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the training step (loss, predictions, etc.)
        """
        ...
    
    def on_eval_batch_start(
        self, 
        trainer: "Trainer", 
        state: "TrainState", 
        batch: Any
    ) -> None:
        """Called before processing each validation batch.
        
        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
        """
        ...
    
    def on_eval_batch_end(
        self, 
        trainer: "Trainer", 
        state: "TrainState", 
        batch: Any, 
        outputs: Any
    ) -> None:
        """Called after processing each validation batch.
        
        Args:
            trainer: The trainer instance
            state: Current training state
            batch: The current batch data
            outputs: Outputs from the evaluation step (loss, metrics, etc.)
        """
        ...


class BaseHook:
    """Base hook with no-op implementations.
    
    Inherit from this class and override only the methods you need.
    This provides better IDE support than implementing the Protocol directly.
    
    Example:
        ```python
        class MyHook(BaseHook):
            def on_epoch_end(self, trainer, state):
                print(f"Epoch {state.epoch} completed")
        ```
    """
    
    def on_train_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called once at the end of training."""
        pass
    
    def on_epoch_start(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: "Trainer", state: "TrainState") -> None:
        """Called at the end of each epoch (after validation)."""
        pass
    
    def on_train_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each training batch."""
        pass
    
    def on_train_batch_end(self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any) -> None:
        """Called after processing each training batch."""
        pass
    
    def on_eval_batch_start(self, trainer: "Trainer", state: "TrainState", batch: Any) -> None:
        """Called before processing each validation batch."""
        pass
    
    def on_eval_batch_end(self, trainer: "Trainer", state: "TrainState", batch: Any, outputs: Any) -> None:
        """Called after processing each validation batch."""
        pass


# Built-in Hooks

class TensorBoardHook(BaseHook):
    """Logs training metrics to TensorBoard.
    
    Args:
        log_dir: Directory to save TensorBoard logs (default: "./runs")
        log_every_n_steps: Log training loss every N steps (default: 1)
    
    Example:
        ```python
        from molix.core.hooks import TensorBoardHook
        
        hook = TensorBoardHook(log_dir="./my_runs", log_every_n_steps=10)
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """
    
    def __init__(self, log_dir: str = "./runs", log_every_n_steps: int = 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard is required for TensorBoardHook. "
                "Install it with: pip install tensorboard"
            )
        
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.writer = None
    
    def on_train_start(self, trainer, state):
        """Initialize TensorBoard writer."""
        self.writer = self.SummaryWriter(self.log_dir)
    
    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Log training loss."""
        if state.global_step % self.log_every_n_steps == 0:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_value = outputs["loss"].item() if hasattr(outputs["loss"], "item") else outputs["loss"]
                self.writer.add_scalar("Loss/train", loss_value, state.global_step)
    
    def on_eval_batch_end(self, trainer, state, batch, outputs):
        """Log validation loss."""
        if isinstance(outputs, dict) and "loss" in outputs:
            loss_value = outputs["loss"].item() if hasattr(outputs["loss"], "item") else outputs["loss"]
            self.writer.add_scalar("Loss/val", loss_value, state.global_step)
    
    def on_epoch_end(self, trainer, state):
        """Log epoch number."""
        self.writer.add_scalar("Epoch", state.epoch, state.global_step)
    
    def on_train_end(self, trainer, state):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class CheckpointHook(BaseHook):
    """Saves model checkpoints during training.
    
    Args:
        checkpoint_dir: Directory to save checkpoints (default: "./checkpoints")
        save_every_n_epochs: Save checkpoint every N epochs (default: 1)
        save_last: Always save the last checkpoint (default: True)
    
    Example:
        ```python
        from molix.core.hooks import CheckpointHook
        
        hook = CheckpointHook(checkpoint_dir="./ckpt", save_every_n_epochs=5)
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """
    
    def __init__(
        self, 
        checkpoint_dir: str = "./checkpoints",
        save_every_n_epochs: int = 1,
        save_last: bool = True
    ):
        import os
        import torch
        self.os = os
        self.torch = torch
        
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
    
    def on_train_start(self, trainer, state):
        """Create checkpoint directory."""
        self.os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, trainer, state):
        """Save checkpoint at specified intervals."""
        if (state.epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, state, f"epoch_{state.epoch}.pt")
    
    def on_train_end(self, trainer, state):
        """Save final checkpoint."""
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")
    
    def _save_checkpoint(self, trainer, state, filename):
        """Save checkpoint to file."""
        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": trainer.model.state_dict() if trainer.model else None,
            "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None,
        }
        self.torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")


class ProgressBarHook(BaseHook):
    """Displays training progress with tqdm.
    
    Args:
        desc: Description for the progress bar (default: "Training")
        leave: Leave progress bar after completion (default: True)
    
    Example:
        ```python
        from molix.core.hooks import ProgressBarHook
        
        hook = ProgressBarHook(desc="My Training")
        trainer = Trainer(model=model, hooks=[hook])
        ```
    """
    
    def __init__(self, desc: str = "Training", leave: bool = True):
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
        except ImportError:
            raise ImportError(
                "tqdm is required for ProgressBarHook. "
                "Install it with: pip install tqdm"
            )
        
        self.desc = desc
        self.leave = leave
        self.pbar = None
        self.total_steps = None
    
    def on_train_start(self, trainer, state):
        """Initialize progress bar."""
        # We don't know total steps yet, will update on first epoch
        self.pbar = None
    
    def on_epoch_start(self, trainer, state):
        """Update progress bar for new epoch."""
        if self.pbar is None:
            # Create progress bar on first epoch
            self.pbar = self.tqdm(desc=f"{self.desc} Epoch {state.epoch}", leave=self.leave)
        else:
            self.pbar.set_description(f"{self.desc} Epoch {state.epoch}")
            self.pbar.reset()
    
    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Update progress bar after each batch."""
        if self.pbar:
            postfix = {}
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_value = outputs["loss"].item() if hasattr(outputs["loss"], "item") else outputs["loss"]
                postfix["loss"] = f"{loss_value:.4f}"
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
    
    def on_train_end(self, trainer, state):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
