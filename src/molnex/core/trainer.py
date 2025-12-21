"""Trainer implementation for MolNex."""

from typing import Any, Callable, Iterable, Optional

import torch.nn as nn

from molnex.core.state import Stage, TrainState
from molnex.graph.builder import Graph
from molnex.steps.eval_step import EvalStep
from molnex.steps.train_step import TrainStep


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
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Callable] = None,
        optimizer_factory: Optional[Callable] = None,
        train_step: Optional[TrainStep] = None,
        eval_step: Optional[EvalStep] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model (for direct training)
            loss_fn: Loss function (for direct training)
            optimizer_factory: Factory to create optimizer from parameters
            train_step: Training step (for step-based training, backward compatible)
            eval_step: Evaluation step (for step-based training, backward compatible)
            
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
    
    def _train_direct(
        self,
        datamodule: DataModule,
        max_epochs: int,
    ) -> TrainState:
        """Execute training loop with direct model/loss/optimizer."""
        for epoch in range(max_epochs):
            # Training phase
            self.state.set_stage(Stage.TRAIN)
            self.model.train()
            
            for batch in datamodule.train_dataloader():
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                targets = batch.get("y_energy") if isinstance(batch, dict) else batch
                loss = self.loss_fn(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.state.increment_step()
            
            # Validation phase
            self.state.set_stage(Stage.EVAL)
            self.model.eval()
            
            for batch in datamodule.val_dataloader():
                # Forward pass (no gradients)
                predictions = self.model(batch)
                
                # Compute loss (for logging)
                targets = batch.get("y_energy") if isinstance(batch, dict) else batch
                loss = self.loss_fn(predictions, targets)
            
            self.state.increment_epoch()
        
        return self.state
    
    def _train_with_steps(
        self,
        datamodule: DataModule,
        max_epochs: int,
    ) -> TrainState:
        """Execute training loop with step objects (backward compatible)."""
        for epoch in range(max_epochs):
            # Training phase
            self.state.set_stage(Stage.TRAIN)
            for batch in datamodule.train_dataloader():
                result = self.train_step.run(self.state, batch=batch)
                self.state.increment_step()
            
            # Validation phase
            self.state.set_stage(Stage.EVAL)
            for batch in datamodule.val_dataloader():
                result = self.eval_step.run(self.state, batch=batch)
            
            self.state.increment_epoch()
        
        return self.state
    
    def to_graph(self) -> Graph:
        """Export graph representation of training structure.
        
        This method:
        - Does NOT execute training
        - Returns a semantic graph (not execution trace)
        - Produces deterministic output
        
        Returns:
            Graph object with nodes, edges, and metadata
        """
        nodes = [self.train_step, self.eval_step]
        
        # Simple edge representing stage flow
        edges = [
            {"source": "train_step", "target": "eval_step", "type": "stage_flow"}
        ]
        
        meta = {
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
        
        return Graph(nodes=nodes, edges=edges, meta=meta)
