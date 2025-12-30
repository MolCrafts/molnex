# Molix

**Unified modeling of molecular potentials and properties with physics-aware ML**

Molix is a standalone machine learning training system capable of unified modeling for molecular potentials. It is designed to be structurally compatible with MolExp protocols while maintaining complete independence.

## Core Features

**Standalone Training System**
Molix operates as a completely independent training system. It manages its own training loops, state, and execution flow without requiring any external dependencies for its core logic. This design ensures that Molix is lightweight and easy to integrate into various environments.

**ML-First API**
We built the API with machine learning practitioners in mind. Instead of generic workflow terms, Molix uses standard ML terminology like `Trainer`, `TrainState`, and `StepResult`. This reduces the cognitive load for users already familiar with the PyTorch ecosystem.

**Structural Protocol Compatibility**
Molix achieves compatibility with MolExp through structure, not inheritance. By using duck typing and matching protocol shapes, Molix objects can be consumed by systems expecting MolExp interfaces without adding a hard dependency on the MolExp library itself.

## Installation

### What it is
The minimal setup required to run Molix on your local machine.

### Why strictly separated
We keep the installation simple to ensure you only get the dependencies you need for training, without the weight of larger frameworks.

### How to use it
To install Molix in editable mode for development:

```bash
cd molix
pip install -e .
```

## Quick Start

### What it is
A minimal example demonstrating how to set up a training loop with a custom model and data.

### Why this design
This example highlights the separation between data, model execution, and the training loop, which gives you granular control over every step of the process.

### How to use it
Here is a complete, runnable script to train a simple model:

```python
from molix import Trainer, TrainStep, EvalStep

# 1. Define your data handling
class MyDataModule:
    """Manages training and validation data streams."""
    def train_dataloader(self):
        # Yield your training batches here
        for batch in training_data:
            yield batch
    
    def val_dataloader(self):
        # Yield your validation batches here
        for batch in validation_data:
            yield batch

# 2. Define the execution steps
def forward_pass(batch):
    """The model's forward pass logic."""
    return model(batch)

def optimizer_step():
    """The optimization logic."""
    optimizer.step()

# Wrap functions into executable steps
train_step = TrainStep(forward_fn=forward_pass, optimizer_fn=optimizer_step)
eval_step = EvalStep(forward_fn=forward_pass)

# 3. Initialize the Trainer
trainer = Trainer(train_step=train_step, eval_step=eval_step)

# 4. Start Training
datamodule = MyDataModule()
final_state = trainer.train(datamodule, max_epochs=10)

print(f"Training complete: {final_state.epoch} epochs, {final_state.global_step} steps")
```

## License

See LICENSE file.

