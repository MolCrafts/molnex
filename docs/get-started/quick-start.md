# Quick Start Guide

This guide walks you through training your first physics-aware model using MolNex. We will define a simple neural network, prepare synthetic molecular data, and run a training loop using the `Trainer`.

## 1. Defining the Model

We start by defining a simple PyTorch model. MolNex is compatible with any standard `torch.nn.Module`, provided it accepts a `TensorDict` as input. You don't need to inherit from special base classes—if it looks like a PyTorch model, it probably works.

Here is a minimal model that embeds atomic numbers and sums them up to predict energy (effectively learning per-atom energies):

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding for elements (e.g., H=1, C=6)
        self.embedding = nn.Embedding(10, 16)
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, td):
        # 1. Embed atoms
        z = td["atoms", "z"]
        h = self.embedding(z)
        
        # 2. Pool (sum over atoms per molecule)
        batch_idx = td["graph", "batch"]
        num_mols = batch_idx.max().item() + 1
        h_mol = torch.zeros(num_mols, 16, device=h.device)
        h_mol.index_add_(0, batch_idx, h)
        
        # 3. Predict properties
        return self.layers(h_mol)
```

## 2. Preparing the Data

Data in MolNex is handled using `tensordict`. This allows us to group related tensors (like atomic numbers, positions, and energies) into a single, cohesive unit, simplifying data passing throughout your pipeline.

Let's create a synthetic batch representing a single molecule (e.g., Methane):

```python
from molix.data.atomic_td import AtomTD

# Create a sample batch of data
batch = AtomTD.create(
    z=torch.tensor([6, 1, 1, 1, 1]),      # Atomic numbers
    x=torch.randn(5, 3),                   # Random positions
    batch=torch.tensor([0, 0, 0, 0, 0]),   # Batch indices
    energy=torch.tensor([[-40.5]]),        # Target energy
)
```

## 3. Setting up the Trainer

The `Trainer` orchestrates the optimization process, connecting your model, loss function, and optimizer. It handles the boilerplate of the training loop so you don't have to.

We define a trainer and a simple loss function that knows how to extract targets from our batch:

```python
from molix.core.trainer import Trainer

# Define a loss function that reads the target from the batch
def energy_loss(pred, batch):
    target = batch["target", "energy"]
    return torch.nn.functional.mse_loss(pred, target)

# Initialize the trainer
trainer = Trainer(
    model=SimpleModel(),
    loss_fn=energy_loss,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
)
```

## 4. Running the Training Loop

Finally, we wrap our batch in a simple `DataModule` to feed the trainer and start the run.

```python
class SimpleDataModule:
    def __init__(self, batch):
        self.batch = batch
    
    def train_dataloader(self):
        # Yield the same batch 100 times for demo purposes
        return (self.batch for _ in range(100))
    
    def val_dataloader(self):
        return (self.batch for _ in range(10))

# Train for 5 epochs
trainer.train(SimpleDataModule(batch), max_epochs=5)
```

## Complete Script

You can run the full example below to see it in action.

```python
import torch
import torch.nn as nn
from molix.core.trainer import Trainer
from molix.data.atomic_td import AtomTD

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 16)
        self.layers = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, td):
        z = td["atoms", "z"]
        h = self.embedding(z)
        batch_idx = td["graph", "batch"]
        num_mols = batch_idx.max().item() + 1
        h_mol = torch.zeros(num_mols, 16, device=h.device)
        h_mol.index_add_(0, batch_idx, h)
        return self.layers(h_mol)

class SimpleDataModule:
    def __init__(self, batch):
        self.batch = batch
    def train_dataloader(self): 
        return (self.batch for _ in range(100))
    def val_dataloader(self): 
        return (self.batch for _ in range(10))

def main():
    # 1. Data
    batch = AtomTD.create(
        z=torch.tensor([6, 1, 1, 1, 1]),
        x=torch.randn(5, 3),
        batch=torch.tensor([0, 0, 0, 0, 0]),
        energy=torch.tensor([[-40.5]]),
    )

    # 2. Trainer
    trainer = Trainer(
        model=SimpleModel(),
        loss_fn=lambda p, b: torch.nn.functional.mse_loss(p, b["target", "energy"]),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
    )

    # 3. Train
    print("Starting training...")
    trainer.train(SimpleDataModule(batch), max_epochs=5)
    print("Training finished!")

if __name__ == "__main__":
    main()
```
