# Molix Quickstart

**Goal**: Train a simple linear regression model using the Molix Trainer in 2 minutes.

This guide focuses purely on the **training mechanics**. We will use a standard PyTorch model and synthetic data to show how `Trainer` handles the loop.

## 1. The Setup

You don't need complex molecules to learn Molix. Let's train a model to learn the function $y = 2x + 1$.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Standard PyTorch Model
model = nn.Linear(1, 1)

# 2. Synthetic Data (y = 2x + 1)
X = torch.randn(100, 1)
y = 2 * X + 1
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)
```

## 2. The Trainer

Instead of writing a `for epoch in range(10):` loop, we initialize the `Trainer`.

```python
from molix.core.trainer import Trainer

class SimpleDataModule:
    def __init__(self, loader):
        self._loader = loader

    def train_dataloader(self):
        return self._loader

    def val_dataloader(self):
        return self._loader

trainer = Trainer(
    model=model,
    loss_fn=nn.MSELoss(),
    optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.1),
)

trainer.train(SimpleDataModule(dataloader), max_epochs=5)
```

## 3. That's it?

Yes. Under the hood, this simple call handled:

* Device placement (CPU/GPU)
* Gradient zeroing and stepping
* Validation loop (no-grad context)
* Loss logging
* Progress bar (if hook enabled)

## Next Steps

Now that you know the basics, explore the powerful features that make Molix unique:

* [**Hooks**](core/hooks.md): Add checkpointing, TensorBoard logging, and early stopping.
* [**Data Modules**](data/datamodules.md): Organize your data loading logic.
* [**The Trainer API**](core/trainer.md): Deep dive into `TrainState` and custom steps.
