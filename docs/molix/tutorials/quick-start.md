# Molix Quick Start

This tutorial trains a small PyTorch model with `molix.core.trainer.Trainer`.
It uses ordinary tensors so you can learn the training mechanics before adding
molecular graph batches.

## 1. Create Data

Molix's default step forwards every batch key except `targets` and `extras` to
the model. The loss function receives both the prediction and the full batch.

```python
import torch
from torch.utils.data import DataLoader

X = torch.randn(100, 1)
y = 2 * X.squeeze(-1) + 1

dataset = [
    {"x": X[i], "targets": {"y": y[i]}}
    for i in range(len(X))
]
loader = DataLoader(dataset, batch_size=10, shuffle=True)
```

## 2. Define the Model and Loss

```python
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loss_fn(predictions, batch):
    return F.mse_loss(predictions, batch["targets"]["y"])
```

## 3. Add a Data Module

`Trainer` expects an object with `train_dataloader()` and, for validation,
`val_dataloader()`.

```python
class SimpleDataModule:
    def __init__(self, loader):
        self.loader = loader

    def train_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return self.loader
```

## 4. Train

```python
import torch
from molix.core.trainer import Trainer

trainer = Trainer(
    model=LinearModel(),
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
)

state = trainer.train(SimpleDataModule(loader), max_epochs=5)
print(state["train/loss"])
```

## Next Steps

- [Train a Graph Model](train-a-graph-model.md)
- [Trainer](../user-guide/trainer.md)
- [Hooks](../user-guide/hooks.md)
