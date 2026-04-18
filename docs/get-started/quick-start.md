# Quick Start Guide

Train a minimal model using MolNex's nested TensorDict batch format.

## 1. Define the Model

Models receive a `GraphBatch` (nested TensorDict) and access data through nested keys:

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 16)
        self.head = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch):
        Z = batch["atoms", "Z"]
        graph_idx = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]

        h = self.embedding(Z)
        pooled = torch.zeros(num_graphs, h.shape[-1], device=h.device)
        pooled.index_add_(0, graph_idx, h)
        return {"energy": self.head(pooled).squeeze(-1)}
```

## 2. Build a Batch

In real usage, `collate_molecules` builds the `GraphBatch` from sample dicts. For a quick demo we can build one manually:

```python
from molix.data.types import AtomData, EdgeData, GraphData, GraphBatch

atoms = AtomData(
    Z=torch.tensor([6, 1, 1, 1, 1]),
    pos=torch.randn(5, 3),
    batch=torch.zeros(5, dtype=torch.long),
    batch_size=[5],
)
edges = EdgeData(
    edge_index=torch.zeros(0, 2, dtype=torch.long),
    bond_diff=torch.zeros(0, 3),
    bond_dist=torch.zeros(0),
    batch_size=[0],
)
graphs = GraphData(
    num_atoms=torch.tensor([5]),
    energy=torch.tensor([-40.5]),
    batch_size=[1],
)
batch = GraphBatch(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])
```

## 3. Configure the Trainer

```python
from molix import Trainer

def loss_fn(predictions, batch):
    target = batch["graphs", "energy"]
    return torch.nn.functional.mse_loss(predictions["energy"], target)

trainer = Trainer(
    model=SimpleModel(),
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
)
```

## 4. Train

```python
class DemoDataModule:
    def __init__(self, batch):
        self.batch = batch

    def train_dataloader(self):
        for _ in range(20):
            yield self.batch

    def val_dataloader(self):
        for _ in range(5):
            yield self.batch

trainer.train(DemoDataModule(batch), max_epochs=3)
```
