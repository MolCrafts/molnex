# Train a Graph Model

This tutorial trains against MolNex's nested `TensorDict` batch format. In real
data pipelines, `molix.data.collate.collate_molecules` creates this structure
from flat molecule samples. Here, the batch is built directly.

## 1. Define the Model

Models that receive a `GraphBatch` access data with nested tuple keys:

```python
import torch
import torch.nn as nn


class SimpleGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(119, 16)
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

```python
from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData

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

## 3. Train

```python
import torch.nn.functional as F
from molix.core.trainer import Trainer


def loss_fn(predictions, batch):
    return F.mse_loss(predictions["energy"], batch["graphs", "energy"])


class DemoDataModule:
    def __init__(self, batch):
        self.batch = batch

    def train_dataloader(self):
        for _ in range(20):
            yield self.batch

    def val_dataloader(self):
        for _ in range(5):
            yield self.batch


trainer = Trainer(
    model=SimpleGraphModel(),
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
)

trainer.train(DemoDataModule(batch), max_epochs=3)
```

## Next Steps

- [Batch Schema](../explanation/batch-schema.md)
- [Data Loading](../user-guide/data-loading.md)
- [Data Modules](../user-guide/data-modules.md)
