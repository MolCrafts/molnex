# MolZoo Tutorial

This tutorial explains the whole MolZoo workflow. It is not a model-by-model
reference page. The goal is to show how a MolZoo encoder is organized inside a
MolNex project and where the neighboring packages enter the pipeline.

The concrete encoder used below is Allegro because that is the model currently
documented in detail. The same organization should be used for future MolZoo
families.

## 1. Know the Package Boundary

MolZoo models are reference encoders. They transform a molecular graph batch
into learned features.

They do not own:

- raw dataset parsing
- neighbor-list construction
- training loops
- energy readout
- force derivation

The complete project shape is:

```text
source samples
  -> molix.data pipeline
  -> GraphBatch
  -> molzoo encoder
  -> learned features
  -> molpot readout / potential head
  -> losses
  -> molix.Trainer
```

For Allegro specifically:

```text
GraphBatch
  -> molzoo.Allegro
  -> ("edges", "edge_features")
  -> molpot.heads.EdgeEnergyHead
  -> graph energy
```

## 2. Prepare the Batch

MolZoo encoders expect a post-collate `GraphBatch`, not a raw sample dict.
For Allegro the required fields are:

```text
("atoms", "Z")
("edges", "edge_index")
("edges", "bond_diff")
("edges", "bond_dist")
```

In a real pipeline, use `NeighborList(symmetry=True)` so every undirected
neighbor pair is represented as two directed edges.

```python
from molix.data import DataModule, NeighborList, Pipeline

pipe = (
    Pipeline("allegro-run")
    .add(NeighborList(cutoff=5.0, symmetry=True))
    .build()
)

dm = DataModule(
    source=source,
    pipeline=pipe,
    batch_size=32,
)
dm.setup("fit")
```

The edge convention is fixed:

```text
edge_index[:, 0] = source / center
edge_index[:, 1] = target / neighbor
bond_diff        = pos[target] - pos[source]
```

## 3. Compute Dataset Statistics

Some encoders and readouts need dataset-level constants. Allegro uses
`avg_num_neighbors` to normalize environment sums and edge-energy reductions.

For experiments, compute it from the same neighbor-list settings used for
training:

```python
def estimate_avg_num_neighbors(loader, max_batches=100):
    total_edges = 0
    total_atoms = 0
    for step, batch in enumerate(loader):
        total_edges += int(batch["edges", "edge_index"].shape[0])
        total_atoms += int(batch["atoms", "Z"].shape[0])
        if step + 1 >= max_batches:
            break
    return total_edges / max(total_atoms, 1)


avg_num_neighbors = estimate_avg_num_neighbors(dm.train_dataloader())
```

Use the same value in the encoder and the downstream readout. Do not recompute
it per batch during training.

## 4. Build the Encoder and Readout

```python
from molpot.heads import EdgeEnergyHead
from molzoo import Allegro

encoder = Allegro(
    num_elements=119,
    r_max=5.0,
    avg_num_neighbors=avg_num_neighbors,
    l_max=2,
    num_layers=2,
    num_scalar_features=64,
    num_tensor_features=16,
)

readout = EdgeEnergyHead(
    input_dim=encoder.output_dim,
    avg_num_neighbors=avg_num_neighbors,
)
```

This is the MolNex split:

- `Allegro` produces edge features.
- `EdgeEnergyHead` maps those features to energy.

## 5. Wrap the Model

Molix `Trainer` expects one model. The usual pattern is to wrap encoder and
readout in a small `torch.nn.Module`.

```python
import torch.nn as nn


class AllegroEnergyModel(nn.Module):
    def __init__(self, encoder, readout):
        super().__init__()
        self.encoder = encoder
        self.readout = readout

    def forward(self, batch):
        batch = self.encoder(batch)
        return self.readout(batch)


model = AllegroEnergyModel(encoder, readout)
```

## 6. Train With Molix

```python
import torch
import torch.nn.functional as F
from molix.core.trainer import Trainer


def loss_fn(pred, batch):
    return F.mse_loss(pred["energy"], batch["graphs", "energy"])


trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
    device="cuda",
)

trainer.train(dm, max_epochs=100)
```

Force training uses the same organization, but the readout/model must produce
forces by differentiating energy with respect to positions, and the loss must
include the atom-level force target.

## 7. Organize Project Code

For a real experiment, keep the model wiring small and explicit:

```text
configs/
  allegro-qm9.toml
src/
  my_project/
    data.py       # DataSource and Pipeline construction
    models.py     # encoder + readout wrapper
    train.py      # Trainer setup
```

Model-family details belong in MolZoo user guides. Experiment-specific choices
belong in your project config.

## Next Page

Read [Allegro User Guide](../user-guide/allegro.md) for the equations, tensor
shapes, implementation contract, and a lower-level hand-built batch tutorial.
