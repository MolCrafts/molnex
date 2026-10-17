# MolZoo Tutorial

This tutorial explains the whole MolZoo workflow. It is not a model-by-model
reference page. The goal is to show how a MolZoo model is organized inside a
MolNex project and where the neighboring packages enter the pipeline.

The concrete model used below is Allegro because it is the one documented in
detail. MACE (`molzoo.mace`) and PiNet (`molzoo.pinet`) follow the same shape.

## 1. Know the Package Boundary

Every MolZoo family has the same three pieces:

- a torch-free spec (`AllegroSpec`, `PiNetSpec`, …);
- a feature **encoder** that writes learned features into the batch
  (`AllegroEncoder` → `pairs.edge_features`, `PiNetEncoder` →
  `atoms.node_features`);
- named **products** that subclass the encoder and add the energy heads
  (`AllegroPotential`, `PiNetPotential`, `MaceMatpes`, …), with
  `forward = Derivative(energy, wrt)` and, where an official checkpoint
  exists, `.load(path)`.

MolZoo does not own:

- dataset acquisition, parsing and caching (MolHub)
- neighbor-list construction (`molix.neighbor.NeighborList`)
- the derivative op (`molix.autograd.Derivative`)
- training loops and losses (`molix`)

The complete project shape is:

```text
dataset (molix.data.BaseDataset, e.g. from MolHub)
  -> molix.data.DataModule (collate -> NeighborList batch task)
  -> collated nested TensorDict
  -> molzoo product (encoder + energy heads + Derivative)
  -> graphs.energy / atoms.forces written into the batch
  -> losses
  -> molix.Trainer
```

For Allegro specifically:

```text
collated nested TensorDict
  -> AllegroPotential(..., wrt=("atoms.pos",))
       AllegroEncoder body       -> pair features (E, F(L+1))
       edge readout + reduce     -> atoms.energy (N,) float64
       graph sum                 -> graphs.energy (B,) float64
       Derivative                -> atoms.forces (N, 3)
```

## 2. Prepare the Batch

MolZoo models read a post-collate batch — the nested `TensorDict` with
`atoms` / `pairs` / `graphs` namespaces — not a molpy `Frame`. Allegro reads:

```text
("atoms", "Z")            (N,)
("atoms", "pos")          (N, 3)
("atoms", "batch")        (N,)     graph membership
("pairs", "pair_index")   (E, 2)   [:, 0] = source / centre
("pairs", "pair_diff")    (E, 3)   optional, imaged pos[target] - pos[source]
("atoms", "mask")         (N,)     optional, real-atom mask of a padded batch
graphs                    batch_size=[B]
```

A dataset carries no neighbor graph; `NeighborList` builds it on every
collated batch as a batch task. Its default `symmetry=True` gives the full
bidirectional list Allegro needs:

```python
from molix.data import DataModule
from molix.neighbor import NeighborList

# train_ds, val_ds: any molix.data.BaseDataset (for example, loaded through MolHub)
dm = DataModule(
    train_ds,
    val_ds,
    batch_tasks=[NeighborList(cutoff=5.0)],
    batch_size=32,
)
dm.setup("fit")
```

See [Data Modules](../../molix/user-guide/data-modules.md) for collation,
padding and capacity-first packing.

The edge convention is fixed:

```text
pair_index[:, 0] = source / center
pair_index[:, 1] = target / neighbor
pair_diff        = pos[target] - pos[source]
```

## 3. Compute Dataset Statistics

Allegro uses `avg_num_neighbors` to normalize environment sums and the edge
energy reduction. Compute it once from the training data with the same
neighbor-list settings:

```python
def estimate_avg_num_neighbors(loader, max_batches=100):
    total_pairs = 0
    total_atoms = 0
    for step, batch in enumerate(loader):
        total_pairs += int(batch["pairs", "pair_index"].shape[0])
        total_atoms += int(batch["atoms", "Z"].shape[0])
        if step + 1 >= max_batches:
            break
    return total_pairs / max(total_atoms, 1)


avg_num_neighbors = estimate_avg_num_neighbors(dm.train_dataloader())
```

It is a constant of the model, not of the batch: never recompute it per
batch. (With a padded batch, count only real atoms and live pairs.)
`per_type_energy_shifts` are usually a linear fit of the reference energies
on composition.

## 4. Build the Model

The energy model is one class over the spec, plus `wrt`:

```python
from molzoo.allegro import AllegroPotential, AllegroSpec

model = AllegroPotential(
    AllegroSpec(
        atomic_numbers=[1, 6, 7, 8],
        r_max=5.0,
        avg_num_neighbors=avg_num_neighbors,
        l_max=2,
        num_layers=2,
        num_scalar_features=64,
        num_tensor_features=16,
    ),
    wrt=("atoms.pos",),  # forces; add "graphs.strain" for the virial
)
```

`model(batch)` writes `graphs.energy` `(B,)` (float64), `atoms.energy` `(N,)`
and `atoms.forces` `(N, 3)` into the batch in place.

To start from an official Allegro model instead, load its package:

```python
model = AllegroPotential.load("Allegro-OAM-L-0.1.nequip.zip", wrt=("atoms.pos",))
```

When only the features are wanted — for a custom head — use the encoder:

```python
from molzoo.allegro import AllegroEncoder, AllegroSpec

encoder = AllegroEncoder(
    AllegroSpec(atomic_numbers=[1, 6, 7, 8], r_max=5.0, avg_num_neighbors=avg_num_neighbors)
)
# encoder(batch)["pairs", "edge_features"]  ->  (E, encoder.output_dim)
```

## 5. Train With Molix

The model writes its predictions into the batch it is handed, under the same
names a label would naturally use. Keep the reference values under other
names (`graphs.energy_ref`, `atoms.forces_ref`), or the loss compares the
prediction with itself:

```python
import torch
import torch.nn.functional as F
from molix.core.trainer import Trainer


def loss_fn(pred, batch):
    energy = F.mse_loss(pred["graphs", "energy"], batch["graphs", "energy_ref"].double())
    forces = F.mse_loss(pred["atoms", "forces"], batch["atoms", "forces_ref"])
    return energy + 10.0 * forces


trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
    device="cuda",
)

trainer.train(dm, max_epochs=100)
```

The default training step marks the derivative leaves before the traced
body, so a force-supervised step compiles as one graph
(`Trainer.compile(step=True)`).

## 6. Organize Project Code

For a real experiment, keep the model wiring small and explicit:

```text
configs/
  allegro-qm9.toml
src/
  my_project/
    data.py       # datasets and DataModule construction
    models.py     # the molzoo product and its spec fields
    train.py      # loss and Trainer setup
```

Model-family details belong in MolZoo user guides. Experiment-specific choices
belong in your project config.

## Next Page

Read [Allegro User Guide](../user-guide/allegro.md) for the equations, tensor
shapes, official-package loading, and a lower-level hand-built batch tutorial.
