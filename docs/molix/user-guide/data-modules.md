# Data Modules

`DataModule` is the last stage of the data path: it wraps two pre-built
datasets in DataLoaders that collate, shard across DDP ranks, and prefetch.
Everything upstream — downloading, per-sample transforms, caching, splitting —
happens before it.

Dataset **acquisition** — downloading, parsing, per-sample preprocessing and
writing the `molix.block/1` stores — is not part of molnex; it lives in
[MolHub](https://github.com/molcrafts/molhub). molnex starts from the cache
file MolHub wrote, read through `molix.data`.

The minimal protocol requires:

- `setup(stage)` - Prepare datasets/samplers for a stage (e.g. `"fit"`)
- `train_dataloader()` - Returns an iterable of collated training batches
- `val_dataloader()` - Returns an iterable of collated validation batches

`DataModuleProtocol` also declares `on_epoch_start(epoch)`, which the `Trainer`
calls when the module defines it (e.g. to reseed a sampler).

Each item yielded by those loaders is a nested `TensorDict` with `atoms` /
`edges` / `graphs` namespaces, see
[Batch Schema](../explanation/batch-schema.md).

## Using the Built-in DataModule

The molnex-side wiring is store → dataset → split → `DataModule`. The
cache was produced upstream by MolHub, which owns the source, the per-sample
transforms (neighbor list, atomic dress) and the fit of any dataset-level
statistic — including the rule that such a fit sees training indices only, or
the val split leaks into the baseline:

```python
from molix.data import MoleculeCollator, MoleculePadder, TotalCharge
from molix.data import DataModule, FrameCollator
import molhub
from molix.data import SubsetDataset

full = molhub.load("qm9")  # MolHub owns acquisition and the on-disk shape

train_idx = list(range(800))
val_idx = list(range(800, 1000))
train_ds = SubsetDataset(full, train_idx)
val_ds = SubsetDataset(full, val_idx)

dm = DataModule(
    train_ds,
    val_ds,
    collate_fn=FrameCollator(
        collator=MoleculeCollator(),
        padder=MoleculePadder(max_atoms=1024, max_pairs=16384, cutoff=5.0),
    ),
    batch_size=32,
)
dm.setup("fit")

for batch in dm.train_dataloader():
    # batch is a nested TensorDict
    Z = batch["atoms", "Z"]           # (N_total,)
    pos = batch["atoms", "pos"]       # (N_total, 3)
    energy = batch["graphs", "U0"]    # (B,)
    break
```

Where a target lands is decided by where its column sits in the molpy `Frame`,
not by configuration — an `atoms` column becomes an `atoms` leaf, a `meta`
scalar becomes a `graphs` leaf. `batch_tasks` carries `molix.data.BatchTask`
instances, applied in order after collation — post-collate work is configured on the DataModule, not
upstream, because it cannot be cached with the rest.

Three tasks ship with molix, and which of them a run needs is decided by the
model it feeds:

| Piece | Writes | Needed by |
|---|---|---|
| `FrameCollator(..., padder=MoleculePadder(max_atoms=, max_pairs=, max_bonds=, …, max_graphs=, cutoff=))` | DataLoader `collate_fn`: concat then ghosts + `atoms.mask` + dead `(ghost, ghost)` edges + padded valence (`bonds`/`angles`/`propers`/`impropers` dummy rows on four offset ghosts, `*.mask`) + dead graph slots (`graphs.mask`; ghosts sit in the last, sink slot) | `torch.compile` / CUDA-graph paths |
| `NeighborList(cutoff=, pbc=, …)` | the `edges` namespace — the cutoff graph (a **batch task**, after collate, before pad) | every model that reads pairs |
| `TotalCharge(charge=0.0)` | `graphs.total_charge` (B,) in e | charge models |

### One static shape for a whole dataset: capacity-first packing

A padded batch only helps `torch.compile` if *every* batch has the same
shape — atoms, edges, each valence family **and the graph count**. Fixing the
number of molecules per batch and padding each axis to `B × per-molecule max`
does give one shape, but at the sum of the largest molecules: on OMol25 4m
(3.99 M molecules, max 350 atoms / 8516 propers) a 2048-molecule batch needs
17 M proper rows while a random one fills 280 k of them. The waste is
structural to fixing `B`, so the shape is fixed the other way round:

```python
from molix.data import MoleculePadder
from molix.data import DataModule, FrameCollator

counts = {"atoms": n_atoms, "pairs": n_pairs, "propers": n_propers, ...}  # (n,) per molecule
padder = MoleculePadder.from_counts(counts, atoms=131072, cutoff=10.0)   # atoms = 2**k, the HBM knob
dm = DataModule(
    train_ds, val_ds,
    collate_fn=FrameCollator(collator=MoleculeCollator(), padder=padder),
    batch_tasks=[NeighborList(cutoff=10.0, max_num_pairs=padder.max_pairs)],
    batch_counts=counts, val_batch_counts=val_counts,
)
```

`from_counts` sizes every other cap as *dataset density × atom budget ×
slack*, never below the largest single molecule, and the graph-slot count from
the mean molecule size. `batch_counts` makes the DataModule pack train (and
`val_batch_counts` the val split, in index order) with a
`TokenBudgetBatchSampler` over `padder.budgets()` — atoms, edges, every padded
family and the graph count are all budgets, so the packer closes a batch
before any axis would overflow. The molecule count per batch floats; the
atom capacity is the one knob you calibrate on the accelerator. A molecule
that could not be packed alone is refused at construction (`padder.check`),
never discovered mid-epoch. On 4m at 131072 atoms a batch holds ~2390
molecules at 99.9 % atom fill and ~91 % on the cheap columns, with 348 k
proper rows instead of 17 M — and the largest 10 % of the dataset packs into
the same shape without a recompile, because nothing about the shape depends
on which molecules arrive.

`TotalCharge()` reads as *the graphs of this pipeline are neutral*: one auditable
declaration for the whole run instead of a zero re-spelled at every collate site.
A dataset that already carries a `total_charge` column keeps its values — the
task then only casts an integer column to the batch's float dtype.

A classical (Class-I) force field owns the first two of its needs itself, so the
pair list is never a number the caller restates:

```python
from molix.data import TotalCharge
from molix.data import DataModule

dm = DataModule(train_ds, val_ds, batch_tasks=[ff.neighbor_list(), TotalCharge()])
```

`ff.neighbor_list()` returns the `NeighborList` at the largest radius among the
pair terms that will run — see [Force-field export](ff-export.md) for the `pbc`
switch and the case it deliberately refuses.

## Minimal Custom DataModule

`collate_molecules` keeps the frames' precision (fp64 at rest); a module that
wants its batches at another dtype hands the loader
`MoleculeCollator(dtype=torch.float32)` instead — that is what `DataModule`
does with its `ftype`.

```python
from torch.utils.data import DataLoader
from molix.data.collate import collate_molecules

class MyDataModule:
    def __init__(self, train_set, val_set, batch_size=32):
        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules,
        )
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
```
