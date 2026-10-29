# Data Loading

MolNex speaks two data shapes and nothing else: a molpy `Frame` at rest, and a
nested `TensorDict` in the model.

1. a molpy `Frame` is the unit of data at rest — a record of a MolHub
   dataset, or built in memory
2. `DataLoader(collate_fn=collate_molecules)` turns frames into one nested
   `TensorDict`; `molix.data.TensorFrame` is its typed view
3. `Trainer` passes that batch to the model and loss function

Step 1 is where the boundary sits: producing those frames — downloading a
dataset, parsing it, preprocessing it and caching it — is **dataset
acquisition**, which lives in [MolHub](https://github.com/molcrafts/molhub).
molnex owns steps 2 and 3, plus the dataset protocol (`molix.data.BaseDataset`)
they read through. molnex defines no file format of its own.

## Reading MolHub Datasets

`molhub.load(name)` opens a named dataset as a map-style dataset whose records
are `Sequence[Frame]` — one geometry, or a scan of conformers:

```python
import molhub

ds = molhub.load("themol/mbis")  # MolHub owns acquisition and the on-disk shape
```

A record is the indivisible unit a sampler packs, and the collate turns every
frame in it into its own graph. Split by record (`molix.data.SubsetDataset`,
`group_split_indices`) so a scan never straddles train and val. The producing
side is documented in MolHub's
[`docs/reference/python.md`](https://github.com/molcrafts/molhub/blob/main/docs/reference/python.md).

## Collation

```python
from torch.utils.data import DataLoader
from molix.data.collate import collate_molecules

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_molecules)
```

`collate_molecules` produces a plain nested `TensorDict` whose namespaces each
carry their own batch size:

- Atom-level fields (`Z`, `pos`, `batch`) → `atoms` (batch_size=[N_total]).
  The `x`/`y`/`z`, `fx`/`fy`/`fz` and `vx`/`vy`/`vz` columns stack into
  `atoms.pos`, `atoms.forces` and `atoms.vel`; every other atom column is
  carried through as `atoms.<name>`.
- Pair fields (`pair_index`, `pair_diff`, `pair_dist`), written by the
  neighbour-list task → `pairs` (batch_size=[E_total])
- Graph-level metadata (`num_atoms`) + graph targets (the frame's `meta`
  scalars) → `graphs` (batch_size=[B]). A frame's `box` becomes
  `graphs.cell` `(B, 3, 3)` with the lattice vectors as rows (molpy's
  `Box.matrix` holds them as columns; the collate transposes). Either every
  frame in a batch carries a box or none does.
- Covalent topology (`bond_index`, `bond_types`), when present → `bonds` (batch_size=[])

The batch keeps the frames' precision — a molpy `Frame` is fp64 at rest and the
collate narrows nothing on its own. The floating dtype a pipeline wants is the
loader's policy: `DataModule` builds each batch at its `ftype` by passing
`collate_molecules(frames, dtype=...)`; a raw caller does the same.

Read it with tuple keys: `batch["atoms", "Z"]`, `batch["pairs", "pair_index"]`,
`batch["graphs", "energy"]`.

There is one collate path. `collate_molecules` takes molpy `Frame` objects
and returns the batch. The `collate_packed` fast path that
used to sit beside it was deleted on 2026-09-07 — it re-gathered every tensor
through an identity index, which was a full copy of the batch for nothing.

## Preprocessing Tasks

Preprocessing that belongs to dataset acquisition is scheduled by MolHub.
Work that needs the collated batch is a `molix.data.BatchTask`, run by the
loader on every batch:

- **`molix.neighbor.NeighborList`** writes `pair_index`, `pair_diff` and
  `pair_dist` for every pair within `cutoff`, rebased onto the batch. Callers
  holding raw tensors rather than a batch use `pairs(pos, box_vectors)`
  instead.
- **`molix.data.EnergyResidual(dress)`** writes `graphs.e0` (the per-graph sum
  of the dress over real atoms, float64) and `graphs.energy_residual` (the
  label minus `e0`), and leaves `graphs.energy` untouched.

### Atomic dress

An atomic dress is a constant per-element energy `c[Z]`. Subtracting it from
the label lets the network learn the residual; adding it back gives the
absolute energy. Unit conversion and constant labels are still MolHub's, but
**fitting the dress is molnex's**, and it sees the training split only — fit
it on the whole dataset and the val labels leak into the baseline.

The table is one `molix.data.AtomicDress`. Each way of obtaining it is its own
type, and each returns an `AtomicDress`:

- `LeastSquaresDress(elems).fit_dataset(train_ds)` — PiNN's no-intercept least
  squares on element counts, one row per frame. `fit(counts, energy)` takes
  the count matrix directly when you already have it.
- `SuppliedDress({Z: c}).table` — a published table (isolated-atom energies, a
  previous fit); it has no `fit` and is never refitted.

```python
from molix.core.losses import energy_loss
from molix.data import DataModule, EnergyResidual, LeastSquaresDress
from molix.schema import ENERGY_KEY

dress = LeastSquaresDress(elems=[1, 6, 8]).fit_dataset(train_ds)
# or: dress = SuppliedDress({1: -13.6, 8: -2043.8}).table
residual = EnergyResidual(dress)  # this instance stays on CPU
dm = DataModule(train_ds, val_ds, batch_tasks=[residual], ...)  # val shares it
loss_fn = energy_loss(target_key="energy_residual", pred_key=ENERGY_KEY)
```

The loss target is `energy_residual`: after the forward, `graphs.energy` holds
the prediction, so `target_key="energy"` would compare the network with
itself. The same numbers from either type give the same
`EnergyResidual.task_id`.

`molix.deploy.DressedPotential` adds the dress back onto the atom energies, so
an exported package — which LAMMPS loads with no Python in between — reports
absolute energy. The package carries the table as the buffer `dress.e0`; the
deploy metadata JSON does not. PiNet2 end to end:

```python
import copy

from molix.deploy import AtomEnergyAdapter, DressedPotential, example_inputs
from molix.deploy.contract import DYNAMIC_SHAPES
from molix.export import aoti
from molzoo.pinet import PiNetPotential, PiNetSpec

spec = PiNetSpec(atomic_numbers=[1, 6, 8])
dress = LeastSquaresDress(elems=spec.atomic_numbers).fit_dataset(train_ds)
dm = DataModule(train_ds, val_ds, batch_tasks=[EnergyResidual(dress)], ...)
model = PiNetPotential(spec, wrt=("atoms.pos",), dtype=torch.float32)
loss_fn = energy_loss(target_key="energy_residual", pred_key=ENERGY_KEY)
# ... train `model` (its checkpoint is the residual network) ...

export_dress = copy.deepcopy(dress).to(device)  # never move the loader's instance
adapter = AtomEnergyAdapter(DressedPotential(model, export_dress))
species = tuple(int(z) for z in export_dress.support.nonzero().flatten() if z > 0)
inputs = example_inputs(list(species), spec.r_max, torch.float32, device)
written = aoti.package(
    adapter, inputs, "out/model.pt2", dynamic_shapes=DYNAMIC_SHAPES,
    inductor_configs={"aot_inductor.use_runtime_constant_folding": True},
)
# DeployMetadata(..., species=species): no e0 field; the package carries it.
```

The add runs in float64 and is stored in the model's dtype, so a float32
package rounds the absolute energy at the ulp of `|c[Z]|`. For an undressed
forward in Python, `dress.restore(frame["graphs", "energy"], frame["graphs",
"e0"])` is the same add on the per-graph totals. In-process MD is unchanged:
it logs whatever `graphs.energy` its force field's potentials write.

Apply the dress **once**:

- A training step runs `EnergyResidual` *or* `DressedPotential`, never both:
  together they give a prediction of `s·o + e0` against a target of
  `E − e0`. Never `restore` the output of `DressedPotential`.
- Do not combine either with a potential that adds its own per-element
  energies inside `energy` (MACE's E0, `AtomicReferenceEnergy`).

For the full batch structure, see [Batch Schema](../explanation/batch-schema.md).
