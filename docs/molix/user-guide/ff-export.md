# Force-field export

`OPLSAAForceField` inherits `molpot.forcefield.ForceField`, which **is** a
`molpy.ForceField`. Styles and native types are molpy's; `to_potentials()` is
overridden to return a torch `Potentials` module. File export is `molpy.io`.

Units are pint-style via `molix.units.units` (a `molpy.UnitSystem`).

## Embed as a PyTorch model

```python
import torch
from molix.md import MD, NeighborList, VelocityVerletIntegrator
from molpot.forcefield import OPLSAAForceField

ff = OPLSAAForceField().with_params(bonds={"k": k, "r0": r0}, ...)
pots = ff.to_potentials()          # BasePotential / nn.Module
out = pots(batch)                  # writes graphs.energy, atoms.forces
E = pots.calc_energy(batch)

md = (
    MD(torch.float64)
    .set_forcefield(ff)            # the ForceField itself — it is an MDForceField
    # 15 Å is this field's periodic radius — see "What the batch must carry"
    .set_neighbors(NeighborList(cell=cell, cutoff=15.0, positions=pos, skin=1.0))
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

`set_forcefield` takes the **ForceField**, not a wrapper around `pots`: `MD`
calls `ff.to_potentials()` itself on every setter. The energy scale is not a
keyword either — it is resolved from `ff.units` (`"real"` → kcal/mol,
`"metal"` → eV) by `molix.units.preset_energy_to_md`, so an energy counted in
the ForceField's own unit system reaches the integrator in amu·Å²/fs² without
anyone writing a conversion factor.

### What the batch must carry

`Potentials` prices a pair list; it never enumerates one. A Class-I field
therefore needs two batch tasks, and both are declarations rather than settings:

```python
from molix.data import TotalCharge
from molix.data import DataModule

dm = DataModule(train_ds, val_ds, batch_tasks=[ff.neighbor_list(), TotalCharge()])
```

`ff.neighbor_list()` is the list *this field* is priced against — the largest
radius among the terms that will run, resolved from the styles the field
declares rather than restated by the caller. `pbc` picks which Coulomb term
that is: `ff.neighbor_list(pbc=True)` covers the Ewald split's real-space
cutoff (15 Å for `OPLSAAForceField`, i.e. 3 × the Gaussian smearing width),
while `pbc=False` covers the declared open `cutoff`. An `OPLSAAForceField` whose
`pair:coulomb` is bare *and* open has no finite radius at all — that is the
untruncated N(N−1)/2 sum — so `ff.neighbor_list()` refuses rather than invent
one; declare `def_pairstyle("coulomb", cutoff=…)` to state the truncation you
are running, or say the system is periodic.

`TotalCharge()` writes `graphs.total_charge` (B,) in e — `0.0` means *this
pipeline's graphs are neutral*, declared once instead of pasted into every
collate site. A dataset carrying its own charge column keeps it; the task only
casts an integer column to the batch float dtype. A charge model is fail-closed
on this column, and `Potentials.bind_topology` cross-checks the per-graph sum of
`atoms.charge` against it, so a declaration the data does not support is caught.

At a collate site that bypasses the `DataModule`, run the same two tasks by
hand: `for task in (ff.neighbor_list(), TotalCharge()): batch = task.execute(batch)`.

## LAMMPS files (molpy.io)

A deck is three files: the *data* file (atoms, box, topology), a bonded
*include* and a *pair* include. Only the first two are molpy's. A field that
declares pair styles has no faithful spelling on the molrs side — LAMMPS needs
`hybrid/overlay` and one explicit `pair_coeff` per type pair — so molnex writes
that block itself, from the force spec the LAMMPS adapter translates:

```python
from molix.ff_export import LammpsAdapter, LammpsWriter

writer = LammpsWriter()
types = ff.assign_types(batch, masses=masses)   # one condensation, three files
writer.write_data("system.data", batch, types=types, masses=masses)
# internally: ff.clone().realize(batch, types=types) then molpy.io.write_lammps_forcefield
writer.write_settings(
    "system.in.settings", ff, batch, types=types, masses=masses, skip_pair_style=True
)
writer.write_pair_block(
    "system.in.pair", LammpsAdapter().translate(ff.to_ir()), types=types
)
```

`batch` must be a nested `TensorDict` with `atoms.pos` and optional
`bonds` / `angles` / `propers` namespaces using `atomi` … `atoml`
columns.

`ForceField.assign_types` is the **one** owner of the typing, and all three
files take the same result because they address the types by the same numbers —
the data file labels them, the bonded include names its coefficients after them
and the pair block writes them as `pair_coeff i j`. A LAMMPS type is an
equivalence class of *parameters*: two atoms with the same (mass, charge, ε, σ)
are one atom type and two interactions between the same atom types with the same
constants are one bonded type, so the 26-atom water/ion box of the parity tool
exports 4 atom types and 10 `pair_coeff` lines rather than 26 and 351. (Before
2026-09-08 molnex exported *identity* types — one per atom — which was correct
and quadratic: ledger debt **F25**.) Labels are `t1`/`b1`/`a1`/`d1`/`i1`-style,
zero-padded to the type count: LAMMPS rejects a type label starting with a
digit, and molrs numbers types by sorting the labels as text, so unpadded
`t1 … t26` would sort `t1, t10, …, t2, t20` and hand every `pair_coeff i j`
another type's Lennard-Jones well depth ε (kcal/mol) and diameter σ (Å).

`write_data` reads the box from `batch["graphs", "cell"]` (Å, rows are the
lattice vectors) when it is there and falls back to a padded open box when it is
not — that one read is the periodic/open switch of the whole export.

`skip_pair_style=True` drops every `pair:*` style from the realized clone, so
the settings include is bonded-only — no `pair_style`, no `pair_coeff`, no
`special_bonds` — and the two includes cannot disagree about the nonbonded
model. It also carries no `units` line, because LAMMPS refuses `units` once
`read_data` has made a box. A bonded-only field needs neither the flag nor the
pair block; `write_settings` alone is then the whole include.

`write_pair_block` renders the rest: the `pair_style` line, one explicit
`pair_coeff i j` for every `i <= j` *type* pair mixed with the model's own combining rule
(LAMMPS mixes geometrically in σ, and under `hybrid/overlay` it does not
mix cross pairs at all), `pair_coeff * *` for the Coulomb sub-style, then
`kspace_style pppm`, `kspace_modify gewald` — the model's own Ewald splitting
parameter, derived from its Gaussian smearing width rather than picked by LAMMPS
from the requested accuracy — and `special_bonds` from `ff.nonbonded_scaling`.

That rule is the `mixing` option of the `pair:lj/cut` style —
`"lorentz-berthelot"` (σ arithmetic, ε geometric) by default and `"geometric"`
(both, the Jorgensen 1996 rule) for `OPLSAAForceField`, declared once with
`def_pairstyle("lj/cut", mixing=…)`, recorded on the model's
`NonbondedMethodRecord`, and read back here so the deck's `pair_coeff` lines
mix exactly the way molnex evaluated.

Source the files in this order; the includes are coefficient lines, so they need
the box and the types `read_data` establishes:

```
units real
atom_style full
read_data system.data
include system.in.settings
include system.in.pair
```

`interface/tests/parity_classical.py` is the recorded check that a deck written
this way still means the same energy once LAMMPS, not molnex, evaluates it:
`--case butane` (bonded-only, open boundary) agrees to 2.6e-12 in relative
energy and `--case periodic` (8 waters plus Na⁺ and Cl⁻ in a 12 Å cube, switched
Lennard-Jones and Ewald-split Coulomb) to 9.9e-9. It drives a real `lmp` binary
and exits 77 — ctest's SKIP — without one.

This path is independent of `pair_style molnex` (AOTI neural potentials). That
one is `molix.engine.export_for_lammps`, which takes an optional `manifest=` and
writes `<name>.manifest.json` beside the `.pt2` and its `meta.json`:

```python
from molix.engine import export_for_lammps

export_for_lammps(
    model,
    "export/",
    species=[1, 6],
    cutoff=5.0,
    receptive_field_layers=3,
    manifest=ff.manifest(profile="A0").to_dict(),
)
```

The manifest travels as a plain mapping — `molix.engine` never imports molpot —
and is written verbatim. `meta.json` is byte-identical whether or not one is
passed, and the C++ side reads only its `lammps` block.

## Compile to a ForceSpec

`ForceFieldCompiler` still translates `PotentialIR` through peer adapters
when you need a JSON force-spec:

```python
from molix.ff_export import ForceFieldCompiler, OpenMMAdapter

spec = ForceFieldCompiler(OpenMMAdapter()).compile(ff.to_ir())
```

Bond `k = 100` kcal mol⁻¹ Å⁻² → OpenMM `41840` kJ mol⁻¹ nm⁻² is

```python
from molix.units import units
(100.0 * units.parse("kilocalorie_per_mole / angstrom ** 2")).to(
    "kilojoule_per_mole / nanometer ** 2"
).magnitude
```
