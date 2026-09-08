# Force-field export

`OPLSAAForceField` inherits `molpot.forcefield.ForceField`, which **is** a
`molpy.ForceField`. Styles and native types are molpy's; `to_potentials()` is
overridden to return a torch `Potentials` module. File export is `molpy.io`.

Units are pint-style via `molix.units.units` (a `molpy.UnitSystem`).

## Embed as a PyTorch model

```python
import torch
from molix.md import MD, VelocityVerletIntegrator
from molpot.forcefield import OPLSAAForceField

ff = OPLSAAForceField().with_params(bonds={"k": k, "r0": r0}, ...)
pots = ff.to_potentials()          # BasePotential / nn.Module
out = pots(batch)                  # writes graphs.energy, atoms.forces
E = pots.calc_energy(batch)

md = (
    MD(torch.float64)
    .set_forcefield(ff)            # the ForceField itself — it is an MDForceField
    .set_neighbors(nl)
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

`set_forcefield` takes the **ForceField**, not a wrapper around `pots`: `MD`
calls `ff.to_potentials()` itself on every setter. The energy scale is not a
keyword either — it is resolved from `ff.units` (`"real"` → kcal/mol,
`"metal"` → eV) by `molix.units.preset_energy_to_md`, so an energy counted in
the ForceField's own unit system reaches the integrator in amu·Å²/fs² without
anyone writing a conversion factor.

## LAMMPS files (molpy.io)

A deck is three files: the *data* file (atoms, box, topology), a bonded
*include* and a *pair* include. Only the first two are molpy's. A field that
declares pair styles has no faithful spelling on the molrs side — LAMMPS needs
`hybrid/overlay` and one explicit `pair_coeff` per type pair — so molnex writes
that block itself, from the force spec the LAMMPS adapter translates:

```python
from molix.ff_export import LammpsAdapter, LammpsWriter

writer = LammpsWriter()
writer.write_data("system.data", batch, masses=masses)
# internally: ff.clone().realize(batch) then molpy.io.write_lammps_forcefield
writer.write_settings(
    "system.in.settings", ff, batch, masses=masses, skip_pair_style=True
)
writer.write_pair_block("system.in.pair", LammpsAdapter().translate(ff.to_ir()))
```

`batch` must be a nested `TensorDict` with `atoms.pos` and optional
`bonds` / `angles` / `propers` namespaces using `atomi` … `atoml`
columns.

`write_data` reads the box from `batch["graphs", "cell"]` (Å, rows are the
lattice vectors) when it is there and falls back to a padded open box when it is
not — that one read is the periodic/open switch of the whole export. Atom types
are *identity types*, one per atom, named `t1`, `t2`, … zero-padded to a common
width (`ForceField.identity_type_label`): LAMMPS rejects a type label starting
with a digit, and molrs numbers types by sorting the labels as text, so unpadded
`t1 … t26` would sort `t1, t10, …, t2, t20` and hand every `pair_coeff i j`
another atom's Lennard-Jones well depth ε (kcal/mol) and diameter σ (Å).

`skip_pair_style=True` drops every `pair:*` style from the realized clone, so
the settings include is bonded-only — no `pair_style`, no `pair_coeff`, no
`special_bonds` — and the two includes cannot disagree about the nonbonded
model. It also carries no `units` line, because LAMMPS refuses `units` once
`read_data` has made a box. A bonded-only field needs neither the flag nor the
pair block; `write_settings` alone is then the whole include.

`write_pair_block` renders the rest: the `pair_style` line, one explicit
`pair_coeff i j` for every `i <= j` mixed with molnex's own Lorentz–Berthelot
rule (LAMMPS mixes geometrically in σ, and under `hybrid/overlay` it does not
mix cross pairs at all), `pair_coeff * *` for the Coulomb sub-style, then
`kspace_style pppm`, `kspace_modify gewald` — the model's own Ewald splitting
parameter, derived from its Gaussian smearing width rather than picked by LAMMPS
from the requested accuracy — and `special_bonds` from `ff.nonbonded_scaling`.

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
