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

```python
from molix.ff_export import LammpsWriter

writer = LammpsWriter()
writer.write_data("system.data", batch, masses=masses)
writer.write_settings("system.ff", ff, batch, masses=masses)
# internally: ff.clone().realize(batch) then molpy.io.write_lammps_*
```

`batch` must be a nested `TensorDict` with `atoms.pos` and optional
`bonds` / `angles` / `propers` namespaces using `atomi` … `atoml`
columns.

This path is independent of `pair_style molnex` (AOTI neural potentials).

## Compile to a ForceSpec

`ForceFieldCompiler` still translates `PotentialIR` through peer adapters
when you need a JSON force-spec:

```python
from molix.ff_export import ForceFieldCompiler, OpenMMAdapter

spec = ForceFieldCompiler(OpenMMAdapter()).compile(ff.to_ir())
```

Bond `k = 100` kcal mol⁻¹ Å⁻² → OpenMM `41840` is

```python
from molix.units import units
(100.0 * units.parse("kilocalorie_per_mole / angstrom ** 2")).to(
    "kilojoule_per_mole / nanometer ** 2"
).magnitude
```
