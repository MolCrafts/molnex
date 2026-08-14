# Force-field export

`ClassIForceField` / `OPLSForceField` inherit `molpot.forcefield.ForceField`,
which **is** a `molpy.ForceField`. Styles and native types are molpy's;
`to_potentials()` is overridden to return a torch `Potentials` module.
File export is `molpy.io`.

Units are pint-style via `molix.units.units` (a `molpy.UnitSystem`).

## Embed as a PyTorch model

```python
from molix.md import MD, PotentialForceField, VelocityVerletIntegrator
from molix.units import units
from molpot.forcefield import OPLSForceField

ff = OPLSForceField().with_params(bonds={"k": k, "r0": r0}, ...)
pots = ff.to_potentials()          # BasePotential / nn.Module
out = pots(batch)                  # writes graphs.energy, atoms.forces
E = pots.calc_energy(batch)

kcal_to_md = (1.0 * units.kilocalorie_per_mole).to(
    "amu * angstrom ** 2 / femtosecond ** 2"
).magnitude
md = (
    MD()
    .set_forcefield(PotentialForceField(pots, batch, energy_scale=1 / kcal_to_md))
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

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
