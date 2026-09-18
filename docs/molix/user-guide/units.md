# Units

`molix.units.units` is a shared `molpy.UnitSystem`. Conversions use
pint-style quantities — do not restate numeric factors.

```python
from molix.units import units

(1.0 * units.k_B).to("eV / kelvin")
(1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2")
(1.0 * units.kilocalorie_per_mole).to(units.kilojoule_per_mole)
(100.0 * units.parse("kilocalorie_per_mole / angstrom ** 2")).to(
    "kilojoule_per_mole / nanometer ** 2"
)
```

The last line is the Class-I bond golden: `k = 100` kcal mol⁻¹ Å⁻² →
`41840` kJ mol⁻¹ nm⁻².

## Presets

`UnitSystem.preset("real")` is LAMMPS `units real` (kcal/mol, Å, fs).
`UnitSystem.preset("openmm")` is OpenMM MD units (kJ/mol, nm).
`UnitSystem.preset("metal")` is eV, Å, ps.

The in-process MD integrator advances in (amu, Å, fs). Scale a potential
into that energy unit with `.to("amu * angstrom ** 2 / femtosecond ** 2")`.

`DEAD_PAIR_CUTOFF_FACTOR` on the same module is not a physical unit: it
is the padding used to silence dead neighbour edges.
