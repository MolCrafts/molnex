# MolPot

MolPot is the potential and composition package in MolNex. Use it for
classical potential terms, learned parameter heads, pooling, energy aggregation,
force derivation, and potential composition.

## Tutorials

- [Build a Potential](tutorials/build-a-potential.md): call a potential term and
  compose learned parameters with pairwise distances.
- [Class-I ForceField](tutorials/class-i-forcefield.md): fill an
  `OPLSAAForceField`, evaluate with torch `to_potentials()`, write LAMMPS
  via `molpy.io`.

## User Guide

- [Components](user-guide/components.md): potential terms, heads, pooling, and
  composition layers.
- [ForceField](user-guide/forcefield.md): classical styles, param tables,
  `to_potentials()`, and `to_ir()`.
- [Gradients and Forces](user-guide/gradients.md): derive forces from energies
  with autograd.

## Explanation

- [Potential Composition](explanation/potential-composition.md): how MolPot
  connects encoder features, heads, potential terms, and physical outputs.

## API

See [molpot API Reference](../api/molpot.md).
