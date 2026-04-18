# molpot

`molpot` is the potentials and composition layer.

## Ownership

`molpot` owns:

- potential terms
- output parameterization
- compositional modeling layers
- structured downstream assemblies built from learned representations

## Non-Goals

`molpot` should not own:

- generic training infrastructure
- the full representation learning stack
- reference model families
- every domain-specific concern in the project

## Stack Position

`molpot` sits between reusable learned features and fully assembled models. It consumes lower-level modeling outputs and exposes structured composition layers that can later be trained by `molix` or packaged into reference models by `molzoo`.

## Next Pages

- [molpot implementation docs](../molpot/index.md)
- [Architecture](../architecture.md)
