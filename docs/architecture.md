# Architecture

This page answers one question: how is MolNex put together?

MolNex is a layered framework split into four packages:

- `molix`: execution, training lifecycle, orchestration, and data flow
- `molrep`: reusable representation learning modules
- `molpot`: compositional modeling and potential construction
- `molzoo`: assembled reference model families

## Stack Layout

Read the stack from bottom to top in terms of reuse, and from top to bottom in terms of execution.

`molrep` provides reusable modeling modules.  
`molpot` builds compositional downstream modeling on top of learned representations when that layer is needed.  
`molzoo` assembles lower-level parts into reference model families.  
`molix` executes training and evaluation across those layers.

That means MolNex is not centered on one top-level model class. It is centered on explicit package ownership.

## Package Boundaries

### `molix`

Owns execution concerns: loops, state, hooks, orchestration, and training-time control flow.

### `molrep`

Owns feature construction and representation learning: embeddings, interaction blocks, and readout primitives.

### `molpot`

Owns structured downstream modeling: composition layers, parameterization, and potential construction.

### `molzoo`

Owns reference assemblies: curated model families built from the lower-level stack.

## Separation Rationale

These layers change for different reasons:

- execution code changes when workflow requirements change
- representation code changes when modeling ideas change
- composition code changes when downstream structure changes
- reference models change when the project curates new assembled architectures

Keeping them separate prevents one package from silently becoming responsible for everything.

## Next Pages

- [Core concepts](concepts.md)
- [Package guide: molix](packages/molix.md)
- [Package guide: molrep](packages/molrep.md)
- [Package guide: molpot](packages/molpot.md)
- [Package guide: molzoo](packages/molzoo.md)
