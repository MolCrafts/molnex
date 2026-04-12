# Core Concepts

This page defines the project vocabulary used across the docs.

## Layer

A layer is a package-level responsibility in the framework. In MolNex, the main layers are execution, representation, composition, and reference model assembly.

## Execution

Execution means the training-time and evaluation-time machinery that runs a model through batches, state transitions, hooks, and orchestration. In MolNex, that belongs to `molix`.

## Representation

Representation means the learned transformation from molecular inputs into internal features that higher layers can consume. In MolNex, that belongs to `molrep`.

## Composition

Composition means downstream modeling logic that turns learned features into structured outputs or potential-based assemblies. In MolNex, that belongs to `molpot`.

## Reference Model Family

A reference model family is a concrete assembled architecture built from lower-level modules. It is an example of how the stack is composed, not a replacement for the stack. In MolNex, that belongs to `molzoo`.

## Ownership

Ownership means the package that should define, maintain, and evolve a concept. If a piece of code has no clear owner, the architecture will drift.

## Boundary

A boundary is the point where one package stops owning a concern and another package begins. The docs emphasize boundaries because they determine where new code should go and how packages are allowed to depend on each other.

## Next Pages

- [Architecture](architecture.md)
- [Package guide: molix](packages/molix.md)
- [Package guide: molrep](packages/molrep.md)
- [Package guide: molpot](packages/molpot.md)
- [Package guide: molzoo](packages/molzoo.md)
