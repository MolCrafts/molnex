# MolNex Documentation

MolNex is a general molecular machine learning framework organized as a stack of packages with distinct responsibilities:

- `molix`: training infrastructure
- `molrep`: representation learning
- `molpot`: potentials and composition
- `molzoo`: assembled reference model families

This documentation is organized to help you understand those responsibilities first, then how the layers compose, and only after that the package-level APIs.

## Reading Path

If you are new to the project, read in this order:

1. [Architecture](architecture.md) for the package boundaries and stack layout.
2. [Core concepts](concepts.md) for the shared vocabulary used across the docs.
3. One of the package guides under [Package Guides](#package-guides), depending on which layer you are working in.
4. The implementation docs under `docs/molix`, `docs/molrep`, or `docs/molpot` when you need package-level APIs.

If you want to run the project first and read later:

1. [Installation](get-started/installation.md)
2. [Quick start](get-started/quick-start.md)

## Docs Map

### Architecture

- [Architecture](architecture.md): the four-layer stack, ownership boundaries, and how the packages depend on one another
- [Core concepts](concepts.md): the shared concepts used across training, representation, composition, and model assembly

### Package Guides

- [molix](packages/molix.md): what the training and execution layer owns, and what it must not absorb
- [molrep](packages/molrep.md): what belongs in the representation layer and how it feeds higher layers
- [molpot](packages/molpot.md): what belongs in composition and potential construction, and where its boundaries stop
- [molzoo](packages/molzoo.md): what a reference model family is in MolNex and why it sits above the reusable layers

### Getting Started

- [Installation](get-started/installation.md): environment setup
- [Quick start](get-started/quick-start.md): minimal end-to-end path through the stack

### Package-Level Implementation Docs

- [molix docs](molix/index.md): trainer, hooks, and data pipeline details
- [molrep docs](molrep/index.md): encoders and representation modules
- [molpot docs](molpot/index.md): composition components, gradients, and related modeling utilities

### Reference and Specs

- [Component taxonomy](component_taxonomy.md): current package/component inventory
- [TensorDict schema](tensordict_schema.md): current batch structure reference
- [Core interface stabilization spec](specs/core-interface-stabilization.md): current interface cleanup work

## Reading Order

Start from responsibilities and boundaries.

Before reading APIs, decide which layer a concept belongs to:

- execution and lifecycle belong to `molix`
- learned molecular features belong to `molrep`
- structured composition and potential assembly belong to `molpot`
- curated reference architectures belong to `molzoo`

After that, move to composition:

- how training invokes models
- how representations feed downstream modeling layers
- how assembled model families are built from lower-level modules

Then move to APIs:

- package guides for ownership
- package implementation docs for concrete modules
- quick starts and reference pages for operational details
