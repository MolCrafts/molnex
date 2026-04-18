# MolNex

MolNex is a general molecular machine learning framework organized as a stack of packages rather than a single monolithic library.

It is built for projects where training infrastructure, representation learning, compositional modeling, and reference model families need to coexist in one codebase without collapsing into one layer. Instead of centering the framework around one model class or one end-to-end API, MolNex splits the system into four packages with distinct ownership:

- `molix`: training and execution
- `molrep`: representation learning
- `molpot`: potentials and composition
- `molzoo`: assembled reference model families

That structure matters because these parts change for different reasons. Training systems evolve around execution and lifecycle concerns. Representation modules evolve around feature construction and interaction design. Compositional modeling evolves around output structure and domain-specific assembly. Reference models evolve as curated combinations of lower-level parts. Treating them as one layer makes the project harder to extend, test, and reason about.

## Framework Scope

MolNex is a framework for building molecular ML systems as a set of cooperating layers.

It is not a single model library, and it is not a trainer wrapped around one preferred architecture. The project is meant to support:

- reusable training infrastructure
- reusable representation modules
- reusable composition and potential layers
- assembled model families built from those lower-level components

The intended result is a codebase where a user can understand which layer they are working in, what that layer is allowed to own, and how it connects to the rest of the stack.

## Layered Infrastructure

The simplest way to read MolNex is as a layered framework:

1. `molix` runs training and evaluation.
2. `molrep` defines how molecular structure becomes learned representations.
3. `molpot` defines how representations are turned into compositional outputs or potential-based models.
4. `molzoo` assembles reference model families from the lower-level pieces.

These concerns are separated because they should not move at the same rate and they should not carry the same responsibilities.

If training logic is embedded in model definitions, experimentation becomes harder to control. If representation modules are tied to one assembled architecture, reuse drops quickly. If compositional modeling is mixed into the training layer, domain logic leaks into infrastructure. If reference models become the framework surface, the lower-level stack becomes harder to evolve.

MolNex is structured to avoid that collapse.

## Package Boundaries

### `molix`

`molix` owns execution, training lifecycle, state, orchestration, and data flow.

It should own:

- training and evaluation loops
- train state and stage tracking
- step protocols
- hooks, callbacks, checkpointing, and orchestration utilities
- the framework-side path by which batches move through execution

It should not own:

- representation learning logic
- model-specific architecture decisions
- potential construction or compositional output logic
- curated model families

It relates to the rest of the stack by running them. `molix` is the execution layer that coordinates models and losses without becoming the definition of those models.

### `molrep`

`molrep` owns representation learning modules.

It should own:

- embeddings
- interaction blocks
- pooling and readout primitives
- reusable modules for turning molecular inputs into learned features

It should not own:

- training lifecycle
- checkpointing and orchestration
- package-level reference architectures
- downstream composition logic that belongs to potential assembly

It relates to the rest of the stack by providing reusable modeling parts. `molrep` is where feature construction and interaction design live, independent of how a full system is trained or assembled.

### `molpot`

`molpot` owns potential construction and compositional modeling layers.

It should own:

- potential terms
- output parameterization and composition
- modeling layers that assemble structured downstream predictions from learned representations

It should not own:

- generic training infrastructure
- the full representation learning stack
- the role of a catch-all domain package
- assembled model families that should live one layer higher

It relates to the rest of the stack by consuming learned representations and exposing structured modeling components. `molpot` is a modeling layer, not the identity of the whole framework.

### `molzoo`

`molzoo` owns assembled reference model families built from lower-level components.

It should own:

- curated model families
- reference assemblies of encoder or end-to-end model stacks
- package-level entry points for known architectures

It should not own:

- generic training infrastructure
- low-level representation primitives
- the reusable composition layer
- framework-wide architectural policy

It relates to the rest of the stack by packaging lower-level parts into concrete reference models. `molzoo` should demonstrate how the system is assembled, not replace the system beneath it.

## Separation Rationale

MolNex is split by architectural responsibility rather than by file type or by research topic.

This has direct consequences:

- training infrastructure can evolve without rewriting model code
- representation modules can be reused across multiple assembled models
- compositional modeling stays separate from orchestration concerns
- reference models remain replaceable examples of assembly rather than framework-defining abstractions
- contributors can make changes in one layer without silently changing ownership in another

The split is meant to keep the framework predictable as it grows.

## Design Rules

These are the architectural rules that shape the project:

- package boundaries are part of the public design, not just an internal folder layout
- execution, representation, composition, and assembly are different responsibilities and should remain different responsibilities
- reusable lower-level modules should not depend on one curated top-level model family
- reference models should be built from the stack, not treated as the stack
- the docs should explain ownership and composition before listing APIs

## Infrastructure Priorities

MolNex is optimized for:

- projects that need more than one model family in one codebase
- reuse of training infrastructure across different model assemblies
- reuse of representation modules across different downstream modeling choices
- composition of structured modeling layers without pushing them into the trainer
- contributor workflows where package ownership and review boundaries are explicit
- documentation that helps readers understand where new code belongs

## Docs Map

The documentation is organized around responsibilities first, then composition, then APIs.

Start here:

- [Documentation home](docs/index.md): navigation map for the docs set
- [Architecture](docs/architecture.md): how the four-layer stack fits together
- [Core concepts](docs/concepts.md): the vocabulary used across the framework
- [Package guide: molix](docs/packages/molix.md)
- [Package guide: molrep](docs/packages/molrep.md)
- [Package guide: molpot](docs/packages/molpot.md)
- [Package guide: molzoo](docs/packages/molzoo.md)

If you want to get running quickly, use:

- [Installation](docs/get-started/installation.md)
- [Quick start](docs/get-started/quick-start.md)

If you want package-level implementation docs, continue into the existing package sections under [`docs/molix`](docs/molix/index.md), [`docs/molrep`](docs/molrep/index.md), and [`docs/molpot`](docs/molpot/index.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
