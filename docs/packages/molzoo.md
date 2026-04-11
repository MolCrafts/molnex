# molzoo

`molzoo` is the reference model family layer.

## Ownership

`molzoo` owns:

- curated model families
- concrete assemblies of lower-level modules
- package-level entry points for known architectures

## Non-Goals

`molzoo` should not own:

- generic training infrastructure
- low-level reusable representation primitives
- lower-level composition building blocks
- framework-wide policy for the rest of the stack

## Stack Position

`molzoo` sits above the reusable layers. It demonstrates how `molrep` and `molpot` can be assembled into concrete models while leaving execution to `molix`.

## Next Pages

- [Architecture](../architecture.md)
- [Core concepts](../concepts.md)
