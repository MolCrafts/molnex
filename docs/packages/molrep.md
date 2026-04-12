# molrep

`molrep` is the representation learning layer.

## Ownership

`molrep` owns:

- embeddings
- interaction blocks
- reusable readout primitives
- modules that convert molecular inputs into learned features

## Non-Goals

`molrep` should not own:

- training orchestration
- package-level model families
- downstream composition responsibilities that belong to `molpot`

## Stack Position

`molrep` provides reusable modeling parts. Those parts can be trained by `molix`, assembled into higher-level modeling layers by `molpot`, and curated into reference families by `molzoo`.

## Next Pages

- [molrep implementation docs](../molrep/index.md)
- [Architecture](../architecture.md)
