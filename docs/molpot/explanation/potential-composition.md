# Potential Composition

MolPot separates representation features from physical output construction.

## Data Flow

```text
node_features
  -> parameter head
  -> per-atom parameters
  -> potential-specific mixing
  -> per-pair parameters
  -> potential terms
  -> per-graph energy
  -> optional force derivation
```

## Why This Exists

Many molecular models share an encoder but differ in the way they produce
physical quantities. `molpot` gives that downstream structure a dedicated home:

- parameter heads learn physical or semi-physical parameters
- potential terms evaluate known functional forms
- composition combines multiple terms
- derivation operators produce forces and stress from energy

## Boundary

`molpot` consumes features. It does not own the training loop (`molix`) or the
low-level representation blocks (`molrep`). Reference encoders that assemble
many lower-level modules live in `molzoo`.
