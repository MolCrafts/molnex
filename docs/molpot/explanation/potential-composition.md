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
  -> per-graph energy          (energy)
  -> forces / virial           (forward = Derivative(energy, wrt))
```

## Why This Exists

Many molecular models share an encoder but differ in the way they produce
physical quantities. `molpot` gives that downstream structure a dedicated home:

- parameter heads learn physical or semi-physical parameters
- potential terms evaluate known functional forms
- composition combines multiple terms
- every composed model is a `molix.autograd.Potential`: `forward` is the
  `molix.autograd.Derivative` of its `energy`, writing forces and the virial;
  `molpot.derivation` adds the per-graph aggregation and the Cartesian Hessian

## Boundary

`molpot` consumes features. It does not own the training loop (`molix`) or the
low-level representation blocks (`molrep`). Reference encoders that assemble
many lower-level modules live in `molzoo`.
