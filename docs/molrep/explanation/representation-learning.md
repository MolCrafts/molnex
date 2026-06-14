# Representation Learning

MolRep turns molecular structure into learned features that can be reused by
downstream tasks.

The package is intentionally lower level than `molzoo`: it provides reusable
modules rather than a single assembled model family.

## Inputs

MolRep modules commonly consume:

- `Z`: atomic numbers
- `pos`: atom positions
- `edge_index`: source-target edge pairs
- `bond_diff`: edge vectors, `pos[target] - pos[source]`
- `bond_dist`: edge distances

## Outputs

Most representation modules produce per-atom or per-edge features. Those
features can then be consumed by:

- `molpot` heads and potential composers
- `molzoo` reference encoder assemblies
- custom PyTorch readouts

## Boundary

MolRep should contain reusable representation machinery. Training loops belong
in `molix`; potential terms and physical output composition belong in `molpot`;
full reference model assemblies belong in `molzoo`.
