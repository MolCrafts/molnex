# MolRep

MolRep is the representation learning package in MolNex. Use it for molecular
embeddings, radial and angular bases, equivariant interaction blocks,
and readout primitives.

## Tutorials

- [Build an Encoder](tutorials/build-an-encoder.md): start from atom embeddings
  and produce node features.

## User Guide

- [Encoders](user-guide/encoders.md): compose embeddings, interaction blocks,
  and readouts.
- [Embeddings](user-guide/embeddings.md): turn atomic numbers and scalar inputs
  into learned features.
- [Equivariant operators](user-guide/equivariance.md): the channelwise
  product, symmetric contraction, spherical harmonics, Allegro contractions
  and linear map on molnex's own kernels, with the CPU reference.

## Explanation

- [Representation Learning](explanation/representation-learning.md): how
  MolRep fits between raw graph data and downstream potentials or model heads.

## API

See [molrep API Reference](../api/molrep.md).
