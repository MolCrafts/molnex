# Encoders

`molrep` provides composable building blocks for molecular encoders. Reference
assemblies such as MACE and Allegro are exposed from `molzoo`.

## Typical Stages

A molecular encoder usually combines:

1. Node embeddings from atomic attributes.
2. Radial features from edge distances.
3. Angular features from edge directions.
4. Interaction blocks that update node features.
5. Readout or pooling layers for downstream tasks.

## Node Embeddings

```python
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64),
    ],
    out_dim=128,
)
```

## Radial and Cutoff Features

```python
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.radial import BesselRBF

rbf = BesselRBF(r_cut=5.0, num_radial=8)
cutoff = CosineCutoff(r_cut=5.0)
```

## Interaction Blocks

Interaction modules process geometric message passing and equivariant updates:

- `ConvTP`
- `EquivariantLinear`
- `SymmetricContraction`
- `MessageAggregation`

## Readout

Readout modules convert learned features into scalar or structured outputs:

- `ProductHead`
- pooling functions from `molrep.readout.pooling`
- downstream heads in `molpot`

Use `molrep` when you need reusable representation blocks. Use `molzoo` when
you want a curated encoder assembly.
