# MolRep Encoders

`molrep` provides composable building blocks for molecular encoders. Pre-built encoder assemblies (MACE, Allegro) are available in `molzoo`.

## Embedding Layer

The first step is converting raw atomic numbers into learnable vectors:

```python
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding

embed = JointEmbedding(
    embedding_specs=[
        DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64),
    ],
    out_dim=128,
)
```

## Radial and Angular Features

Distance and angle information is expanded into basis functions:

```python
from molrep.embedding.radial import BesselRBF
from molrep.embedding.cutoff import CosineCutoff

rbf = BesselRBF(num_rbf=8, cutoff=5.0)
cutoff = CosineCutoff(cutoff=5.0)
```

## Interaction Blocks

Interaction modules process message-passing updates with equivariant operations:

- `ConvTP`: Tensor product convolution
- `EquivariantLinear`: SO(3)-equivariant linear layer
- `SymmetricContraction`: Multi-body basis construction
- `MessageAggregation`: Scatter-sum aggregation

## Readout

- `ProductHead`: Multi-body basis to scalar features
- `ScalarHead`: Pooling + MLP for scalar prediction
- `masked_sum_pooling` / `masked_mean_pooling`: Graph-level aggregation
