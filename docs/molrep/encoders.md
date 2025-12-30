# MolRep Encoders

Encoders are the backbone of any representation learning model. They effectively "read" a molecule—translating it from raw atoms and bonds into dense vector representations. These vectors can then be used for property prediction, pre-training, or similarity search.

`molrep` offers two distinct families of encoders depending on your data: **Geometry** and **Topology**.

## Geometry Encoders

If you are building an interatomic potential or predicting quantum properties (like Formation Energy), the 3D shape of the molecule is critical. A `GeometryEncoder` uses the precise $(x, y, z)$ coordinates of every atom.

It works by constructing a radius graph—connecting atoms that are close to each other in 3D space—and passing messages along these edges. The message functions are typically rotationally invariant, meaning the specific orientation of the molecule doesn't change the result.

```python
from molrep import MolRepModel

# Create a geometry-aware model
model = MolRepModel.from_config(
    encoder_type='geometry',
    hidden_dim=128,
    num_heads=8,
    cutoff=5.0  # Interaction radius in Angstroms
)

features = model.extract_features(batch)
```

## Topology Encoders

For many tasks, especially in drug discovery, precise 3D coordinates are unknown or unreliable. Instead, you might only have a SMILES string or a 2D connectivity graph.

The `TopologyEncoder` ignores spatial positions entirely. It treats the molecule as a graph where atoms are nodes and chemical bonds are edges. This makes it perfect for large-scale pre-training on databases like PubChem or ChEMBL.

```python
# Create a topology-based (graph) model
model = MolRepModel.from_config(
    encoder_type='topology',
    hidden_dim=128,
    num_layers=4,
    use_bond_types=True
)
```

## Atom Embedding

Before any graph or geometry processing begins, the raw atomic numbers must be converted into vectors. The `AtomEmbedding` layer handles this automatically. It maps integer element types (H=1, C=6) to learnable vectors, often initializing them with basic chemical knowledge.

This embedding layer is shared across different encoder types, allowing you to easily transfer chemical knowledge between geometric and topological models.
