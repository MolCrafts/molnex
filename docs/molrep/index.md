# Deep Representation Learning with MolRep

## MolRep Overview

### What it is
`molrep` is the module responsible for translating raw atomic numbers and positions into rich vector representations. It serves as the "encoding" layer for your molecular machine learning models.

### Why specialized encoders?
Raw atomic coordinates are difficult for standard neural networks to process because they lack invariance to rotation and translation. Specialized encoders (like $E(3)$-equivariant networks or graph transformers) solve this by baking physical symmetries directly into the model architecture.

## Topology vs. Geometry

# Getting Started with MolRep

MolRep turns molecules into math. It provides state-of-the-art encoders to convert atomic systems into meaningful vector representations.

## Start Here

1.  [**Quickstart**](quickstart.md): Embed your first molecule.
2.  [**Encoders**](encoders.md): Choose between Geometry (3D) and Topology (Graph) encoders.
3.  [**Embeddings**](modules/embedding.md): Understanding the first layer of the network.

## Key Concepts

### Geometry First
Unlike traditional graph neural networks, MolRep's default mode is **geometric**. It considers the precise angle, distance, and torsion of bonds, making it suitable for predicting quantum chemical properties.

### Topology Fallback
For datasets without 3D structures (like large SMILES screenings), MolRep provides robust graph encoders that work purely on connectivity.

## Design

`molrep` components are designed to be plug-and-play. They all respect the `TensorDict` protocol, meaning you can drop them into any `molix` pipeline without writing adapter code.

## Components

### Encoders
Encoders transform atoms and bonds into feature vectors. `molrep` provides two main categories:
*   [**Geometry Encoders**](encoders.md#geometry-encoder): Use 3D coordinates. Essential for potentials.
*   [**Topology Encoders**](encoders.md#topology-encoder): Use bond graphs. Great for 2D data or pre-training.

[Read the Guide on Encoders →](encoders.md)

### Building an Encoder
The `MolRepModel` is a configurable factory that creates the appropriate encoder based on your needs.

### How to use it
Here is how to instantiate a 3D geometric encoder.

```python
from molrep import MolRepModel

# Define a geometric encoder configuration
model = MolRepModel.from_config(
    encoder_type='geometry',
    hidden_dim=64,
    num_heads=4,
    num_types=11,  # e.g., H, C, N, O...
)

# Forward pass with a batch
# features shape: [num_atoms, 64]
features = model.extract_features(batch) 
```

## Integration

### What it is
MolRep models output a standard feature vector for each atom.

### Why standardized output?
This standardization allows `molpot` (the potential module) to consume features from *any* `molrep` encoder without caring about the internal details of how those features were generated.
