# MolRep Quickstart

**Goal**: Transform a raw molecule (positions and atoms) into a dense feature vector.

## 1. Create a Model

MolRep provides a `MolRepModel` factory that builds the entire encoder pipeline for you. Let's create a **Geometric** encoder that listens to 3D positions.

```python
from molrep import MolRepModel

# Create a model that outputs 128-dimensional vectors
model = MolRepModel.from_config(
    encoder_type='geometry',
    hidden_dim=128,
    num_heads=8,
    cutoff=5.0
)
```

## 2. Prepare Input

We wrap our atomic data in an `AtomicTD`.

```python
from molix.data.atomic_td import AtomicTD
import torch

# Define a water molecule
batch = AtomicTD.create(
    z=torch.tensor([8, 1, 1]),               # O, H, H
    x=torch.tensor([[0., 0., 0.],            # Oxygen at origin
                    [0.75, 0.58, 0.],        # Hydrogen 1
                    [-0.75, 0.58, 0.]]),     # Hydrogen 2
    batch=torch.tensor([0, 0, 0])            # All in batch 0
)
```

## 3. Run Inference

Pass the batch through the model. It automatically handles neighbor list construction and message passing.

```python
# Forward pass
features = model.extract_features(batch)

print(features.shape) 
# torch.Size([3, 128]) -> One vector per atom
```

## Next Steps

*   [**Encoders Deep Dive**](encoders.md): Learn the difference between Geometric and Topological encoders.
*   [**Embedding Layers**](modules/embedding.md): How raw atomic numbers become vectors.
