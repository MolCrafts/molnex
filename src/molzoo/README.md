# MolZoo: Molecular Architecture Zoo

Complete neural network architectures for molecular ML potentials.

## Overview

MolZoo combines components from **molrep** (representations) and **molpot** (interactions) into end-to-end models for predicting molecular energies and forces.

### Architecture Layers

```
Input: (atomic numbers, positions, batch indices)
  ↓
[Embedding Layer]
  - Discrete: atomic number → learned embeddings
  - Continuous: distances, angles → RBF/SH features
  ↓
[Equivariant Interaction Blocks]
  - Message passing with tensor products
  - Learnable element-specific updates
  - Feature aggregation
  ↓
[Output Heads]
  - Energy prediction via scatter pooling
  - Force prediction via autograd -dE/dpos
```

## Models

### MACE (Message-Passing Equivariant Convolution)

Feature extractor that builds equivariant representations through message-passing layers.

```python
from molzoo import MACE

mace = MACE(
    num_species=100,
    hidden_dim=128,
    num_layers=3,
    l_max=2,
    r_cut=5.0,
    num_radial=20,
)

# Returns features for each atom
features = mace(atom_td)  # [num_atoms, hidden_dim]
```

**When to use**: When you need intermediate representations for downstream tasks (classification, property prediction, etc.)

### ScaleShiftMACE

Complete model combining MACE with energy/force prediction heads.

```python
from molzoo import ScaleShiftMACE

model = ScaleShiftMACE(
    num_species=100,
    hidden_dim=128,
    num_layers=3,
    l_max=2,
    r_cut=5.0,
    num_radial=20,
    pooling="sum",  # or "mean"
    out_dim=1,      # output dimension (e.g., 1 for energy)
)

# Predict energy
output = model(**batch.to_model_kwargs())

# output = {
#     'energy': Tensor[batch_size],
#     'forces': Tensor[num_atoms, 3],  # if pos requires_grad=True
# }
```

**When to use**: For direct energy and force prediction on molecular systems.

## Configuration

Both models use Pydantic specs for configuration:

```python
from molzoo import ScaleShiftMACESpec

spec = ScaleShiftMACESpec(
    num_species=100,
    hidden_dim=128,
    num_layers=3,
    l_max=2,
    r_cut=5.0,
    num_radial=20,
    pooling="sum",
    out_dim=1,
)

model = spec.build()
```

## Integration with Training

Use with molix trainer:

```python
from molix import Trainer, TrainState, Stage
from molix.core.losses import MSELoss
from molzoo import ScaleShiftMACE

# Create model and loss
model = ScaleShiftMACE(num_species=100, hidden_dim=128, ...)
loss_fn = MSELoss()

# Create trainer
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=torch.optim.Adam(model.parameters()),
)

# Train
for epoch in range(num_epochs):
    state = trainer.train_step(batch)
    print(f"Loss: {state.loss}")
```

## Extending with Custom Heads

To use MACE features with custom output heads:

```python
from molzoo import MACE
from molpot.heads import CustomHead

# Get intermediate features
mace = MACE(num_species=100, hidden_dim=128, ...)

# Add custom head
custom_head = CustomHead(input_dim=128, output_dim=...)

# Combine
class CustomMACE(nn.Module):
    def __init__(self, mace, head):
        super().__init__()
        self.mace = mace
        self.head = head

    def forward(self, td):
        features = self.mace(td)
        return self.head(features, td)
```

## References

- **molrep**: Embedding and representation components
- **molpot**: Interaction blocks and prediction heads
- **molix**: Training infrastructure and data handling
