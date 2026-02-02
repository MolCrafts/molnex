# MolPot: Componentized ML Potential Toolkit

MolPot provides reusable building blocks for molecular machine learning potentials, designed to work seamlessly with the MolNex training system.

## Design Philosophy

**Component-First Architecture**: All reusable components (heads, pooling, etc.) are implemented as TensorDictModuleBase for consistency.

**Clear Boundaries**:
- **Multi-head outputs** → molpot.heads (TensorDictModule-based prediction heads)
- **Generic losses** → molix.core.losses (no domain-specific naming)
- **Configurations** → molix.config (Pydantic specs)
- **Training strategy** → molix.core.trainer (AMP, grad clip, scheduler, logging, etc.)

**Pure PyTorch**: No PyTorch Geometric dependency. All graph operations implemented in pure PyTorch.

## Installation

```bash
cd molnex/src/molpot
pip install -e .
```

## Quick Start

```python
import torch
from molpot import EnergyHead, ForceHead, LJ126
from molix.core.losses import MSELoss, WeightedLoss
from molix.core.trainer import Trainer

# 1. Create model (pure composition of components)
model = MyModel(
    hidden_dim=128,
    num_layers=3,
    cutoff=5.0,
)

# 2. Create generic losses (no domain-specific naming)
loss_fn = WeightedLoss([
    (1.0, MSELoss(pred_key="energy", target_key="energy")),
    (10.0, MSELoss(pred_key="forces", target_key="forces")),
])

# 3. Train with direct interface
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
)

trainer.train(datamodule, max_epochs=100)
```

## Components

### Classic Potentials
- **BasePotential**: Base class for all potentials (PyTorch nn.Module)
- **LJ126**: Lennard-Jones 12-6 potential
- **BondHarmonic**: Harmonic bond stretching
- **AngleHarmonic**: Harmonic angle bending
- **DihedralHarmonic**: Harmonic dihedral torsion

### Prediction Heads
- **EnergyHead**: Molecular energy from atomic energies
- **ForceHead**: Forces via autograd (F = -dE/dpos)
- **TypeHead**: Atom type classification

### Readout Operations
- **SumPooling**: Aggregate atomic features via summation
- **MeanPooling**: Aggregate atomic features via averaging
- **MaxPooling**: Aggregate atomic features via max operation

## Architecture

```
molnex/src/molpot/
├── potentials/     # Classic force field potentials
├── heads/          # ML prediction heads (TensorDictModule)
└── readout/        # Pooling operations (TensorDictModuleBase)
```

Generic losses and configs are in `molix`:
```
molnex/src/molix/
├── core/
│   ├── losses/     # Generic loss functions (MSELoss, MAELoss, WeightedLoss)
│   └── trainer.py  # Training engine
└── config/         # Pydantic configuration specs
    ├── heads.py    # Head configurations
    └── losses.py   # Loss configurations
```

## Example: Building a Custom Model

```python
from molpot import EnergyHead, SumPooling
from molrep import TransformerEncoder  # Representations from molrep
from molix.core.losses import MSELoss, WeightedLoss

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use molrep for representations
        self.encoder = TransformerEncoder(...)
        
        # Use molpot for prediction
        self.energy_head = EnergyHead(hidden_dim=128)
    
    def forward(self, batch):
        # Encode molecular structure
        node_feats = self.encoder(batch)
        
        # Predict energy
        batch["node_feats"] = node_feats
        return self.energy_head(batch)
```

## Testing

```bash
# Run tests
pytest molnex/src/molpot/tests/ -v

# Run example
python molnex/examples/train_example.py
```

## Adding New Potentials

To add a new potential (e.g., SchNet, PaiNN):

1. **Reuse existing components** from molrep (encoder), molpot (heads, pooling)
2. **Implement model-specific logic** (message passing, updates)
3. **Compose into model class**
4. **Use generic molix losses** (configurable keys, no domain assumptions)

Example:
```python
from molpot import EnergyHead
from molix import MSELoss, WeightedLoss, Trainer

class SchNet(nn.Module):
    def __init__(self):
        # Reuse components
        self.energy_head = EnergyHead(...)
        
        # SchNet-specific
        self.interaction_blocks = ...
    
    def forward(self, batch):
        # SchNet logic using components
        ...

# Training with generic losses
loss_fn = WeightedLoss([
    (1.0, MSELoss(pred_key="energy", target_key="energy")),
    (10.0, MSELoss(pred_key="forces", target_key="forces")),
])
trainer = Trainer(model=SchNet(), loss_fn=loss_fn)
```

## Dependencies

- PyTorch >= 2.0
- TensorDict >= 0.2

## License

See LICENSE file.
