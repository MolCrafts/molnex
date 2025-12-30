# MolPot: Componentized ML Potential Toolkit

MolPot provides reusable building blocks for molecular machine learning potentials, designed to work seamlessly with the MolNex training system.

## Design Philosophy

**Component-First Architecture**: All reusable components (RBF, MLP, pooling, etc.) live in common modules. Model-specific code is pure composition.

**Clear Boundaries**:
- **Multi-head outputs** → model (via componentized heads)
- **Multi-loss composition** → loss_fn (composable loss modules)
- **Training strategy** → Trainer (AMP, grad clip, scheduler, logging, etc.)

**Pure PyTorch**: No PyTorch Geometric dependency. All graph operations implemented in pure PyTorch.

## Installation

```bash
cd molix/src/molpot
pip install -e .
```

## Quick Start

```python
import torch
from molpot import PiNet2, EnergyLoss, AtomicTD
from molix.core.trainer import Trainer

# 1. Create model (pure composition of components)
model = PiNet2(
    hidden_dim=128,
    num_layers=3,
    cutoff=5.0,
)

# 2. Create loss
loss_fn = EnergyLoss()

# 3. Train with direct interface
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
)

trainer.train(datamodule, max_epochs=100)
```

## Components

### Data Structures
- **AtomicTD**: TensorDict-based atomistic batch representation
- **collate_atomic**: Collation function for batching molecules

### Graph Construction
- **radius_graph**: Pure PyTorch neighbor list construction

### Feature Engineering
- **GaussianRBF**: Radial basis functions for distance featurization
- **CosineCutoff**: Smooth cutoff function
- **PolynomialCutoff**: Alternative cutoff with continuous derivatives
- **Geometry utilities**: Distance, angle, dihedral calculations

### Neural Network Primitives
- **MLP**: Configurable multi-layer perceptron

### Readout Operations
- **SumPooling**: Aggregate atomic features via summation
- **MeanPooling**: Aggregate atomic features via averaging
- **MaxPooling**: Aggregate atomic features via max operation

### Prediction Heads
- **EnergyHead**: Molecular energy from atomic energies
- **ForceHead**: Forces via autograd (F = -dE/dpos)

### Loss Functions
- **EnergyLoss**: MSE loss for energy prediction
- **ForceLoss**: MSE loss for force prediction
- **CombinedLoss**: Weighted combination of energy and force losses

### Models
- **PiNet2**: Simplified PiNet implementation (component-first)

## Architecture

```
molnex/src/molpot/
├── data/           # AtomicTD, collation
├── graph/          # Radius graph (pure PyTorch)
├── feats/          # RBF, cutoff, geometry
├── nn/             # MLP, normalization
├── readout/        # Pooling operations
├── heads/          # Energy, force heads
├── losses/         # Energy, force, combined losses
└── models/         # PiNet2, etc.
```

## Example: Building a Custom Model

```python
from molpot import (
    GaussianRBF, CosineCutoff, radius_graph,
    MLP, SumPooling, EnergyHead
)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Reuse components
        self.rbf = GaussianRBF(num_rbf=50, cutoff=5.0)
        self.cutoff = CosineCutoff(cutoff=5.0)
        self.mlp = MLP(in_dim=50, out_dim=128, hidden_dims=[128])
        self.energy_head = EnergyHead()
    
    def forward(self, batch):
        # Build graph
        edge_index, edge_vec = radius_graph(
            batch["pos"], batch["batch"], cutoff=5.0
        )
        
        # Compute features
        distances = torch.norm(edge_vec, dim=-1)
        rbf_feats = self.rbf(distances)
        cutoff_vals = self.cutoff(distances)
        
        # ... your model logic ...
        
        return self.energy_head(atomic_energies, batch["batch"])
```

## Testing

```bash
# Run tests
pytest molix/src/molpot/tests/ -v

# Run example
python molix/examples/train_pinet2_simple.py
```

## Adding New Potentials

To add a new potential (e.g., SchNet, PaiNN):

1. **Reuse existing components** (RBF, MLP, pooling, heads, losses)
2. **Implement model-specific logic** (message passing, updates)
3. **Compose into model class**
4. **Use same Trainer** (no training code needed)

Example:
```python
class SchNet(nn.Module):
    def __init__(self):
        # Reuse components
        self.rbf = GaussianRBF(...)
        self.mlp = MLP(...)
        self.energy_head = EnergyHead()
        
        # SchNet-specific
        self.interaction_blocks = ...
    
    def forward(self, batch):
        # SchNet logic using components
        ...
```

## Dependencies

- PyTorch >= 2.0
- TensorDict >= 0.2

## License

See LICENSE file.
