# Component Taxonomy

## Overview

MolNex v2.0 organizes components into 7 roles, all following the TensorDictModule pattern with explicit `in_keys`/`out_keys`.

## Component Roles

### 1. Data Pipeline (`molix.data`)

**Purpose**: Construct and preprocess AtomicTD

**Components**:
- `TopologyBuilder` - Build neighbor topology
- `GeometryPreprocessor` - Compute bond vectors/distances
- `Normalizer` - Center molecules

**Contract**:
- Input: Raw atomic data or partially constructed AtomicTD
- Output: Fully preprocessed AtomicTD
- No ML logic (pure data processing)

**Example**:
```python
from molix.data import TopologyBuilder, Normalizer

pipeline = torch.nn.Sequential(
    TopologyBuilder(cutoff=5.0),
    Normalizer(center=True),
)
```

### 2. Representation Initializers (`molrep`)

**Purpose**: Initialize learned representations from atomic features

**Components**:
- `AtomEmbedding` - Embed atomic numbers → invariant features
- `SphericalEmbedding` - Embed bond vectors → spherical harmonics

**Contract**:
- Input: AtomicTD (with topology)
- Output: AtomicTD with `atoms.h`, `atoms.h_sph`, etc.
- Declare representation space (invariant/equivariant/spherical)

**Example**:
```python
from molrep import AtomEmbedding

initializer = AtomEmbedding(num_types=100, hidden_dim=64)
# in_keys: [("atoms", "z")]
# out_keys: [("atoms", "h")]
```

### 3. Interaction Blocks (`molrep`)

**Purpose**: Update representations via message passing

**Components**:
- `TopologyInteraction` - Message passing on graph topology
- `GeometryInteraction` - Geometry-aware message passing with RBF

**Contract**:
- Input: AtomicTD with representations
- Output: Updated representations (in-place or new)
- Preserve representation space properties

**Example**:
```python
from molrep import TopologyInteraction

interaction = TopologyInteraction(hidden_dim=64)
# in_keys: [("atoms", "h"), ("bonds", "i"), ("bonds", "j")]
# out_keys: [("atoms", "h")]
```

### 4. Prediction Heads (`molpot.heads`)

**Purpose**: Map representations to predictions

**Components**:
- `EnergyHead` - Predict molecular energy
- `ForceHead` - Derive forces from energy (physics-consistent)
- `TypeHead` - Predict atom types

**Contract**:
- Input: AtomicTD with representations
- Output: Predictions in `target.*` namespace
- Key alignment: predictions use same keys as targets

**Example**:
```python
from molpot.heads import EnergyHead

head = EnergyHead(hidden_dim=64)
# in_keys: [("atoms", "h"), ("graph", "batch")]
# out_keys: [("target", "energy")]
```

### 5. Loss Functions (`molpot.losses`)

**Purpose**: Compute loss between predictions and targets

**Components**:
- `EnergyLoss` - MSE on molecular energy
- `ForceLoss` - MSE on atomic forces
- `CombinedLoss` - Weighted combination

**Contract**:
- Input: Two TensorDicts (predictions, targets)
- Output: TensorDict with `loss`
- Key alignment enables direct comparison

**Example**:
```python
from molpot.losses.tensordict_losses import EnergyLoss

loss_fn = EnergyLoss(weight=1.0)
loss_td = loss_fn(pred_td, true_td)
loss = loss_td["loss"]
```

## Composition Patterns

### Sequential Composition

```python
model = torch.nn.Sequential(
    TopologyBuilder(cutoff=5.0),
    AtomEmbedding(num_types=100, hidden_dim=64),
    TopologyInteraction(hidden_dim=64),
    TopologyInteraction(hidden_dim=64),
    EnergyHead(hidden_dim=64),
)
```

**Requirements**:
- `out_keys` of component N must provide `in_keys` of component N+1
- TensorDict threaded through automatically
- No manual unpacking

### Parallel Heads

```python
# Shared encoder
encoder = torch.nn.Sequential(
    TopologyBuilder(cutoff=5.0),
    AtomEmbedding(num_types=100, hidden_dim=64),
    TopologyInteraction(hidden_dim=64),
)

# Multiple heads
energy_head = EnergyHead(hidden_dim=64)
type_head = TypeHead(hidden_dim=64, num_types=10)

# Forward
td = encoder(atomic_td)
td = energy_head(td)
td = type_head(td)
```

**Benefits**:
- Share encoder weights
- Multiple predictions from same representation
- Clear declaration of all inputs/outputs

### Representation Reuse

```python
# Train encoder
encoder = torch.nn.Sequential(
    TopologyBuilder(cutoff=5.0),
    AtomEmbedding(num_types=100, hidden_dim=64),
    TopologyInteraction(hidden_dim=64),
)

# Freeze and reuse
for param in encoder.parameters():
    param.requires_grad = False

# New head
new_head = TypeHead(hidden_dim=64, num_types=20)
```

## Design Invariants

### Explicit Contracts

Every component declares:
- `in_keys`: What fields it reads
- `out_keys`: What fields it writes

```python
component.in_keys  # [("atoms", "z")]
component.out_keys  # [("atoms", "h")]
```

### Key-Based Wiring

- No positional arguments for data
- No manual unpacking
- TensorDict passed through
- Mutations explicit

### Immutability

- Topology fields immutable after construction
- Can be enforced: `td["bonds"].lock_()`
- Violations raise `RuntimeError`

### Gradient Handling

- `ForceHead` requires `atoms.x.requires_grad = True`
- Documented in component docstring
- Raises error if gradient not available

### Failure Modes

Components fail fast on:
1. Missing `in_key`
2. Duplicate `out_key`
3. Mutate immutable field
4. Wrong dtype
5. Missing gradient

## Package Boundaries

### molix: Data Pipeline
- **Responsibility**: Construct/preprocess AtomicTD
- **No ML logic**: Pure data processing
- **Exports**: AtomicTD, Config, TopologyBuilder, etc.

### molrep: Representation Learning
- **Responsibility**: Learn atomic representations
- **Exports**: Initializers, Interaction blocks
- **Representation spaces**: Invariant, Equivariant, Spherical

### molpot: Prediction & Physics
- **Responsibility**: Predict properties, enforce physics
- **Exports**: Heads, Losses
- **Physics-consistent**: ForceHead uses autograd

## Migration Guide

### From Legacy Components

**Before** (v1.0):
```python
class MyEncoder(nn.Module):
    def forward(self, batch):
        z = batch["atoms", "z"]
        h = self.embed(z)
        return {("atoms", "feat"): h}
```

**After** (v2.0):
```python
class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.in_keys = [("atoms", "z")]
        self.out_keys = [("atoms", "h")]
    
    def forward(self, td):
        z = td["atoms", "z"]
        h = self.embed(z)
        td["atoms", "h"] = h
        return td
```

**Key changes**:
1. Add `in_keys`, `out_keys` attributes
2. Return TensorDict (not dict)
3. Use new field names (`atoms.h` not `atoms.feat`)
