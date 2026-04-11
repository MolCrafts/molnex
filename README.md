# MolNex

**Dict-first molecular ML framework for unified modeling of molecular potentials and properties with physics-aware ML.**

MolNex is a modular framework composed of four packages that cover the full pipeline from molecular representation to potential evaluation and training.

## Packages

| Package | Role | Description |
|---------|------|-------------|
| **molix** | Training infrastructure | Trainer, TrainState, Step protocol, Hook lifecycle, data utilities |
| **molrep** | Representation learning | Embedding, Interaction, Readout pipeline with equivariant operations |
| **molpot** | Potential functions | Classical potentials, autograd forces, PotentialComposer |
| **molzoo** | Pre-built encoders | MACE, Allegro (encoder-only, no built-in readout) |

## Installation

Requires Python >= 3.10 and PyTorch >= 2.6.

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from molzoo import MACE, MACESpec
from molpot import LayerPooling, PotentialComposer, LJParameterHead, LJ126

# 1. Build an encoder
spec = MACESpec(num_elements=119, num_features=64, r_max=5.0)
encoder = MACE(spec)

# 2. Compose a potential from encoder features
pool = LayerPooling("mean")
composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64),
    terms={"lj": LJ126()},
)

# 3. Train with molix
from molix import Trainer

trainer = Trainer(
    model=encoder,
    loss_fn=my_loss,
    optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
)
final_state = trainer.train(datamodule, max_epochs=100)
```

## Data Format

MolNex uses nested `TensorDict` subclasses (`molix.data.types`) with per-level batch sizes:

```
GraphBatch (batch_size=[])
├── "atoms": AtomData (batch_size=[N])
│   ├── Z: atomic numbers (N,)
│   ├── pos: positions (N, 3)
│   └── batch: graph membership (N,)
├── "edges": EdgeData (batch_size=[E])
│   ├── edge_index: source-target pairs (E, 2)
│   ├── bond_diff: edge vectors (E, 3)
│   └── bond_dist: edge distances (E,)
└── "graphs": GraphData (batch_size=[B])
    ├── num_atoms: (B,)
    └── <targets>: energy, forces, etc.
```

Samples are plain dicts (`{"Z": tensor, "pos": tensor, "targets": {...}}`).
The transition to nested TensorDict happens at `collate_molecules`.

Access: `batch["atoms", "Z"]`, `batch["edges", "bond_dist"]`.

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
