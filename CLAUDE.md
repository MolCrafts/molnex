# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MolNex** (v2.0.0) is a dict-first molecular ML framework for unified modeling of molecular potentials and properties with physics-aware ML. It is composed of four packages:

| Package | Role | Key Patterns |
|---------|------|-------------|
| **molix** | Training infrastructure | Trainer, TrainState (dict), Step protocol, Hook lifecycle |
| **molrep** | Representation learning | Embedding → Interaction → Readout pipeline, equivariance via cuEquivariance |
| **molpot** | Potential functions | BasePotential (nn.Module + ABC), autograd forces, PotentialComposer |
| **molzoo** | Pre-built encoders | Encoder-only (MACE, Allegro), no readout — downstream uses molpot |

## Build & Development

```bash
# Install (editable, with C++ extensions via scikit-build-core + CMake >=4.0)
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_molzoo/test_mace.py -v

# Run single test
python -m pytest tests/test_molzoo/test_mace.py::test_mace_forward -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Python >=3.10 required. Requires `torch>=2.6` (always use latest stable PyTorch).

## Architecture

### Nested TensorDict Data Flow

All molecular batch data uses nested `TensorDict` subclasses (`molix/data/types.py`) with per-level batch sizes:

```
GraphBatch (batch_size=[])
├── "atoms": AtomData (batch_size=[N])
│   ├── Z: atomic numbers (N,)
│   ├── pos: positions (N, 3)
│   └── batch: graph membership (N,)
├── "edges": EdgeData (batch_size=[E])
│   ├── edge_index: source-target pairs (E, 2)   # [:,0]=source, [:,1]=target
│   ├── bond_diff: edge vectors (E, 3)            # pos[target] - pos[source]
│   └── bond_dist: edge distances (E,)
└── "graphs": GraphData (batch_size=[B])  [optional]
    ├── num_atoms: (B,)
    └── <targets>
```

Access: `batch["atoms", "Z"]`, `batch["edges", "bond_dist"]`. Encoder outputs extend via inheritance: `NodeRepAtoms` adds `node_features`, `EdgeRepEdges` adds `edge_features`.

### Edge Convention (MUST follow everywhere)

```
edge_index[:, 0]  — source atom  (the "centre" in Allegro; the "sender" in MACE ConvTP)
edge_index[:, 1]  — target atom  (the "neighbour" in Allegro; the "receiver" in MACE ConvTP)
bond_diff         — pos[target] - pos[source]   (displacement vector, source → target)
bond_dist         — ‖bond_diff‖
```

`NeighborList` defaults to **full bidirectional** edges (`symmetry=True`, `E = 2 × n_pairs`).
Pass `symmetry=False` to get only the upper-triangle half-pairs (`E = n_pairs`) when you explicitly
want to exploit Newton's-3rd-law symmetry.  The two modes produce different `task_id`s so pipeline
caches are kept separate.

**Why bond_diff = pos[target] − pos[source]?**  This makes the displacement vector point in the
same direction as the edge (source → target), which is the convention expected by `SphericalHarmonics`
and all `cuEquivariance`-based tensor products in this repo.  The C++ `getNeighborPairs` kernel
returns `pos[rows] − pos[cols]` (opposite sign); `NeighborList.execute` negates it.

### Module Dependency Graph

```
molix.config (global dtype singleton)
    ↓
molrep.embedding → molrep.interaction → molrep.readout
    ↓                                       ↓
molzoo (MACE, Allegro encoders)         molpot.heads
    ↓                                       ↓
molpot.composition (PotentialComposer)  molpot.potentials
    ↓
molix.core (Trainer, TrainState, Step, Hook)
    ↓
molix.data (Dataset, collate, preprocess)
molix.datasets (QM9, MD17)
```

### Key Design Patterns

- **Encoder-only molzoo**: Encoders return raw features `(N, layers, features)`, readout/potentials handled by molpot
- **Pydantic configs**: All block configs use `BaseModel` with `ConfigDict(arbitrary_types_allowed=True)`
- **cuEquivariance**: Tensor products use `cuequivariance` / `cuequivariance_torch` for GPU-accelerated equivariant operations
- **Autograd forces**: `BasePotential.calc_forces()` computes `F = -dE/dx` via `torch.autograd.grad`
- **Functional composition**: `PotentialComposer` chains pooling → parameter heads → potential terms → aggregation
- **Hook protocol**: Lifecycle callbacks (`on_train_start`, `on_epoch_end`, etc.) via `Hook` protocol
- **Step protocol**: `DefaultTrainStep` / `DefaultEvalStep` wrap forward → loss → backward → optimizer

### Adding New Components

**New encoder** (molzoo): Accept `(Z, bond_dist, bond_diff, edge_index)`, return `(N, layers, features)`. Use `molrep` building blocks. Add paper reference.

**New potential** (molpot): Inherit `BasePotential`, implement `forward() -> scalar energy Tensor`. Forces come from autograd automatically.

**New embedding/interaction** (molrep): Pure `nn.Module`, use `cuequivariance` for equivariant layers.

## Scientific Correctness Requirements

Every implementation of a physical model, potential, or operator MUST:
1. Reference the original publication (arXiv/DOI) in the module docstring
2. Match the equations in the paper — document any deviations with rationale
3. Include numerical validation tests against reference implementations or published values
4. Preserve physical symmetries (energy conservation, rotational/translational invariance, permutation equivariance)

## PyTorch Version Policy

This project tracks **latest stable PyTorch** (currently >=2.6). Use modern PyTorch APIs:
- `torch.compile` for performance-critical paths
- `torch.export` for model serialization
- `torch.nn.functional` over deprecated module alternatives
- Native `torch.nested` for variable-length sequences where appropriate

## Docstring Convention

Google-style docstrings with tensor shape annotations:

```python
def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Message passing step.

    Args:
        node_feats: Node features ``(n_nodes, hidden_dim)``.
        edge_index: Edge indices ``(n_edges, 2)``.

    Returns:
        Updated node features ``(n_nodes, hidden_dim)``.

    Reference:
        Author et al. "Paper Title" Venue Year
        https://arxiv.org/abs/XXXX.XXXXX
    """
```

## MolNex Development Skills

Use these slash commands for structured development:

| Skill | Purpose |
|-------|---------|
| `/molnex-impl <feature_or_spec>` | Full implementation workflow: spec → design → TDD → implement → review → docs |
| `/molnex-spec <description>` | Convert natural language to detailed technical spec with literature review |
| `/molnex-arch` | Architecture validation against module dependency rules |
| `/molnex-test` | Test coverage analysis and gap identification |
| `/molnex-perf` | Performance profiling and PyTorch optimization review |
| `/molnex-docs` | Documentation and docstring completeness audit |
| `/molnex-review` | Code review for correctness, immutability, and scientific accuracy |
| `/molnex-litrev <topic>` | Literature review: search arXiv/papers for relevant methods before implementation |

## MolNex Development Agents

| Agent | Purpose |
|-------|---------|
| `molnex-architect` | Architecture design, module boundary validation, component integration |
| `molnex-scientist` | Scientific correctness: literature lookup, equation verification, symmetry checks |
| `molnex-tester` | TDD workflow: write failing tests → implement → verify |
| `molnex-optimizer` | PyTorch performance: torch.compile, memory, GPU utilization |
| `molnex-documenter` | Docstrings, docs/ updates, API documentation |
| `molnex-ml-expert` | Training dynamics, loss design, benchmark methodology, hyperparameter analysis |
| `molnex-pm` | API usability, breaking-change analysis, feature prioritization, downstream integration |
| `molnex-compute-scientist` | Algorithm complexity, numerical stability, reproducibility, HPC/DDP readiness |

## Workflow

For any non-trivial implementation:

1. `/molnex-litrev <topic>` — search literature, verify scientific basis
2. `/molnex-spec <feature>` — generate detailed spec with equations and references
3. `/molnex-impl <spec>` — orchestrates: architect → TDD → implement → review → docs
