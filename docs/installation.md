# Installation

## Prerequisites

Before installing MolNex, ensure you have:

- Python >= 3.12
- PyTorch >= 2.10 (with CUDA support recommended for GPU training)
- `molcrafts-molpy` ≥ 0.13 with `UnitSystem` `k_B` / `openmm` preset
  (pint-style `molix.units.units` requires them)
- `molcrafts-mollog` ≥ 1.3.0

Check your versions:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"
```

## Installing MolNex

Clone the repository and install in editable mode:

```bash
git clone https://github.com/molcrafts/molnex.git
cd molnex

# Install with dev dependencies
pip install -e ".[dev]"
```

To install documentation dependencies as well:

```bash
pip install -e ".[dev,docs]"
```

## Verification

Check that all components are importable:

```python
import molix
import molrep
import molpot
import molzoo.mace

print("MolNex ecosystem installed successfully.")
```

## Next Steps

- [Molix Quick Start](molix/tutorials/quick-start.md)
- [Train a Graph Model](molix/tutorials/train-a-graph-model.md)
