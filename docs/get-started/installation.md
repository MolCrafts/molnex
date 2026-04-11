# Installation

## Prerequisites

Before installing MolNex, ensure you have:

- Python >= 3.10
- PyTorch >= 2.6 (with CUDA support recommended for GPU training)

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

## Verification

Check that all components are importable:

```python
import molix
import molrep
import molpot
import molzoo

print("MolNex ecosystem installed successfully.")
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Train a model in 5 minutes.
