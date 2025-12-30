# Installation

## Prerequisites

### What you need
Before installing MolNex, ensure you have a standard Python environment set up. We recommend using Python 3.10 or newer. You will also need PyTorch 2.0+ installed, ideally with CUDA support if you plan to train on GPUs.

### Why these versions
MolNex leverages modern Python features for type safety and performance. We rely on recent PyTorch versions for the best support of dynamic graphs and compiled operations, which are critical for molecular machine learning.

### How to check
You can check your current versions with:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"
```

## Installing MolNex

### What to do
The recommended way to install MolNex is to clone the repository and install it in editable mode.

### Why editable?
Installing in editable mode (`-e`) allows you to modify the source code and essentially "develop" on top of the framework without constantly reinstalling. This is standard practice for research-heavy ML workflows.

### How to install

```bash
# Clone the repository
git clone https://github.com/yourusername/molcrafts.git
cd molcrafts/molnex

# Install in editable mode
pip install -e .
```

## Installing Individual Modules

### What it is
You can choose to install only specific sub-packages like `molix`, `molrep`, or `molpot` if you do not need the full ecosystem.

### Why split them up
If you only need our representation learning layers (`molrep`) for your existing project, there is no need to pull in the full training framework. This keeps your dependencies light.

### How to do it

**To install only the training framework:**
```bash
cd molnex/src/molix
pip install -e .
```

**To install only representations:**
```bash
cd molnex/src/molrep
pip install -e .
```

**To install only potentials:**
```bash
cd molnex/src/molpot
pip install -e .
```

## Verification

### What it is
A simple check to ensure that all components are correctly installed and importable.

### Why verify
It is better to catch installation issues now than to encounter `ImportError` in the middle of a long training run.

### How to verify

```python
import molix
import molrep
import molpot

print(f"MolNex ecosystem installed successfully.")
print(f"molix: {molix.__version__}")
print(f"molrep: {molrep.__version__}")
print(f"molpot: {molpot.__version__}")
```

## Next Steps

Now that you have the framework installed, you are ready to build your first model.

- [Quick Start Guide](quick-start.md) - Train a model in 5 minutes.
