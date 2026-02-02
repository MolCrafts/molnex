# Getting Started with MolPot

MolPot is the physics engine of MolNex. It allows you to build composite potentials—from simple harmonic springs to complex Message Passing Neural Networks (MPNNs)—using a unified "Lego block" interface.

## Start Here

1.  [**Quickstart**](quickstart.md): Calculate bond energy in 3 minutes.
2.  [**Components**](components.md): Build a Neural Network Potential (RBFs, MPNNs).
3.  [**Gradients & Forces**](gradients.md): How we compute Molecular Dynamics forces.

## Key Concepts

### Component-Based
Unlike monolithic codes like LAMMPS, MolPot potentials are PyTorch Modules. You can mix a classical Lennard-Jones potential with a Neural Network correction term by simply adding them together: `model = Classical() + NN()`.

### Pure PyTorch
We avoid heavy C++ kernels where possible. This makes the code readable, debuggable, and easy to modify for research. We rely on `torch.compile` for performance.

## Design

All potentials take an `AtomTD` as input and output scalar energy (and optionally forces). This standardized contract makes it easy to swap potentials in benchmarks.
python
from molpot import GaussianRBF, CosineCutoff

# Create 50 filters for distances up to 5.0 Angstroms
rbf = GaussianRBF(num_rbf=50, cutoff=5.0)

cutoff = CosineCutoff(cutoff=5.0)
```

**2. Interaction (The Neural Network)**
Process these features to learn interactions.

```python
from molpot import MLP

# A simple Multi-Layer Perceptron
mlp = MLP(in_dim=50, out_dim=128, hidden_dims=[128])
```

**3. Readout**
Sum individual atomic contributions to get total molecular energy.

```python
from molpot import SumPooling, EnergyHead

# Sum atomic energies
pooling = SumPooling()
head = EnergyHead()
```

## Pre-Assembled Models

### What it is
While custom building is powerful, `molpot` also ships with industry-standard architectures ready to use.

### Why use them?
They provide a proven baseline. You can use them to benchmark your new ideas or simply as a reliable force field for your simulations.

### How to use it
Instantiate `PiNet2` with a single line.

```python
from molpot import PiNet2

# Ready to train out of the box
model = PiNet2(
    hidden_dim=128,
    num_layers=3,
    cutoff=5.0
)
```

## Pure PyTorch Implementation

### What it is
All `molpot` operations are written in native PyTorch. We do not rely on PyTorch Geometric (PyG) or custom C++ CUDA kernels for the core logic.

### Why avoid PyG/CUDA?
*   **Debugging**: You can step through every line of code with a standard debugger.
*   **Deployment**: No complex compilations or wheel compatibility matrices. It runs wherever PyTorch runs.
