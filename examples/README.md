# Molnex Examples

This directory contains example scripts demonstrating how to use molnex for molecular property prediction and potential modeling.

## QM9 Energy Prediction

**File**: [`qm9_energy_prediction.py`](qm9_energy_prediction.py)

A complete example of training a transformer-based model to predict molecular energy (U0) on the QM9 dataset.

### Architecture

```
AtomEmbedding (atomic numbers + positions)
    ↓
TransformerEncoder (4 layers, 4 heads, d_model=128)
    ↓
ScalarHead (mean pooling + MLP)
    ↓
Energy Prediction (U0 in eV)
```

### Features

- **Dataset**: QM9 (~130k molecules, automatic download)
- **Target**: Internal energy at 0K (U0)
- **Model**: Transformer with NestedTensor support for variable-length molecules
- **Training**: MSE loss with MAE metric tracking

### Quick Start

```bash
# From the molnex directory
cd molnex

# Install dependencies (if not already installed)
pip install -e .

# Run the example
python examples/qm9_energy_prediction.py
```

### Expected Output

```
Using device: cuda

==================================================
Loading QM9 Dataset
==================================================
Downloading QM9 dataset from https://ndownloader.figshare.com/files/3195389...
This may take a few minutes (~350 MB)...
Downloaded to ./data/qm9/qm9.tar.bz2
Found 133885 valid molecules in QM9 dataset
Train size: 107108
Val size: 13388

==================================================
Building Model
==================================================
Model parameters: 789,377

==================================================
Starting Training
==================================================

Epoch 1/10
--------------------------------------------------
  Batch 50/3348: Loss=0.2341, MAE=0.3456 eV
  Batch 100/3348: Loss=0.1823, MAE=0.2987 eV
  ...
  
Train - Loss: 0.1234, MAE: 0.2456 eV
Val   - Loss: 0.1156, MAE: 0.2312 eV

...

==================================================
Training Complete!
==================================================

Model saved to: ./qm9_energy_model.pt
```

### Hyperparameters

You can modify these in the `main()` function:

```python
# Model hyperparameters
d_model = 128              # Model dimension
nhead = 4                  # Number of attention heads
num_layers = 4             # Number of transformer layers
dim_feedforward = 512      # FFN hidden dimension
dropout = 0.1              # Dropout probability
pooling = "mean"           # Pooling method: 'mean', 'sum', or 'max'

# Training hyperparameters
batch_size = 32            # Batch size
learning_rate = 1e-4       # Learning rate
num_epochs = 10            # Number of training epochs
```

### Performance Notes

- **First run**: Dataset download takes a few minutes (~350 MB)
- **Training time**: ~5-10 minutes per epoch on GPU, ~30-60 minutes on CPU
- **Expected MAE**: 0.02-0.05 eV after 10 epochs (depends on hyperparameters)

### Code Structure

The example demonstrates the complete molnex workflow:

1. **Data Loading**: `QM9Dataset` with automatic download
2. **Data Collation**: Custom `qm9_collate_with_target` for batching with targets
3. **Model Definition**: `QM9EnergyModel` combining embedding, encoder, and head
4. **Training Step**: `QM9TrainStep` with MSE loss and gradient updates
5. **Evaluation Step**: `QM9EvalStep` for validation without gradients
6. **Training Loop**: Epoch-based training with metrics tracking

### Extending the Example

#### Change Target Property

To predict different QM9 properties (e.g., HOMO, LUMO, gap):

```python
# In qm9_collate_with_target()
u0_list = []
for frame in frames:
    u0 = frame.metadata["target"]["homo"]  # Change "U0" to "homo", "lumo", etc.
    u0_list.append(u0)

batch["target", "U0"] = torch.tensor(u0_list, dtype=torch.float32)
```

Available properties in QM9:
- `U0`, `U`, `H`, `G`: Energies at 0K (eV)
- `Cv`: Heat capacity (cal/mol K)
- `homo`, `lumo`, `gap`: Frontier orbitals (eV)
- `mu`: Dipole moment (Debye)
- `alpha`: Polarizability (Bohr³)

#### Use Different Pooling

```python
model = QM9EnergyModel(
    pooling="sum",  # or "max"
    ...
)
```

#### Increase Model Capacity

```python
model = QM9EnergyModel(
    d_model=256,           # Larger model
    nhead=8,               # More attention heads
    num_layers=6,          # Deeper network
    dim_feedforward=1024,  # Larger FFN
    ...
)
```

## Additional Examples

More examples coming soon:

- **MD17**: Molecular dynamics force field prediction
- **Multi-task**: Predicting multiple properties simultaneously
- **Custom architectures**: GNN-based models, equivariant networks
- **Transfer learning**: Pre-training and fine-tuning strategies

## Contributing

Have an interesting use case? Feel free to contribute examples via pull request!
