# Data Pipelines

Organizing data code is as important as organizing model code. Molix encourages the use of **DataModules** and **AtomicTD** (Atomic TensorDict) to keep your pipelines clean.

## Atomic TensorDict

### What is it?
`AtomicTD` is a specialized dictionary for molecular data. It groups arrays with specific meanings (like atomic positions `x` or atomic numbers `z`) and handles batching automatically.

### Creating Data
```python
from molix.data.atomic_td import AtomicTD
import torch

# Create a single molecule (Methane)
mol = AtomicTD.create(
    z=torch.tensor([6, 1, 1, 1, 1]),      # Carbon + 4 Hydrogens
    x=torch.randn(5, 3),                   # Random 3D positions
    batch=torch.zeros(5, dtype=torch.long) # All belong to molecule 0
)

# Access data
print(mol["atoms", "z"])  # tensor([6, 1, 1, 1, 1])
```

## DataModules

### Why use them?
A `DataModule` encapsulates all steps needed to process data: downloading, splitting, and creating DataLoaders. This makes your training script reproducible and easy to share.

Under the hood, DataModules should follow the [Standard Data Loading Pattern](loading.md) used throughout Molnex. This means your datasets will typically wrap `Sequence[molpy.Frame]` and your loaders will use `molix.data.collate.nested_collate_fn`.

### How to write one
A DataModule is any class with `train_dataloader()` and `val_dataloader()` methods.

```python
class MyDataModule:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # 1. Download or load data
        self.train_dataset, self.val_dataset = load_my_data(self.data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
```

### Using it with Trainer

```python
dm = MyDataModule("./data")
dm.prepare_data()

trainer.train(dm, max_epochs=100)
```
