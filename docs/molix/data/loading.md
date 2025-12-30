# Data Loading Patterns

Molnex follows a strict but flexible pattern for loading molecular data. The core philosophy is to keep data in a standard, easy-to-manipulate format (`molpy.Frame`) as long as possible, and only convert to efficient tensor structures (`AtomicTD`) at the last possible moment (collate).

## The Pipeline

The recommended data pipeline consists of three stages:

1.  **Source**: Data is loaded into a python sequence of `molpy.Frame` objects (`Sequence[molpy.Frame]`).
2.  **Dataset**: A `torch.utils.data.Dataset` wraps this sequence, responsible for indexing and on-the-fly transformations.
3.  **Loader**: A `torch.utils.data.DataLoader` batches the frames and uses a specific `collate_fn` to convert them into a batched `AtomicTD`.

```mermaid
graph LR
    A[Source Files] --> B[Sequence[molpy.Frame]]
    B --> C[Dataset]
    C --> D[DataLoader]
    D -- nested_collate_fn --> E[Batched AtomicTD]
```

## 1. The Common Currency: `molpy.Frame`

The foundational data structure for IO in Molnex is the `molpy.Frame`. A `Frame` represents a molecular system (atoms, positions, cells, properties).

**Why `molpy.Frame`?**
- It is a pure Python/NumPy object, easy to debug and inspect.
- It abstracts away file formats (XYZ, LAMMPS, VASP, etc.).
- It is efficient enough for storage in memory lists for small-to-medium datasets (up to millions of frames).

> **Note**: You can also use standard Python dictionaries if they follow the structure `{"atoms": {"x": ..., "z": ...}}`.

## 2. The Dataset Wrapper

You should implement a `torch.utils.data.Dataset` that wraps your list of frames. This dataset is the place to apply frame-level transformations (like filtering, property calculation, or neighbor list pre-computation if strictly necessary).

```python
import torch
from typing import Sequence
import molpy

class FramesDataset(torch.utils.data.Dataset):
    def __init__(self, frames: Sequence[molpy.Frame], transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        if self.transform:
            frame = self.transform(frame)
            
        return frame
```

## 3. The Collate Function

The magic happens in the `collate_fn`. This is where a list of `molpy.Frame` objects (a batch) gets converted into a single `AtomicTD` (Atomic TensorDict).

Molnex provides **`molix.data.collate.nested_collate_fn`** for this purpose.

### NestedTensors

`nested_collate_fn` produces an `AtomicTD` containing PyTorch **NestedTensors** for atomic properties. This is critical for handling batches of molecules with different numbers of atoms without wasteful padding.

- `("atoms", "x_nt")`: NestedTensor of positions.
- `("atoms", "z_nt")`: NestedTensor of atomic numbers.

```python
from torch.utils.data import DataLoader
from molix.data.collate import nested_collate_fn

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=nested_collate_fn
)
```

## Advanced Patterns

### Lazy Loading (Large Datasets)
For datasets that do not fit in memory (e.g., millions of structures), do not load `frames` into a list. Instead, store file paths or indices and load the frame in `__getitem__`.

```python
class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Read from disk on demand
        frame = molpy.io.read_xyz_single(self.file_paths[idx])
        return frame
```

### Custom Collation (Labels & Energies)
`nested_collate_fn` only handles standard structural data (`x`, `z`, `box`). If you need to batch training targets like Energy or Forces, wrap the collate function.

```python
def custom_collate(batch):
    # 1. Collate structure
    atomic_td = nested_collate_fn(batch)
    
    # 2. Collate properties
    # Assume 'energy' is stored in frame.properties
    energies = torch.stack([
        torch.tensor(b["properties"]["energy"]) for b in batch
    ])
    
    # Return a tuple or add to atomic_td if appropriate
    return atomic_td, energies
```

### Performance Tuning
- **Num Workers**: Set `num_workers > 0` in `DataLoader` to parallelize data loading (crucial for Lazy Loading).
- **Pin Memory**: Set `pin_memory=True` to speed up transfer to GPU.

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    collate_fn=nested_collate_fn,
    num_workers=4,
    pin_memory=True
)
```
