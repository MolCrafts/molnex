"""Base dataset class for molix with built-in preprocess support."""

from typing import Sequence, Tuple
from torch.utils.data import Dataset as TorchDataset
import torch

import molpy as mp

from .preprocess import DatasetPreprocessor


class Dataset(TorchDataset[mp.Frame]):
    """Base dataset class for molix with built-in preprocess support.
    
    This class extends torch.utils.data.Dataset with molix-specific features:
    - Built-in preprocess support
    - Type hints for Frame objects
    - Automatic preprocessing and caching
    - Dataset splitting functionality
    
    Args:
        preprocessors: Optional sequence of preprocessors to apply to each sample
        
    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data, preprocessors=None):
        ...         super().__init__(preprocessors=preprocessors)
        ...         self.data = data
        ...     
        ...     def __len__(self):
        ...         return len(self.data)
        ...     
        ...     def __getitem__(self, idx) -> mp.Frame:
        ...         item = self.data[idx]
        ...         return self._apply_preprocess(item)
        >>> 
        >>> # Use with preprocessors
        >>> from molix.data.preprocess import AtomicDressPreprocessor, NeighborListPreprocessor
        >>> atomic_dress = AtomicDressPreprocessor(elements=[1, 6, 7, 8, 9])
        >>> neighbor_list = NeighborListPreprocessor(cutoff=5.0)
        >>> atomic_dress.fit(base_dataset)
        >>> dataset = MyDataset(data, preprocessors=[atomic_dress, neighbor_list])
        >>> 
        >>> # Split dataset
        >>> train_dataset, val_dataset = dataset.split(train_ratio=0.8, random_seed=42)
    """
    
    def __init__(self, preprocessors: Sequence[DatasetPreprocessor] | None = None):
        """Initialize dataset with optional preprocessors.
        
        Args:
            preprocessors: Optional sequence of preprocessors to apply
        """
        self.preprocessors = list(preprocessors) if preprocessors else []
        self._preprocessed = False
        self._processed_frames: dict[int, mp.Frame] | None = None
    
    def _apply_preprocess(self, frames: Sequence[mp.Frame]) -> Sequence[mp.Frame]:
        """Apply preprocessors to all frames.
        
        This method processes all frames together. Preprocessors that need
        global statistics (like AtomicDressPreprocessor) compute statistics
        from all frames, then apply transformations to each frame.
        
        Preprocessors that process frames individually (like NeighborListPreprocessor)
        are applied to each frame.
        
        Args:
            frames: Sequence of all frames to preprocess
            
        Returns:
            Sequence of preprocessed frames
        """
        if self._preprocessed and self._processed_frames is not None:
            # Return cached processed frames
            return [self._processed_frames[i] for i in range(len(frames))]
        
        if not self.preprocessors:
            self._preprocessed = True
            self._processed_frames = {i: frame for i, frame in enumerate(frames)}
            return frames
        
        # Process all frames through preprocessors
        # Preprocessors that need global statistics will compute them from all frames
        # Preprocessors that process individually will process each frame
        result_frames = list(frames)
        
        for preprocessor in self.preprocessors:
            # Preprocessors receive all frames and process them
            result_frames = preprocessor.preprocess(result_frames)
        
        # Cache processed frames
        self._processed_frames = {i: frame for i, frame in enumerate(result_frames)}
        self._preprocessed = True
        return result_frames
    
    def split(
        self,
        train_ratio: float = 0.8,
        random_seed: int | None = 42,
    ) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into train and validation subsets.
        
        This method uses torch.utils.data.random_split internally but returns
        Dataset instances that preserve preprocessors and other dataset properties.
        
        Args:
            train_ratio: Ratio of training samples (default: 0.8)
            random_seed: Random seed for splitting (default: 42)
            
        Returns:
            Tuple of (train_dataset, val_dataset)
            
        Example:
            >>> train_dataset, val_dataset = dataset.split(train_ratio=0.8, random_seed=42)
        """
        from torch.utils.data import random_split
        
        total_size = len(self)  # type: ignore[arg-type]
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
        train_subset, val_subset = random_split(
            self, [train_size, val_size], generator=generator
        )
        
        # Wrap Subset in Dataset to preserve preprocessors
        train_dataset = _SubsetDataset(self, train_subset.indices)
        val_dataset = _SubsetDataset(self, val_subset.indices)
        
        print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val\n")
        return train_dataset, val_dataset


class _SubsetDataset(Dataset):
    """Internal wrapper for dataset subsets that preserves preprocessors."""
    
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        """Initialize subset dataset.
        
        Args:
            dataset: Parent dataset
            indices: Indices to include in subset
        """
        super().__init__(preprocessors=dataset.preprocessors)
        self.dataset = dataset
        self.indices = list(indices)
    
    def __len__(self) -> int:
        """Return length of subset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> mp.Frame:
        """Get item from subset.
        
        Args:
            idx: Index in subset
            
        Returns:
            Frame from parent dataset
        """
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]
