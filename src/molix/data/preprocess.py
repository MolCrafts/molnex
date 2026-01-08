"""Dataset preprocessors for unified data processing and caching.

DatasetPreprocessor provides a base class for preprocessing operations that:
1. Can be fitted on a dataset (if needed)
2. Process individual frames
3. Are cached within the dataset during creation/loading
"""

"""Data preprocessing for molecular datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Generic, TypeVar
from pydantic import BaseModel, Field
import numpy as np
import torch
from tqdm import tqdm
from molnex.F.locality import get_neighbor_pairs
from molix.data.atomic_td import AtomicTD

import molpy


T = TypeVar("T", bound=BaseModel)


class DatasetPreprocessor(ABC, Generic[T]):
    """Base class for dataset preprocessing operations.
    
    DatasetPreprocessor provides a unified interface for preprocessing operations
    that process all frames. Preprocessors can compute global statistics from all
    frames (like AtomicDressPreprocessor) or process frames individually
    (like NeighborListPreprocessor).
    
    Subclasses should:
    1. Define a config_type (Pydantic BaseModel) for configuration
    2. Implement preprocess() to process all frames
    
    Usage:
        >>> class MyPreprocessor(DatasetPreprocessor):
        ...     config_type = MyConfig
        ...     
        ...     def preprocess(self, frames):
        ...         # Process all frames
        ...         return processed_frames
        >>> 
        >>> preprocessor = MyPreprocessor(param=value)
        >>> processed_frames = preprocessor.preprocess(all_frames)
    """
    
    config_type: type[T] = BaseModel
    
    def __init__(self, **config_kwargs: Any):
        """Initialize preprocessor with configuration.
        
        Args:
            **config_kwargs: Configuration parameters (validated against config_type)
        """
        self.config: T = self.config_type(**config_kwargs)
    
    @abstractmethod
    def preprocess(self, frames: Sequence[molpy.Frame]) -> Sequence[molpy.Frame]:
        """Process all frames.
        
        This method receives all frames and processes them. Preprocessors that
        need global statistics (like AtomicDressPreprocessor) compute them from
        all frames, then apply transformations. Preprocessors that process
        individually (like NeighborListPreprocessor) process each frame.
        
        Args:
            frames: Sequence of all frames to process
            
        Returns:
            Sequence of processed frames
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement preprocess() method"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(config={self.config!r})"


# ============================================================================
# Concrete Preprocessor Implementations
# ============================================================================
class AtomicDressConfig(BaseModel):
    """Configuration for atomic dress preprocessing."""
    elements: list[int]
    target_key: str = "U0"
    unit: str = "eV"


class AtomicDressPreprocessor(DatasetPreprocessor):
    """Subtract element-dependent atomic energies from target property.
    
    Fits a linear model: E_total = sum_i(E_atom_i) + E_residual
    Then subtracts atomic contributions from the target.
    """
    
    def __init__(self, elements: list[int], target_key: str = "U0", unit: str = "eV"):
        self.config = AtomicDressConfig(elements=elements, target_key=target_key, unit=unit)
        self.atomic_energies: dict[int, float] = {}
        self.fit_error: float | None = None
    
    def fit(self, frames: Sequence["AtomicTD"]) -> None:
        """Fit atomic energies from frames.
        
        Args:
            frames: Sequence of AtomicTD to fit on
        """
        print(f"Fitting atomic dress for elements {self.config.elements}...")
        
        X_list = []
        y_list = []
        
        for frame in tqdm(frames, desc="Fitting atomic dress"):
            # Count atoms of each element
            counts = torch.zeros(len(self.config.elements))
            for i, elem in enumerate(self.config.elements):
                counts[i] = (frame['atoms', "Z"] == elem).sum().item()
            
            X_list.append(counts)
            y_list.append(frame[("target", self.config.target_key)].item())
        
        X = torch.stack(X_list).numpy()
        y = np.array(y_list)
        
        # Solve: X @ beta = y using pseudoinverse
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        # Store atomic energies
        self.atomic_energies = {
            elem: float(beta[i]) for i, elem in enumerate(self.config.elements)
        }
        
        # Compute fit error
        residual = X @ beta - y
        self.fit_error = float(np.sqrt(np.mean(residual**2)))
        
        print(f"Atomic energies ({self.config.unit}):")
        for elem, energy in self.atomic_energies.items():
            print(f"  Element {elem}: {energy:.4f}")
        print(f"Fit RMSE: {self.fit_error:.4f} {self.config.unit}\n")
    
    def preprocess(self, frames: Sequence["AtomicTD"]) -> Sequence["AtomicTD"]:
        """Subtract atomic contributions from target.
        
        Must call fit() first!
        
        Args:
            frames: Sequence of AtomicTD to preprocess
            
        Returns:
            Sequence of AtomicTD with atomic contributions subtracted
        """
        if not self.atomic_energies:
            self.fit(frames)
        
        result_frames = []
        for frame in tqdm(frames, desc="atomic dressing"):
            # Compute atomic contribution
            contrib = sum(self.atomic_energies[int(z.item())] for z in frame['atoms', "Z"])
            
            # Subtract from target
            target_key = ("target", self.config.target_key)
            new_target = frame[target_key] - contrib
            
            # Create new frame with updated target
            new_data = {k: v for k, v in frame.items()}
            new_data[target_key] = new_target
            
            new_frame = AtomicTD(new_data, batch_size=[])
            result_frames.append(new_frame)
        return result_frames


class NeighborListConfig(BaseModel):
    """Configuration for neighbor list computation."""
    cutoff: float = Field(default=5.0)
    max_num_pairs: int = Field(default=128)
    pbc: bool = Field(default=False)
    check_errors: bool = Field(default=True)
    filter_padding: bool = Field(default=True, description="Filter out NaN/invalid padding from output")


class NeighborListPreprocessor(DatasetPreprocessor[NeighborListConfig]):
    """Add neighbor list to each frame.
    
    Computes pairwise neighbor lists within a cutoff distance and adds
    bond topology fields to AtomicTD.
    """
    
    config_type = NeighborListConfig
    
    def __init__(self, **config_kwargs: Any):
        super().__init__(**config_kwargs)
    
    def preprocess(self, frames: Sequence["AtomicTD"]) -> Sequence["AtomicTD"]:
        """Compute neighbor lists for each frame.
        
        Args:
            frames: Sequence of AtomicTD objects
            
        Returns:
            Sequence of AtomicTD with bond topology added
        """
        
        result_frames = []
        for frame in tqdm(frames, desc="Computing neighbor lists"):
            # Get positions (already a tensor in AtomicTD)
            positions = frame["atoms", "xyz"]
            
            # Determine box_vectors for PBC
            box_vectors = None
            if self.config.pbc:
                # AtomicTD doesn't have metadata for box, so use None
                # This will be handled by get_neighbor_pairs
                box_vectors = None
            
            # Compute neighbor pairs
            edge_index, edge_vec, edge_dist, num_pairs = get_neighbor_pairs(
                positions=positions,
                cutoff=self.config.cutoff,
                max_num_pairs=self.config.max_num_pairs,
                box_vectors=box_vectors,
                check_errors=self.config.check_errors,
            )
            
            # Filter out padding if requested
            if self.config.filter_padding:
                # C++ op returns fixed-size arrays with -1 and nan for unused slots
                valid_mask = ~torch.isnan(edge_dist)
                edge_i = edge_index[0][valid_mask]
                edge_j = edge_index[1][valid_mask]
                edge_vec = edge_vec[valid_mask]
                edge_dist = edge_dist[valid_mask]
            else:
                edge_i = edge_index[0]
                edge_j = edge_index[1]
            
            frame[("pairs", "i")] = edge_i
            frame[("pairs", "j")] = edge_j
            frame[("pairs", "diff")] = edge_vec
            frame[("pairs", "dist")] = edge_dist
        
        return frames
