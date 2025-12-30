"""Common data processing operations."""

from typing import Any, Callable, List, Optional

from molix.data.node import DataNode


class TransformOp(DataNode):
    """Transform data using a custom function.
    
    Applies a transformation function to each data item.
    
    Attributes:
        task_id: Operation identifier
        transform_fn: Function to apply to each data item
    """
    
    task_id = "transform"
    
    def __init__(self, upstream, transform_fn: Callable[[Any], Any]):
        """Initialize transform operation.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
            transform_fn: Function to apply to each data item
        """
        super().__init__(upstream)
        self.transform_fn = transform_fn
    
    def process(self, data: Any) -> Any:
        """Apply transformation function.
        
        Args:
            data: Input data item
            
        Returns:
            Transformed data item
        """
        return self.transform_fn(data)


class FilterOp(DataNode):
    """Filter data based on a predicate.
    
    Filters dataset to include only items satisfying the predicate.
    Pre-computes valid indices for efficient access.
    
    Attributes:
        task_id: Operation identifier
        predicate: Function returning True for items to keep
    """
    
    task_id = "filter"
    
    def __init__(self, upstream, predicate: Callable[[Any], bool]):
        """Initialize filter operation.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
            predicate: Function returning True for items to keep
        """
        super().__init__(upstream)
        self.predicate = predicate
        # Pre-compute valid indices
        self._valid_indices = [
            i for i in range(len(upstream))
            if predicate(upstream[i])
        ]
    
    def __len__(self) -> int:
        """Return number of items after filtering.
        
        Returns:
            Number of valid items
        """
        return len(self._valid_indices)
    
    def __getitem__(self, idx: int) -> Any:
        """Get filtered item by index.
        
        Args:
            idx: Index in filtered dataset
            
        Returns:
            Data item at filtered index
        """
        real_idx = self._valid_indices[idx]
        return self.upstream[real_idx]
    
    def process(self, data: Any) -> Any:
        """Pass through data (filtering handled in __getitem__).
        
        Args:
            data: Input data item
            
        Returns:
            Unchanged data item
        """
        return data


class NormalizeOp(DataNode):
    """Normalize data using mean and standard deviation.
    
    Applies z-score normalization: (data - mean) / std
    Supports both dict and tensor data.
    
    Attributes:
        task_id: Operation identifier
        mean: Mean value for normalization
        std: Standard deviation for normalization
        key: Optional key for dict data
    """
    
    task_id = "normalize"
    
    def __init__(
        self,
        upstream,
        mean: float,
        std: float,
        key: Optional[str] = None
    ):
        """Initialize normalize operation.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
            mean: Mean value for normalization
            std: Standard deviation for normalization
            key: Optional key for dict data (normalizes data[key])
        """
        super().__init__(upstream)
        self.mean = mean
        self.std = std
        self.key = key
    
    def process(self, data: Any) -> Any:
        """Apply normalization.
        
        Args:
            data: Input data item (dict or tensor)
            
        Returns:
            Normalized data item
        """
        if self.key and isinstance(data, dict):
            # Normalize specific key in dict
            data = data.copy()  # Avoid mutating original
            data[self.key] = (data[self.key] - self.mean) / self.std
            return data
        else:
            # Normalize entire data
            return (data - self.mean) / self.std


class CacheOp(DataNode):
    """Cache data in memory.
    
    Pre-loads all data from upstream into memory for faster access.
    Useful for small datasets that fit in memory.
    
    Attributes:
        task_id: Operation identifier
    """
    
    task_id = "cache"
    
    def __init__(self, upstream):
        """Initialize cache operation.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
        """
        super().__init__(upstream)
        # Pre-load all data
        self._cache: List[Any] = [
            upstream[i] for i in range(len(upstream))
        ]
    
    def __getitem__(self, idx: int) -> Any:
        """Get cached item by index.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Cached data item
        """
        return self._cache[idx]
    
    def process(self, data: Any) -> Any:
        """Pass through data (caching handled in __init__).
        
        Args:
            data: Input data item
            
        Returns:
            Unchanged data item
        """
        return data
