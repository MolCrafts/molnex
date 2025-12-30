"""Data pipeline that inherits from DataLoader."""

from typing import Any, Iterable, Mapping, Optional

from torch.utils.data import DataLoader

from molix.data.node import DataNode


class DataPipeline(DataLoader):
    """Data pipeline inheriting from DataLoader, implementing GraphLike protocol.
    
    DataPipeline is a DataLoader that can trace its upstream data processing
    nodes, enabling visualization and composition as a workflow subgraph.
    
    The pipeline can be used directly as a DataLoader for iteration, while
    also exposing nodes, edges, and meta properties for workflow introspection.
    
    Example:
        >>> dataset = MyDataset([...])
        >>> pipeline = DataPipeline(
        ...     TransformOp(
        ...         NormalizeOp(dataset, mean=0, std=1),
        ...         lambda x: {'input': x['coords'], 'target': x['energy']}
        ...     ),
        ...     batch_size=32,
        ...     shuffle=True
        ... )
        >>> for batch in pipeline:
        ...     print(batch)
        >>> print(pipeline.nodes)  # [NormalizeOp, TransformOp]
    
    Attributes:
        upstream: Upstream node (Dataset or DataNode)
    """
    
    def __init__(
        self,
        upstream,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler=None,
        batch_sampler=None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """Initialize data pipeline.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
            batch_size: How many samples per batch to load
            shuffle: Set to True to have data reshuffled at every epoch
            sampler: Defines the strategy to draw samples from the dataset
            batch_sampler: Like sampler, but returns a batch of indices at a time
            num_workers: How many subprocesses to use for data loading
            collate_fn: Merges a list of samples to form a mini-batch
            pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory
            drop_last: Set to True to drop the last incomplete batch
            timeout: Timeout value for collecting a batch from workers
            worker_init_fn: If not None, called on each worker subprocess
            multiprocessing_context: Multiprocessing context
            generator: Random number generator for shuffling
            prefetch_factor: Number of batches loaded in advance by each worker
            persistent_workers: If True, the data loader will not shutdown workers
            pin_memory_device: The device to pin_memory to
        """
        # Call DataLoader's __init__
        super().__init__(
            dataset=upstream,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            **({"prefetch_factor": prefetch_factor} if prefetch_factor is not None else {}),
            persistent_workers=persistent_workers,
            **({"pin_memory_device": pin_memory_device} if pin_memory_device else {}),
        )
        self.upstream = upstream
        self._batch_size = batch_size
    
    @property
    def tasks(self) -> Iterable[DataNode]:
        """Return all data processing tasks (from upstream trace).
        
        Traces upstream from the current node to collect all DataNode
        instances in the processing chain.
        
        Returns:
            List of DataNode instances in processing order
        """
        tasks = []
        current = self.upstream
        while current is not None:
            if isinstance(current, DataNode):
                tasks.append(current)
                current = current.upstream
            else:
                # Reached base Dataset, stop
                break
        return list(reversed(tasks))  # Reverse to get source-to-sink order
    
    @property
    def links(self) -> Iterable[Mapping[str, Any]]:
        """Return links connecting data processing tasks.
        
        Returns:
            List of link dictionaries describing data flow
        """
        tasks = self.tasks
        links = []
        for i in range(len(tasks) - 1):
            links.append({
                "source": tasks[i].task_id,
                "target": tasks[i + 1].task_id,
                "type": "data_flow"
            })
        return links
    
    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return pipeline metadata.
        
        Returns:
            Dictionary with pipeline configuration and statistics
        """
        return {
            "pipeline_type": "data",
            "batch_size": self._batch_size,
            "num_ops": len(self.tasks),
            "loader_config": {
                "shuffle": self.sampler is not None or getattr(self, '_shuffle', False),
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "drop_last": self.drop_last,
            }
        }
