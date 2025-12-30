"""Base class for data processing nodes."""

from typing import Any, Mapping


class DataNode:
    """Data node base class implementing OpLike protocol.
    
    DataNode provides a composable interface for data transformations.
    Each node holds a reference to its upstream node (Dataset or DataNode),
    forming a chain that can be traced to visualize the data pipeline.
    
    Attributes:
        task_id: Unique identifier for this operation type
        upstream: Upstream node (Dataset or DataNode)
    """
    
    task_id: str = "data_node"
    
    def __init__(self, upstream=None):
        """Initialize data node.
        
        Args:
            upstream: Upstream node (Dataset or DataNode)
        """
        self.upstream = upstream
    
    def input_schema(self) -> Mapping[str, Any]:
        """Return input schema specification.
        
        Returns:
            Mapping describing expected inputs
        """
        return {"data": "Any"}
    
    def output_schema(self) -> Mapping[str, Any]:
        """Return output schema specification.
        
        Returns:
            Mapping describing output structure
        """
        return {"data": "Any"}
    
    def run(self, state=None, *, data=None) -> Mapping[str, Any]:
        """Execute the operation (OpLike protocol).
        
        Args:
            state: Optional state object
            data: Input data
            
        Returns:
            Mapping with processed data
        """
        return {"data": self.process(data)}
    
    def process(self, data: Any) -> Any:
        """Process data - core method to be implemented by subclasses.
        
        Args:
            data: Input data item
            
        Returns:
            Processed data item
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process() method"
        )
    
    def __len__(self) -> int:
        """Return length of upstream dataset.
        
        Returns:
            Number of items in upstream
        """
        return len(self.upstream)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item by index, applying processing.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Processed data item
        """
        data = self.upstream[idx]
        return self.process(data)
