"""Protocol definitions for structural compatibility with MolExp.

These protocols are LOCAL to MolNex and exist for:
- Documentation
- Local type checking
- Internal consistency

They do NOT imply any shared dependency with MolExp.
Compatibility is achieved through duck typing (structural compatibility).
"""

from typing import Any, Mapping, Protocol, Sequence


class OpLike(Protocol):
    """Protocol defining the shape MolExp expects for operations.
    
    This is a MolNex-local definition for documentation and type checking.
    No inheritance or runtime checks against MolExp are performed.
    
    Step objects (TrainStep, EvalStep, etc.) satisfy this protocol structurally.
    """
    
    op_name: str
    """Unique identifier for this operation."""
    
    def input_schema(self) -> Mapping[str, Any]:
        """Return input schema specification.
        
        Returns:
            Mapping describing expected inputs (e.g., {"batch": "Any", "train_state": "TrainState"})
        """
        ...
    
    def output_schema(self) -> Mapping[str, Any]:
        """Return output schema specification.
        
        Returns:
            Mapping describing output structure (e.g., {"loss": "float", "logs": "Mapping"})
        """
        ...
    
    def run(self, train_state: Any, *, batch: Any) -> Mapping[str, Any]:
        """Execute the operation.
        
        Args:
            train_state: Current training state
            batch: Input batch data
            
        Returns:
            Mapping with execution results (loss, result, logs, etc.)
        """
        ...


class GraphLike(Protocol):
    """Protocol defining the shape MolExp expects for graphs.
    
    This is a MolNex-local definition for documentation and type checking.
    No inheritance or runtime checks against MolExp are performed.
    
    The Graph class satisfies this protocol structurally.
    """
    
    nodes: Sequence[OpLike]
    """Sequence of operation nodes (step objects)."""
    
    edges: Sequence[Any]
    """Sequence of edges representing flow between nodes."""
    
    meta: Mapping[str, Any]
    """Metadata about the graph (stage ordering, loop structure, etc.)."""
