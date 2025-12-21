"""Graph representation for training structure."""

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence


@dataclass
class Graph:
    """Graph representation of training structure.
    
    This graph:
    - Is produced by Trainer.to_graph()
    - Is deterministic
    - Represents semantic structure, not execution trace
    - Is structurally compatible with MolExp's GraphLike protocol
    
    Attributes:
        nodes: Sequence of step objects (OpLike protocol)
        edges: Sequence of edges representing flow
        meta: Metadata about graph structure
    """
    
    nodes: Sequence[Any] = field(default_factory=list)
    edges: Sequence[Any] = field(default_factory=list)
    meta: Mapping[str, Any] = field(default_factory=dict)
