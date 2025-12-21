from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Data:
    """
    Data object representing a single molecule.
    """
    z: torch.Tensor          # (N,) LongTensor: atomic numbers
    pos: torch.Tensor        # (N, 3) FloatTensor: coordinates
    edge_index: Optional[torch.Tensor] = None # (2, E) LongTensor: graph connectivity
    edge_attr: Optional[torch.Tensor] = None  # (E, D) FloatTensor: edge features
    y: Optional[torch.Tensor] = None          # (N,) LongTensor: node labels (types)
    
    def __post_init__(self):
        if self.z.dim() != 1:
            raise ValueError(f"z must be 1D, got {self.z.shape}")
        if self.pos.dim() != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"pos must be (N, 3), got {self.pos.shape}")
        if self.z.shape[0] != self.pos.shape[0]:
            raise ValueError(f"z and pos must have same number of atoms, got {self.z.shape[0]} and {self.pos.shape[0]}")

    @property
    def num_nodes(self) -> int:
        return self.z.shape[0]
