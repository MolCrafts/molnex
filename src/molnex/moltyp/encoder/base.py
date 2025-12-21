from typing import Protocol
import torch
from ..data import Batch

class Encoder(Protocol):
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Encode a batch of molecules into node embeddings.
        
        Args:
            batch: Batch object containing molecule data.
            
        Returns:
            torch.Tensor: Node embeddings of shape (N_total_atoms, hidden_dim)
        """
        ...
