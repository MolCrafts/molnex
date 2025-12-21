from typing import Protocol, Any, Optional
import torch

class Head(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map node embeddings to outputs.
        
        Args:
            x: Node embeddings (N, D)
            
        Returns:
            torch.Tensor: Outputs (N, C)
        """
        ...
