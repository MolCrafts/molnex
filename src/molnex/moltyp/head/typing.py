import torch
import torch.nn as nn
from .base import Head

class TypeClassifier(nn.Module, Head):
    """
    Classification head for atom types.
    """
    def __init__(self, hidden_dim: int, num_types: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_types)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
