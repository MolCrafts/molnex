"""Atom-type classification head."""

import torch
import torch.nn as nn


class TypeHead(nn.Module):
    """Predict atom types from atomic representations."""

    def __init__(self, hidden_dim: int = 64, num_types: int = 100):
        """Initialize type head.

        Args:
            hidden_dim: Dimension of hidden representation
            num_types: Number of atom types to predict
        """
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, atoms_h: torch.Tensor) -> torch.Tensor:
        """Predict atom type logits.

        Args:
            atoms_h: Atomic hidden states [N, D]

        Returns:
            Type logits [N, num_types]
        """
        return self.module(atoms_h)
