"""Atom-type classification head."""

import torch
import torch.nn as nn

from molix import config


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
            nn.Linear(hidden_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_types, dtype=config.ftype),
        )

    def forward(self, atoms_h: torch.Tensor) -> torch.Tensor:
        """Predict atom type logits.

        Args:
            atoms_h: Atomic hidden states [N, D]

        Returns:
            Type logits [N, num_types]
        """
        return self.module(atoms_h)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to type indices."""
        return logits.argmax(dim=-1)

    def decode_with_confidence(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode with confidence scores (softmax max).

        Args:
            logits: Type logits ``(N, num_types)``.

        Returns:
            ``(indices, confidence)`` each of shape ``(N,)``.
        """
        probs = torch.softmax(logits, dim=-1)
        confidence, indices = probs.max(dim=-1)
        return indices, confidence
