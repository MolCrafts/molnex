"""Type Head for discrete atom type classification."""

import torch
import torch.nn as nn


class TypeHead(nn.Module):
    """Classification head for discrete atom type prediction.

    Args:
        hidden_dim: Dimension of input embeddings
        num_types: Number of type classes
        dropout: Dropout rate
    """

    def __init__(self, hidden_dim: int, num_types: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_types = num_types

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute type logits from embeddings."""
        return self.classifier(embeddings)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to type indices."""
        return logits.argmax(dim=-1)

    def decode_labels(
        self,
        logits: torch.Tensor,
        type_map: dict[int, str],
    ) -> list[str]:
        """Decode logits to string labels."""
        indices = self.decode(logits)
        return [type_map.get(idx.item(), f"UNK_{idx.item()}") for idx in indices]

    def decode_with_confidence(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode with confidence scores."""
        probs = torch.softmax(logits, dim=-1)
        confidence, indices = probs.max(dim=-1)
        return indices, confidence
