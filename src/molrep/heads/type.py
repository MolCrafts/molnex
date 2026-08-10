"""Type heads for discrete classification (atom types and multi-class systems).

``TypeHead`` remains the single-system atom/interaction classifier.
``MultiTypeHead`` holds one :class:`TypeHead` per named interaction class
without a ``method=`` switch — composition is the caller's job.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from molix import config

if TYPE_CHECKING:
    from molrep.condensation.type_system import TypeSystem

__all__ = ["TypeHead", "MultiTypeHead"]


class TypeHead(nn.Module):
    """Classification head for discrete type prediction.

    Args:
        hidden_dim: Dimension of input embeddings.
        num_types: Number of type classes.
        dropout: Dropout rate.
    """

    def __init__(self, hidden_dim: int, num_types: int, dropout: float = 0.0):
        super().__init__()
        if num_types < 1:
            raise ValueError(f"num_types must be >= 1, got {num_types}")
        self.hidden_dim = hidden_dim
        self.num_types = num_types

        ftype = config.ftype
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=ftype),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types, dtype=ftype),
        )

    @classmethod
    def from_type_system(
        cls,
        hidden_dim: int,
        type_system: TypeSystem,
        *,
        dropout: float = 0.0,
    ) -> TypeHead:
        """Build a head whose ``num_types`` matches a condensed type system.

        Args:
            hidden_dim: Input embedding dimension.
            type_system: Frozen :class:`~molrep.condensation.TypeSystem`.
            dropout: Dropout rate.

        Returns:
            A :class:`TypeHead` with ``num_types == type_system.n_types``.

        Raises:
            ValueError: If the type system is empty.
        """
        n = type_system.n_types
        if n < 1:
            raise ValueError("TypeHead.from_type_system requires n_types >= 1")
        return cls(hidden_dim=hidden_dim, num_types=n, dropout=dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute type logits from embeddings.

        Args:
            embeddings: Features ``(N, hidden_dim)``.

        Returns:
            Logits ``(N, num_types)``.
        """
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
        """Decode with confidence scores.

        Returns:
            ``(indices, confidence)`` from softmax max.
        """
        probs = torch.softmax(logits, dim=-1)
        confidence, indices = probs.max(dim=-1)
        return indices, confidence


class MultiTypeHead(nn.Module):
    """Named collection of :class:`TypeHead` modules (multi-system / multi-class).

    Does not use a ``method=`` switch — each head is an independent classifier
    keyed by interaction name (typically :class:`InteractionClass` value).

    Args:
        heads: Mapping from class key (str) to a configured :class:`TypeHead`.
    """

    def __init__(self, heads: Mapping[str, TypeHead]) -> None:
        super().__init__()
        if not heads:
            raise ValueError("MultiTypeHead requires at least one TypeHead")
        self.heads = nn.ModuleDict({str(k): v for k, v in heads.items()})

    @classmethod
    def from_type_systems(
        cls,
        hidden_dims: Mapping[str, int] | int,
        type_systems: Mapping[str, TypeSystem],
        *,
        dropout: float = 0.0,
    ) -> MultiTypeHead:
        """Build one head per condensed type system.

        Args:
            hidden_dims: Per-key hidden dim, or a single int shared by all.
            type_systems: Mapping of class key → :class:`TypeSystem`.
            dropout: Dropout rate for every head.

        Returns:
            A :class:`MultiTypeHead` covering every non-empty type system.
        """
        heads: dict[str, TypeHead] = {}
        for key, ts in type_systems.items():
            if ts.n_types < 1:
                continue
            dim = hidden_dims if isinstance(hidden_dims, int) else hidden_dims[key]
            heads[str(key)] = TypeHead.from_type_system(dim, ts, dropout=dropout)
        return cls(heads)

    @property
    def num_types(self) -> dict[str, int]:
        """Per-key ``num_types`` for each contained head."""
        return {k: h.num_types for k, h in self.heads.items()}

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute logits for each provided embedding key that has a head.

        Args:
            embeddings: Mapping of class key → features ``(N_k, D_k)``.

        Returns:
            Mapping of class key → logits ``(N_k, num_types_k)``.
        """
        out: dict[str, torch.Tensor] = {}
        for key, emb in embeddings.items():
            k = str(key)
            if k in self.heads:
                out[k] = self.heads[k](emb)
        return out

    def decode(self, logits: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Argmax decode per class key."""
        return {k: self.heads[k].decode(v) for k, v in logits.items() if k in self.heads}

    def decode_with_confidence(
        self,
        logits: Mapping[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Softmax-max decode with confidence per class key."""
        return {
            k: self.heads[k].decode_with_confidence(v)
            for k, v in logits.items()
            if k in self.heads
        }
