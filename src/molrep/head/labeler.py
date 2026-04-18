"""Labeler protocol and implementations."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Labeler(Protocol):
    """Protocol for atom type labelers."""

    num_types: int
    type_map: dict[int, str]

    def label(self, z: torch.Tensor) -> torch.Tensor:
        """Generate type labels for atoms in batch."""
        ...


class ProxyLabeler:
    """Simple proxy labeler using atomic numbers."""

    _ELEMENT_TYPES = {
        1: 0,  # H
        6: 1,  # C
        7: 2,  # N
        8: 3,  # O
        9: 4,  # F
        15: 5,  # P
        16: 6,  # S
        17: 7,  # Cl
        35: 8,  # Br
        53: 9,  # I
    }

    def __init__(self):
        self._type_map = {
            0: "H",
            1: "C",
            2: "N",
            3: "O",
            4: "F",
            5: "P",
            6: "S",
            7: "Cl",
            8: "Br",
            9: "I",
            10: "OTHER",
        }
        self._num_types = len(self._type_map)

    @property
    def num_types(self) -> int:
        return self._num_types

    @property
    def type_map(self) -> dict[int, str]:
        return self._type_map

    def label(self, z: torch.Tensor) -> torch.Tensor:
        """Generate proxy labels based on atomic numbers."""
        labels = torch.full_like(z, 10)  # Default to OTHER

        for element, type_idx in self._ELEMENT_TYPES.items():
            labels[z == element] = type_idx

        return labels
