"""ChemEmbeddings — first-class continuous chemical-perception payload.

Holds per-atom and per-interaction feature tensors consumed by Classical MM
heads (``molpot``) and neural parameterizers. No energy, no forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class ChemEmbeddings:
    """Continuous chemical perception feature tensors.

    Attributes:
        atom: Per-atom features ``(N, D_a)``.
        bond: Per-bond features ``(N_bonds, D_b)``.
        angle: Per-angle features ``(N_angles, D_ang)``.
        proper: Per-proper-torsion features ``(N_propers, D_p)``.
        improper: Per-improper features ``(N_impropers, D_imp)``. Empty
            topology uses shape ``(0, D)`` rather than omitting the field.

    Notes:
        Counts must align with the valence namespaces on the batch
        (``bonds`` / ``angles`` / ``propers`` / ``impropers``).
    """

    atom: torch.Tensor
    bond: torch.Tensor
    angle: torch.Tensor
    proper: torch.Tensor
    improper: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return a plain mapping of the five feature tensors.

        Returns:
            Dict with keys ``atom``, ``bond``, ``angle``, ``proper``,
            ``improper``.
        """
        return {
            "atom": self.atom,
            "bond": self.bond,
            "angle": self.angle,
            "proper": self.proper,
            "improper": self.improper,
        }

    def interaction_dict(self) -> dict[str, torch.Tensor]:
        """Feature mapping keyed for ClassicalMMComposer.

        Returns:
            Dict with keys ``atoms``, ``bonds``, ``angles``, ``propers``,
            ``impropers`` (plural namespace names used by MM heads).
        """
        return {
            "atoms": self.atom,
            "bonds": self.bond,
            "angles": self.angle,
            "propers": self.proper,
            "impropers": self.improper,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, torch.Tensor]) -> ChemEmbeddings:
        """Build from a fixed-key mapping.

        Args:
            data: Mapping with ``atom``/``bond``/``angle``/``proper``/
                ``improper`` keys (or plural ``atoms``/``bonds``/…).

        Returns:
            A :class:`ChemEmbeddings` instance.
        """

        def _get(*names: str) -> torch.Tensor:
            for name in names:
                if name in data:
                    return data[name]
            raise KeyError(f"missing ChemEmbeddings field among {names}")

        return cls(
            atom=_get("atom", "atoms"),
            bond=_get("bond", "bonds"),
            angle=_get("angle", "angles"),
            proper=_get("proper", "propers"),
            improper=_get("improper", "impropers"),
        )
