"""Nonbonded 1–2 / 1–3 / 1–4 scaling factors for Class-I force fields.

Defaults follow AMBER / GAFF / SMIRNOFF Class-I conventions:

| relation | scale_q | scale_lj |
|----------|---------|----------|
| 1–2      | 0       | 0        |
| 1–3      | 0       | 0        |
| 1–4      | 5/6     | 0.5      |

References:
    OpenMM User Guide §19 / SMIRNOFF nonbonded section
    Cornell et al., JACS 1995 DOI 10.1021/ja00124a002 (AMBER)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NonbondedScaling"]


@dataclass
class NonbondedScaling:
    """Pair-exclusion scale factors for Coulomb and LJ interactions.

    Attributes:
        scale_q_12: Coulomb scale for 1–2 (bonded) pairs. Default ``0``.
        scale_q_13: Coulomb scale for 1–3 (angle) pairs. Default ``0``.
        scale_q_14: Coulomb scale for 1–4 (proper torsion) pairs. Default ``5/6``.
        scale_lj_12: LJ scale for 1–2 pairs. Default ``0``.
        scale_lj_13: LJ scale for 1–3 pairs. Default ``0``.
        scale_lj_14: LJ scale for 1–4 pairs. Default ``0.5``.
    """

    scale_q_12: float = 0.0
    scale_q_13: float = 0.0
    scale_q_14: float = 5.0 / 6.0
    scale_lj_12: float = 0.0
    scale_lj_13: float = 0.0
    scale_lj_14: float = 0.5

    def scales_for(self, relation: str) -> tuple[float, float]:
        """Return ``(scale_q, scale_lj)`` for a bonded-path relation.

        Args:
            relation: One of ``"1-2"``, ``"1-3"``, ``"1-4"`` (also accepts
                underscore forms ``"1_2"`` …).

        Returns:
            Coulomb and LJ scale factors for that relation.

        Raises:
            ValueError: Unknown relation label.
        """
        key = relation.strip().replace("_", "-")
        table = {
            "1-2": (self.scale_q_12, self.scale_lj_12),
            "1-3": (self.scale_q_13, self.scale_lj_13),
            "1-4": (self.scale_q_14, self.scale_lj_14),
        }
        if key not in table:
            raise ValueError(
                f"Unknown nonbonded relation {relation!r}; expected one of {sorted(table)}"
            )
        return table[key]
