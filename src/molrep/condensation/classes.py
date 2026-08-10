"""InteractionClass — discrete MM interaction kinds for condensation / SMARTS."""

from __future__ import annotations

from enum import Enum

__all__ = ["InteractionClass"]


class InteractionClass(str, Enum):
    """Chemical interaction classes that own a discrete type table.

    Values align with valence TensorDict namespaces where applicable
    (``bonds`` / ``angles`` / ``propers`` / ``impropers``) plus nonbonded
    classes for LJ and partial charges.

    Attributes:
        BOND: Pair bonded stretch terms.
        ANGLE: Three-body angle terms.
        PROPER: Four-body proper torsions.
        IMPROPER: Four-body impropers (center-first topology).
        LJ: Lennard-Jones (or equivalent) nonbonded types.
        CHARGE: Partial-charge classes (often not condensed aggressively).
    """

    BOND = "bond"
    ANGLE = "angle"
    PROPER = "proper"
    IMPROPER = "improper"
    LJ = "lj"
    CHARGE = "charge"
