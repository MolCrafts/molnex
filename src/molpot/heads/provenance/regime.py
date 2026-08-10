"""Coverage regimes for chemical-space support of Class-I parameters.

Reference:
    Spec: learnable-classical-ff-09-provenance
"""

from __future__ import annotations

from enum import Enum

__all__ = ["CoverageRegime"]


class CoverageRegime(Enum):
    """Where a predicted parameter sits relative to chemical support.

    Members:
        IN_SUPPORT: Type id / embedding inside the support index **and**
            confidence at or above ``conf_in``.
        NEAR_SUPPORT: Inside support with mid confidence
            (``conf_near <= conf < conf_in``), or within a configured
            distance margin when using continuous banks.
        EXTRAPOLATING: Finite prediction outside support (confidence still
            usable: ``conf >= conf_near``) — treat as extrapolation.
        UNKNOWN: Below the confidence floor (``conf < conf_near``), missing
            topology, or otherwise unusable for a regime call.
    """

    IN_SUPPORT = "in_support"
    NEAR_SUPPORT = "near_support"
    EXTRAPOLATING = "extrapolating"
    UNKNOWN = "unknown"
