"""Translation case matrix for Potential IR → backend force forms.

Four explicit outcomes — never a silent drop of an IR term.

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-08-ff-export
"""

from __future__ import annotations

from enum import Enum

__all__ = ["TranslationCase"]


class TranslationCase(Enum):
    """How an IR interaction bag maps into a backend force form.

    Attributes:
        DIRECT_UNIT_SCALE: Same functional form; only unit conversion
            (e.g. harmonic bond/angle with matching ``½ k x²``).
        FORM_REPARAMETERIZE: Same physics, different parameter convention
            (e.g. torsion ``k`` vs ``k/2``, idivf absorption, LJ ε/σ vs A/B).
        DECOMPOSE: One IR bag expands to multiple backend force parameters
            (e.g. multi-term proper → multiple PeriodicTorsion rows).
        UNSUPPORTED: No faithful mapping; raise :class:`UnsupportedTermError`.
    """

    DIRECT_UNIT_SCALE = "direct_unit_scale"
    FORM_REPARAMETERIZE = "form_reparameterize"
    DECOMPOSE = "decompose"
    UNSUPPORTED = "unsupported"
