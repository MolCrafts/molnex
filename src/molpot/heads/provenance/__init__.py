"""Confidence, coverage, and parameter provenance surfaces.

Thin audit types for learnable Class-I force fields. No active-learning
loop lives here — consumers attach records / regimes externally.

Public API:
    CoverageRegime, SupportClassifier, ParameterProvenance, attach_provenance

Reference:
    Spec: learnable-classical-ff-09-provenance
"""

from molpot.heads.provenance.classifier import SupportClassifier
from molpot.heads.provenance.parameter import ParameterProvenance, attach_provenance
from molpot.heads.provenance.regime import CoverageRegime

__all__ = [
    "CoverageRegime",
    "ParameterProvenance",
    "SupportClassifier",
    "attach_provenance",
]
