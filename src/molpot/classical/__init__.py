"""Learnable Class-I force field: LAMMPS styles and the heads that fill their parameters.

This package sits beside :mod:`molpot.heads` and :mod:`molpot.potentials`.
:mod:`molpot.heads` is generic prediction. :mod:`molpot.potentials` prices a
style. The classes here are the styles a model carries.

The styles (``BondHarmonic``, ``AngleHarmonic``, ``DihedralPeriodic``,
``ImproperCvff``, ``LJCut``, ``PairCoulCut``) carry the same class names as
:mod:`molpot.potentials`, which prices the same styles. The bonded ones are
the published model's heads (:class:`SymmetricHead`, :class:`TanhWindow`); the pair
ones are Rondo's (EEM charges, LJ around GAFF2). Every one reads
:class:`~molrep.encoders.features.EncoderFeatures`.
"""

from molpot.classical.heads import (
    K_BOUNDS,
    LENGTH_BOUNDS,
    ExpAround,
    LJHead,
    SymmetricHead,
    TanhWindow,
)
from molpot.classical.tables import ParameterMap
from molpot.classical.terms import (
    TERMS,
    AngleHarmonic,
    BondChargeCorrection,
    BondHarmonic,
    DihedralPeriodic,
    Hypervalent,
    ImproperCvff,
    LJCut,
    LJCutDirect,
    PairCoulCut,
    Term,
)

__all__ = [
    "ParameterMap",
    "AngleHarmonic",
    "BondChargeCorrection",
    "BondHarmonic",
    "LJCutDirect",
    "TanhWindow",
    "DihedralPeriodic",
    "ExpAround",
    "Hypervalent",
    "ImproperCvff",
    "K_BOUNDS",
    "LENGTH_BOUNDS",
    "LJCut",
    "LJHead",
    "PairCoulCut",
    "SymmetricHead",
    "TERMS",
    "Term",
]
