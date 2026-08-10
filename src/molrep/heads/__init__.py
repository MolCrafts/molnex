"""Representation-side heads (NOT physical energy / multipole heads).

``molrep.heads`` maps features → generic task projections (classification,
scalar regression demos). Physical quantities (energy, charge, multipoles,
scale-shift) live in :mod:`molpot.heads` — do not import those from here.
"""

from .labeler import Labeler, ProxyLabeler, TypeSystemLabeler
from .scalar import ScalarHead
from .type import MultiTypeHead, TypeHead

__all__ = [
    "TypeHead",
    "MultiTypeHead",
    "Labeler",
    "ProxyLabeler",
    "TypeSystemLabeler",
    "ScalarHead",
]
