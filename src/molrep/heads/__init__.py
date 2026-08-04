"""Representation-side heads (NOT physical energy / multipole heads).

``molrep.heads`` maps features → generic task projections (classification,
scalar regression demos). Physical quantities (energy, charge, multipoles,
scale-shift) live in :mod:`molpot.heads` — do not import those from here.
"""

from .labeler import Labeler, ProxyLabeler
from .scalar import ScalarHead
from .type import TypeHead

__all__ = ["TypeHead", "Labeler", "ProxyLabeler", "ScalarHead"]
