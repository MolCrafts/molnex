"""PiNet package — encoder, potential, and property heads.

Layout (industrial split):

* :mod:`molzoo.pinet.spec` — config only
* :mod:`molzoo.pinet.geometry` — PBC-safe edge geometry
* :mod:`molzoo.pinet.encoder` — feature encoder (molrep GC blocks)
* :mod:`molzoo.pinet.potential` — energy + functorch forces
* :mod:`molzoo.pinet.properties` — dipole / polarizability façades

Public import surface is stable::

    from molzoo.pinet import PiNet, PiNetPotential, PiNetSpec
"""

from .encoder import PiNet
from .geometry import compute_d5, edge_bond_diff
from .potential import PiNetPotential
from .properties import PiNetDipole, PiNetPolarizability, pool_layer
from .spec import PiNetSpec

# Back-compat private aliases used by older call sites / docs.
_compute_d5 = compute_d5
_edge_bond_diff = edge_bond_diff
_pool_layer = pool_layer

__all__ = [
    "PiNet",
    "PiNetSpec",
    "PiNetPotential",
    "PiNetDipole",
    "PiNetPolarizability",
    "compute_d5",
    "edge_bond_diff",
    "pool_layer",
]
