"""PiNet2 — spec, feature body, energy model and property products.

* :mod:`molzoo.pinet.spec` — :class:`PiNetSpec`, torch-free validation
* :mod:`molzoo.pinet.encoder` — :class:`PiNetEncoder`, the feature body
* :mod:`molzoo.pinet.potential` — :class:`PiNetPotential` (energy, ``wrt=``)
* :mod:`molzoo.pinet.properties` — :class:`PiNetDipole`, :class:`PiNetPolarizability`

* :mod:`molzoo.pinet.layers` — the PiNN layers (PI/IP, Cartesian PIX/Scale/Dot,
  ``GCBlock``, ``ResUpdate``, ``OutLayer``)

Generic pieces come from :mod:`molrep` (``embedding.mlp.FFLayer``, cutoffs,
radial bases, ``embedding.geometry``).
"""

from .encoder import PiNetEncoder, PiNetLayer
from .potential import PiNetPotential
from .properties import PiNetDipole, PiNetPolarizability
from .spec import PiNetSpec

__all__ = [
    "PiNetDipole",
    "PiNetEncoder",
    "PiNetLayer",
    "PiNetPolarizability",
    "PiNetPotential",
    "PiNetSpec",
]
