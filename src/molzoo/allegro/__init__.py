"""Allegro — spec, feature body, energy model and official-package loading.

* :mod:`molzoo.allegro.spec` — :class:`AllegroSpec`, :class:`NequipConfig`,
  torch-free validation
* :mod:`molzoo.allegro.encoder` — :class:`AllegroEncoder`, the feature body
* :mod:`molzoo.allegro.potential` — :class:`AllegroPotential` (``energy``, ``wrt=``, ``.load``)
* :mod:`molzoo.allegro.checkpoint` — official key table
* :mod:`molzoo.allegro.utils` — ``.nequip.zip`` reader (no nequip / allegro import)

* :mod:`molzoo.allegro.layers` — ``ProductTypeEmbedding``, ``WeightedChannels``,
  ``Contracter``

Generic pieces come from :mod:`molrep` (``embedding.radial.SincBesselRBF``,
``embedding.mlp.ScalarMLPFunction``, cutoffs, spherical harmonics, geometry).
"""

from .encoder import AllegroEncoder
from .potential import AllegroPotential
from .spec import AllegroSpec, NequipConfig

__all__ = ["AllegroEncoder", "AllegroPotential", "AllegroSpec", "NequipConfig"]
