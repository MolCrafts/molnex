"""Allegro — the energy model and official-package loading.

* :mod:`molzoo.allegro.spec` — :class:`AllegroSpec` (the encoder's fields,
  :class:`~molrep.encoders.allegro.spec.AllegroEncoderSpec`, and the readout),
  :class:`NequipConfig`, torch-free validation
* :mod:`molzoo.allegro.potential` — :class:`AllegroPotential` (``energy``, ``wrt=``, ``.load``)
* :mod:`molzoo.allegro.checkpoint` — official key table
* :mod:`molzoo.allegro.utils` — ``.nequip.zip`` reader (no nequip / allegro import)

The encoder and its blocks are :mod:`molrep.encoders.allegro`.
"""

from .potential import AllegroPotential
from .spec import AllegroSpec, NequipConfig

__all__ = ["AllegroPotential", "AllegroSpec", "NequipConfig"]
