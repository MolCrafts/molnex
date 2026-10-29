"""Irreps-string helpers shared by the equivariant encoders.

The message-passing blocks themselves live with the model that owns them
(``molrep.encoders.mace.layers``, ``molzoo.allegro.layers``).
"""

from .product import irreps_from_l_max, sh_irreps_from_l_max

__all__ = [
    "irreps_from_l_max",
    "sh_irreps_from_l_max",
]
