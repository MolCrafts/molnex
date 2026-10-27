"""molnex's equivariant operators: cuEquivariance's descriptors on molnex's own kernels.

Every module here evaluates one contraction read off a ``molrep.eq``
descriptor — the channelwise product and symmetric contraction of MACE, the
spherical harmonics, the Allegro contracter and weighted channels — with
``libmolnex_equivariance`` on CUDA tensors (:mod:`.backend`) and
:mod:`.reference` on CPU tensors, and :class:`Linear` as one GEMM per irrep
block. No cuEquivariance kernel runs anywhere in molnex.
"""

from .backend import EdgeSchedule
from .channelwise import ChannelWiseTensorProduct
from .contracter import Contracter
from .linear import Linear
from .spherical_harmonics import SphericalHarmonics
from .symmetric_contraction import SymmetricContraction
from .weighted_channels import WeightedChannels

__all__ = [
    "ChannelWiseTensorProduct",
    "Contracter",
    "EdgeSchedule",
    "Linear",
    "SphericalHarmonics",
    "SymmetricContraction",
    "WeightedChannels",
]
