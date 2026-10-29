"""MACE package — configuration, energy/force potentials, checkpoints.

Public construction is a product over its typed spec::

    from molzoo.mace import MACEMatpesSpec, MaceMatpes, MaceOmol, MacePolar

    model = MaceMatpes(MACEMatpesSpec(atomic_numbers=..., atomic_energies=...), dtype=torch.float64)
    model = MaceMatpes.load(path, wrt=("atoms.pos",), dtype=torch.float64)

Every default lives on the spec. Official archives are read and remapped by
``MaceXX.load``; weights must already be the official cuEq-converted ones —
nothing here converts the upstream basis. The encoder and its blocks are
:mod:`molrep.encoders.mace`.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from molzoo.mace.polar import MacePolar
from molzoo.mace.potential import MaceLes, MaceMatpes, MaceOff, MaceOmol, MacePotential
from molzoo.mace.spec import (
    MACELesSpec,
    MACEMatpesSpec,
    MaceOfficialConfig,
    MACEOffSpec,
    MACEOMolSpec,
    MACEPolarSpec,
    MACESpec,
)

__all__ = [
    "MACELesSpec",
    "MACEMatpesSpec",
    "MACEOMolSpec",
    "MACEOffSpec",
    "MACEPolarSpec",
    "MACESpec",
    "MaceLes",
    "MaceMatpes",
    "MaceOff",
    "MaceOfficialConfig",
    "MaceOmol",
    "MacePolar",
    "MacePotential",
]
