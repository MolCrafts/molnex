"""MACE package — configuration, encoder, energy/force potential, checkpoints.

Public construction is the product class with a flat ``__init__``::

    from molzoo.mace import MaceMatpes, MaceOmol, MacePolar

    model = MaceMatpes(atomic_numbers=..., atomic_energies=...)
    model = MaceMatpes.load(path, format="official")

The spec objects stay as internal validators assembled inside those
constructors. Official archives are remapped by ``MaceXX.load``; there is no
public ``*_REMAP`` constant and no convert CLI. ``e3nn2cueq`` lives in
:mod:`molzoo.mace.utils`.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from typing import TYPE_CHECKING

from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACEPolarSpec, MACESpec

if TYPE_CHECKING:
    from molrep.embedding.mace import EmbeddingBlock, EmbeddingSpec
    from molrep.interaction.mace.block import InteractionBlock, InteractionSpec
    from molzoo.mace.checkpoint import CheckpointRemap
    from molzoo.mace.polar import MacePolar
    from molzoo.mace.potential import MaceMatpes, MaceOmol, MacePotential
    from molzoo.mace.research import MACE, MACEResearchSpec
    from molzoo.mace.utils import e3nn2cueq

_LAZY = {
    "MACE": "molzoo.mace.research",
    "MACEResearchSpec": "molzoo.mace.research",
    "MacePotential": "molzoo.mace.potential",
    "MaceMatpes": "molzoo.mace.potential",
    "MaceOmol": "molzoo.mace.potential",
    "MacePolar": "molzoo.mace.polar",
    "CheckpointRemap": "molzoo.mace.checkpoint",
    "e3nn2cueq": "molzoo.mace.utils",
    "EmbeddingBlock": "molrep.embedding.mace",
    "EmbeddingSpec": "molrep.embedding.mace",
    "InteractionBlock": "molrep.interaction.mace.block",
    "InteractionSpec": "molrep.interaction.mace.block",
}

__all__ = [
    "CheckpointRemap",
    "EmbeddingBlock",
    "EmbeddingSpec",
    "InteractionBlock",
    "InteractionSpec",
    "MACE",
    "MACEMatpesSpec",
    "MACEOMolSpec",
    "MACEPolarSpec",
    "MACEResearchSpec",
    "MACESpec",
    "MaceMatpes",
    "MaceOmol",
    "MacePolar",
    "MacePotential",
    "e3nn2cueq",
]


def __getattr__(name: str):
    """Import a lazily exported name on first attribute access (PEP 562).

    Args:
        name: Attribute requested on the ``molzoo.mace`` module.

    Returns:
        The object named ``name``, imported from its module in :data:`_LAZY`.

    Raises:
        AttributeError: If ``name`` is not a lazily exported symbol — the same
            failure a missing module attribute would give.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Report the public surface, so completion survives the lazy table.

    Returns:
        The names in :data:`__all__`, sorted.
    """
    return sorted(__all__)
