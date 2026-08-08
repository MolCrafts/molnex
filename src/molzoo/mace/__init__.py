"""MACE sub-package: the research encoder plus the foundation-model core layer.

MACE is a message-passing neural network for the potential energy of a set of
atoms (Batatia et al., NeurIPS 2022, https://arxiv.org/abs/2206.07697). It is
*equivariant*: rotate the atoms and its internal directional features rotate
with them, so the predicted energy is unchanged and the predicted forces rotate
correctly. A *foundation* variant is one shipped with weights already fitted on
a large, chemically broad dataset. This package holds molnex's native
implementation of both sides — a configurable research encoder, and the two
foundation variants (MatPES, OMol) behind
:class:`~molzoo.mace.potential.MACEPotential`.

Transitional layout, mid-way through the ``mace-subpackage-restructure`` spec
chain (``.claude/specs/INDEX.md``). ``molzoo/mace.py`` became
:mod:`molzoo.mace.research`, and the configuration-driven core layer lands
beside it:

* :mod:`molzoo.mace.spec` — torch-free configuration family
* :mod:`molzoo.mace.geometry` — MACE edge displacement / length
* :mod:`molzoo.mace.encoder` — configuration-driven backbone
* :mod:`molzoo.mace.potential` — :class:`~molzoo.mace.potential.MACEPotential`,
  the energy/force host on top of that backbone: per-graph energy ``(B,)`` in
  eV and per-atom forces ``(N, 3)`` in eV/Å on the post-collate batch
* :mod:`molzoo.mace.checkpoint` — official-checkpoint key dialect:
  :class:`~molzoo.mace.checkpoint.CheckpointRemap`, the two family tables
  (:data:`~molzoo.mace.checkpoint.MATPES_KEY_REMAP` /
  :data:`~molzoo.mace.checkpoint.OMOL_KEY_REMAP`) and their preset instances —
  :data:`~molzoo.mace.checkpoint.MATPES_REMAP`, which refuses a checkpoint key
  with no home, and :data:`~molzoo.mace.checkpoint.OMOL_REMAP`, which returns
  the unhoused keys instead. Both are strict about unfilled parameters.
* :mod:`molzoo.mace.research` — the research encoder (former ``molzoo/mace.py``)

Every legacy ``from molzoo.mace import …`` name still resolves; the final
re-export surface is settled by
``.claude/specs/mace-subpackage-restructure-06-wire.md``.

``MACESpec`` changed meaning here, deliberately: it is now the shared
foundation-variant configuration base (:class:`molzoo.mace.spec.MACESpec`).
The research encoder's own configuration is
:class:`molzoo.mace.research.MACEResearchSpec`.

The configuration models are imported eagerly — they are torch-free, and
reading a config should not cost the cuEquivariance stack. Everything else is
loaded lazily on first attribute access (PEP 562), mirroring
``molzoo/__init__.py``.
"""

from typing import TYPE_CHECKING

from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec

if TYPE_CHECKING:
    from molrep.embedding.mace import EmbeddingBlock, EmbeddingSpec
    from molrep.interaction.mace.block import InteractionBlock, InteractionSpec
    from molzoo.mace.checkpoint import (
        MATPES_KEY_REMAP,
        MATPES_REMAP,
        OMOL_KEY_REMAP,
        OMOL_REMAP,
        CheckpointRemap,
    )
    from molzoo.mace.potential import MACEPotential
    from molzoo.mace.research import MACE, MACEResearchSpec

#: Lazily exported (PEP 562): each of these pulls in cuEquivariance (or, for
#: the checkpoint names, torch).
_LAZY = {
    "MACE": "molzoo.mace.research",
    "MACEPotential": "molzoo.mace.potential",
    "MACEResearchSpec": "molzoo.mace.research",
    "CheckpointRemap": "molzoo.mace.checkpoint",
    "MATPES_KEY_REMAP": "molzoo.mace.checkpoint",
    "MATPES_REMAP": "molzoo.mace.checkpoint",
    "OMOL_KEY_REMAP": "molzoo.mace.checkpoint",
    "OMOL_REMAP": "molzoo.mace.checkpoint",
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
    "MACEPotential",
    "MACEResearchSpec",
    "MACESpec",
    "MATPES_KEY_REMAP",
    "MATPES_REMAP",
    "OMOL_KEY_REMAP",
    "OMOL_REMAP",
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
