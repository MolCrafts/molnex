"""MACE package — configuration, encoder, energy/force potential, checkpoints.

MACE is a message-passing neural network for the potential energy of a set of
atoms (Batatia et al., NeurIPS 2022, https://arxiv.org/abs/2206.07697). It is
*equivariant*: rotate the atoms and its internal directional features rotate
with them, so the predicted energy is unchanged and the predicted forces rotate
correctly. A *foundation* variant is one shipped with weights already fitted on
a large, chemically broad dataset. This package holds molnex's native
implementation of both sides — a configurable research encoder, and the two
foundation variants (MatPES, OMol) behind
:class:`~molzoo.mace.potential.MACEPotential`.

Layout (industrial split) — one module, one responsibility:

* :mod:`molzoo.mace.spec` — configuration and variant presets (torch-free)
* :mod:`molzoo.mace.geometry` — edge displacement vectors and their lengths,
  correct under periodic boundary conditions (PBC — the simulation cell is
  taken to repeat for ever in every direction, so an atom's nearest neighbour
  may be the copy of an atom across a cell face)
* :mod:`molzoo.mace.encoder` — feature encoder built from molrep blocks
* :mod:`molzoo.mace.potential` — per-graph energy ``(B,)`` in eV for the ``B``
  graphs of a batch (one graph = one molecule or one periodic cell) and
  per-atom forces ``(N, 3)`` in eV/Å for its ``N`` atoms, on the post-collate
  batch (the nested ``TensorDict`` schema described in CLAUDE.md)
* :mod:`molzoo.mace.checkpoint` — official-weight key remap
  (:class:`~molzoo.mace.checkpoint.CheckpointRemap`, the two family tables and
  their presets :data:`~molzoo.mace.checkpoint.MATPES_REMAP` /
  :data:`~molzoo.mace.checkpoint.OMOL_REMAP`; both strict about unfilled
  parameters, and they differ only in what an unhoused checkpoint key means)
* :mod:`molzoo.mace.variants` — named foundation models
  (:class:`~molzoo.mace.variants.MACEMatpes` /
  :class:`~molzoo.mace.variants.MACEOMol`), kept for the call sites that
  construct them by keyword
* :mod:`molzoo.mace.research` — the freely configurable research encoder

Four names in :data:`__all__` are **not** defined here: ``EmbeddingBlock`` /
``EmbeddingSpec`` come from :mod:`molrep.embedding.mace` and ``InteractionBlock``
/ ``InteractionSpec`` from :mod:`molrep.interaction.mace.block`, where they were
promoted so molrep owns the reusable blocks. They are re-exported unchanged —
the same class objects, pinned by
``tests/test_molzoo/test_imports.py::TestMolzooMaceReexports`` — so existing
``from molzoo.mace import EmbeddingBlock`` imports keep resolving.

Public import surface is stable::

    from molzoo.mace import MACE, MACEMatpesSpec, MACEPotential

``MACESpec`` names the shared foundation-variant configuration base
(:class:`molzoo.mace.spec.MACESpec`); the research encoder's own configuration
is :class:`molzoo.mace.research.MACEResearchSpec`.

The configuration models are imported eagerly — they are torch-free, and
reading a config should not cost the cuEquivariance stack (NVIDIA's GPU library
for the equivariant tensor algebra MACE is built from). Everything else is
loaded **lazily**: the module object does not hold the attribute until someone
asks for it, at which point the module-level ``__getattr__`` hook of PEP 562
imports it. That is the same policy as ``molzoo/__init__.py``, which applies it
to every one of its names.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
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
    from molzoo.mace.variants import (
        MACEMatpes,
        MACEOMol,
        load_matpes_state_dict,
        load_omol_state_dict,
    )

#: Lazily exported (PEP 562): each of these pulls in cuEquivariance (or, for
#: the checkpoint names, torch).
_LAZY = {
    "MACE": "molzoo.mace.research",
    "MACEPotential": "molzoo.mace.potential",
    "MACEResearchSpec": "molzoo.mace.research",
    "MACEMatpes": "molzoo.mace.variants",
    "MACEOMol": "molzoo.mace.variants",
    "load_matpes_state_dict": "molzoo.mace.variants",
    "load_omol_state_dict": "molzoo.mace.variants",
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
    "MACEMatpes",
    "MACEMatpesSpec",
    "MACEOMol",
    "MACEOMolSpec",
    "MACEPotential",
    "MACEResearchSpec",
    "MACESpec",
    "MATPES_KEY_REMAP",
    "MATPES_REMAP",
    "OMOL_KEY_REMAP",
    "OMOL_REMAP",
    "load_matpes_state_dict",
    "load_omol_state_dict",
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

    Without this, ``dir(molzoo.mace)`` lists the module globals — which, under
    the lazy policy, is the torch-free config names and nothing else. It
    returns exactly :data:`__all__`, the same set ``from molzoo.mace import *``
    binds, so the two cannot drift apart. Same contract as ``molzoo/__init__``.

    Returns:
        The names in :data:`__all__`, sorted.
    """
    return sorted(__all__)
