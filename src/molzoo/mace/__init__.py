"""MACE sub-package: the research encoder plus the foundation-model core layer.

Transitional layout (mace-subpackage-restructure-02-core). ``molzoo/mace.py``
became :mod:`molzoo.mace.research`, and the configuration-driven core layer
lands beside it:

* :mod:`molzoo.mace.spec` — torch-free configuration family
* :mod:`molzoo.mace.geometry` — MACE edge displacement / length
* :mod:`molzoo.mace.encoder` — configuration-driven backbone
* :mod:`molzoo.mace.research` — the research encoder (former ``molzoo/mace.py``)

Every legacy ``from molzoo.mace import …`` name still resolves; the final
re-export surface is settled in 06-wire.

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
    from molzoo.mace.research import MACE, MACEResearchSpec

#: Lazily exported (PEP 562): each of these pulls in cuEquivariance.
_LAZY = {
    "MACE": "molzoo.mace.research",
    "MACEResearchSpec": "molzoo.mace.research",
    "EmbeddingBlock": "molrep.embedding.mace",
    "EmbeddingSpec": "molrep.embedding.mace",
    "InteractionBlock": "molrep.interaction.mace.block",
    "InteractionSpec": "molrep.interaction.mace.block",
}

__all__ = [
    "EmbeddingBlock",
    "EmbeddingSpec",
    "InteractionBlock",
    "InteractionSpec",
    "MACE",
    "MACEMatpesSpec",
    "MACEOMolSpec",
    "MACEResearchSpec",
    "MACESpec",
]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
