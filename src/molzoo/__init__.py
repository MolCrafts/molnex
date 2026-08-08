"""MolZoo: molecular model zoo.

This package provides encoder architectures and potential models.
"""

from typing import TYPE_CHECKING

from molzoo.allegro import Allegro, AllegroSpec
from molzoo.mace import MACE, MACESpec
from molzoo.pinet import PiNet, PiNetSpec

if TYPE_CHECKING:
    from molzoo.mace_matpes import MACEMatpes, load_matpes_state_dict
    from molzoo.mace_omol import MACEOMol, load_omol_state_dict

# Lazily exported (PEP 562): the foundation models pull in the full MACE
# cuEquivariance stack + weight-conversion helpers, which not every molzoo
# consumer needs.
_LAZY = {
    "MACEOMol": "molzoo.mace_omol",
    "load_omol_state_dict": "molzoo.mace_omol",
    "MACEMatpes": "molzoo.mace_matpes",
    "load_matpes_state_dict": "molzoo.mace_matpes",
}

__all__ = [
    "Allegro",
    "AllegroSpec",
    "MACE",
    "MACESpec",
    "MACEMatpes",
    "MACEOMol",
    "PiNet",
    "PiNetSpec",
    "load_matpes_state_dict",
    "load_omol_state_dict",
]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
