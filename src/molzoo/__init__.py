"""MolZoo: molecular model zoo.

This package provides encoder architectures and potential models.
"""

from typing import TYPE_CHECKING

from molzoo.allegro import Allegro, AllegroSpec
from molzoo.mace import MACE, MACESpec
from molzoo.pinet import PiNet, PiNetSpec

if TYPE_CHECKING:
    from molzoo.mace_omol import MACEOMol, load_omol_state_dict

# Lazily exported (PEP 562): MACEOMol pulls in the full MACE-OMOL cuEquivariance
# stack + weight-conversion helpers, which not every molzoo consumer needs.
_LAZY = {"MACEOMol", "load_omol_state_dict"}

__all__ = [
    "Allegro",
    "AllegroSpec",
    "MACE",
    "MACESpec",
    "MACEOMol",
    "PiNet",
    "PiNetSpec",
    "load_omol_state_dict",
]


def __getattr__(name: str):
    if name in _LAZY:
        from molzoo import mace_omol

        return getattr(mace_omol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
