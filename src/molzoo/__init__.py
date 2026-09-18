"""MolZoo: molecular model zoo — encoder architectures and foundation models.

An *encoder* here maps a batch of atoms (their chemical element numbers and
Cartesian positions) to a feature vector per atom, which a downstream head
turns into an energy or another property; a *foundation* model is one shipped
with weights already fitted on a large, chemically broad dataset, used
unchanged rather than trained by the caller.

**Every symbol in :data:`__all__` is exported lazily** — this module imports no
model module at import time, and a name is resolved only when someone asks for
it, through the module-level ``__getattr__`` hook of PEP 562.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molzoo.allegro import Allegro, AllegroSpec
    from molzoo.chem import ChemPerception, ChemPerceptionSpec
    from molzoo.mace.polar import MacePolar
    from molzoo.mace.potential import MaceMatpes, MaceOmol, MacePotential
    from molzoo.mace.research import MACE
    from molzoo.mace.spec import MACESpec
    from molzoo.pinet import PiNet, PiNetSpec
    from molzoo.rondo import Rondo, RondoForceField, RondoPotential, RondoSpec

_LAZY = {
    "Allegro": "molzoo.allegro",
    "AllegroSpec": "molzoo.allegro",
    "ChemPerception": "molzoo.chem",
    "ChemPerceptionSpec": "molzoo.chem",
    "MACE": "molzoo.mace.research",
    "MACESpec": "molzoo.mace.spec",
    "MaceMatpes": "molzoo.mace.potential",
    "MaceOmol": "molzoo.mace.potential",
    "MacePolar": "molzoo.mace.polar",
    "MacePotential": "molzoo.mace.potential",
    "PiNet": "molzoo.pinet",
    "PiNetSpec": "molzoo.pinet",
    "Rondo": "molzoo.rondo",
    "RondoForceField": "molzoo.rondo",
    "RondoPotential": "molzoo.rondo",
    "RondoSpec": "molzoo.rondo",
}

__all__ = [
    "Allegro",
    "AllegroSpec",
    "ChemPerception",
    "ChemPerceptionSpec",
    "MACE",
    "MACESpec",
    "MaceMatpes",
    "MaceOmol",
    "MacePolar",
    "MacePotential",
    "PiNet",
    "PiNetSpec",
    "Rondo",
    "RondoForceField",
    "RondoPotential",
    "RondoSpec",
]


def __getattr__(name: str):
    """Import a lazily exported name on first attribute access (PEP 562).

    Args:
        name: Attribute requested on the ``molzoo`` module.

    Returns:
        The object named ``name``, imported from its module in :data:`_LAZY`.

    Raises:
        AttributeError: If ``name`` is not a lazily exported symbol — a typo
            must fail the way a missing module attribute does, not as an
            ``ImportError`` from somewhere inside the package.
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
