"""MolZoo: molecular model zoo — encoder architectures and foundation models.

An *encoder* here maps a batch of atoms (their chemical element numbers and
Cartesian positions) to a feature vector per atom, which a downstream head
turns into an energy or another property; a *foundation* model is one shipped
with weights already fitted on a large, chemically broad dataset, used
unchanged rather than trained by the caller.

**Every symbol in :data:`__all__` is exported lazily** — this module imports no
model module at import time, and a name is resolved only when someone asks for
it, through the module-level ``__getattr__`` hook of PEP 562. The reason is
cost: resolving one of them pulls in the cuEquivariance stack (NVIDIA's GPU
library for the equivariant tensor algebra MACE and Allegro are built from),
seconds of import time and a large chunk of memory that a consumer reading a
config, a dataset or a checkpoint manifest should not have to pay. The one name
that would be cheap, ``MACESpec`` (its module, :mod:`molzoo.mace.spec`, imports
nothing but ``typing`` and pydantic), is in the table anyway: a single rule with
no exception list is what keeps this file and ``molzoo/mace/__init__.py``
honest.

Direct sub-module imports are unaffected and stay ordinary imports::

    import molzoo.mace                     # torch-free: the configurations only
    from molzoo.mace import MACE           # pulls the equivariance stack
    from molzoo import MACEMatpes          # the same, through the lazy table
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molzoo.allegro import Allegro, AllegroSpec
    from molzoo.mace.research import MACE
    from molzoo.mace.spec import MACESpec
    from molzoo.mace.variants import (
        MACEMatpes,
        MACEOMol,
        load_matpes_state_dict,
        load_omol_state_dict,
    )
    from molzoo.pinet import PiNet, PiNetSpec

#: Public name → the module the lazy import goes to. The one lazy-export
#: table: every entry is also in :data:`__all__`, and nothing outside it
#: resolves. For the MACE names the target is the **defining** module
#: (``molzoo.mace.research`` / ``.spec`` / ``.variants``) rather than the
#: ``molzoo.mace`` re-export surface, so resolving one here does not depend on
#: that package's own lazy table as well. The two PiNet names go to
#: ``molzoo.pinet``, whose ``__init__`` re-exports them eagerly and is
#: therefore already the defining surface for an importer.
_LAZY = {
    "Allegro": "molzoo.allegro",
    "AllegroSpec": "molzoo.allegro",
    "MACE": "molzoo.mace.research",
    "MACESpec": "molzoo.mace.spec",
    "MACEMatpes": "molzoo.mace.variants",
    "MACEOMol": "molzoo.mace.variants",
    "PiNet": "molzoo.pinet",
    "PiNetSpec": "molzoo.pinet",
    "load_matpes_state_dict": "molzoo.mace.variants",
    "load_omol_state_dict": "molzoo.mace.variants",
}

__all__ = [
    "Allegro",
    "AllegroSpec",
    "MACE",
    "MACEMatpes",
    "MACEOMol",
    "MACESpec",
    "PiNet",
    "PiNetSpec",
    "load_matpes_state_dict",
    "load_omol_state_dict",
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

    Without this, ``dir(molzoo)`` lists the module globals — which, under the
    lazy policy, is every name *except* the models. It returns exactly
    :data:`__all__`, the same set ``from molzoo import *`` binds, so the two
    cannot drift apart.

    Returns:
        The names in :data:`__all__`, sorted.
    """
    return sorted(__all__)
