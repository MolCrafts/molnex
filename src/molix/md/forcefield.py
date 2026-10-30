"""Re-export :class:`molpy.ff.forcefield.ForceField` plus the structural contract MD binds to.

``from molix.md import ForceField`` and ``from molpy import ForceField`` are
the same object. Evaluation is ``ff.to_potentials()`` composed under a
:class:`molix.autograd.Derivative` over a working ``TensorDict``, wired by
:class:`~molix.md.driver.MD`.

:class:`MDForceField` is what the driver actually requires: a **nullary**
``to_potentials`` and a ``units`` style label. ``molpy.ff.forcefield.ForceField``'s own
``to_potentials`` takes the frame to enumerate topology over, so the bare
container does not satisfy MD — molnex's :class:`molpot.forcefield.ForceField`
subclass does, because a parameterized field carries its own realization and
defaults ``frame``. Stating the difference as a protocol (rather than
annotating the molpy base) is what keeps ``MD.set_forcefield`` from admitting
an object whose ``to_potentials()`` call would fail at wire time; the one-way
package rule (``molix`` must not import ``molpot``) rules out naming the
subclass here directly.

:class:`~molix.schema.CellBound` is re-exported here for MD-side readers; it
is declared in :mod:`molix.schema` (which molpot already imports), and that
module's class docstring is the sole home of its contract.
"""

from typing import Protocol, runtime_checkable

import torch
from molpy.ff.forcefield import ForceField

from molix.autograd import Potential
from molix.schema import CellBound

__all__ = ["CellBound", "ForceField", "MDForceField", "RetypableForceField"]


@runtime_checkable
class MDForceField(Protocol):
    """A force field :class:`~molix.md.driver.MD` can evaluate.

    Attributes:
        units: LAMMPS-style unit label (``"real"`` / ``"metal"``), resolved by
            :func:`molix.units.preset_energy_to_md`.
    """

    units: str

    def to_potentials(self) -> Potential:
        """Realized torch potentials — the :class:`~molix.autograd.Potential` MD evaluates."""
        ...


@runtime_checkable
class RetypableForceField(MDForceField, Protocol):
    """An :class:`MDForceField` that restates its parameter tables at another dtype.

    ``MD(potential_dtype=)`` needs one (``molpot.forcefield.ForceField`` is).
    """

    def to_dtype(self, dtype: torch.dtype) -> "RetypableForceField":
        """The same field with its tables at *dtype*."""
        ...
