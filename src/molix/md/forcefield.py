"""Re-export :class:`molpy.ForceField` plus the structural contract MD binds to.

``from molix.md import ForceField`` and ``from molpy import ForceField`` are
the same object. Evaluation is ``ff.to_potentials()`` then ``calc_energy`` /
``calc_forces`` on a working ``TensorDict``, wired by :class:`~molix.md.driver.MD`.

:class:`MDForceField` is what the driver actually requires: a **nullary**
``to_potentials`` and a ``units`` style label. ``molpy.ForceField``'s own
``to_potentials`` takes the frame to enumerate topology over, so the bare
container does not satisfy MD — molnex's :class:`molpot.forcefield.ForceField`
subclass does, because a parameterized field carries its own realization and
defaults ``frame``. Stating the difference as a protocol (rather than
annotating the molpy base) is what keeps ``MD.set_forcefield`` from admitting
an object whose ``to_potentials()`` call would fail at wire time; the one-way
package rule (``molix`` must not import ``molpot``) rules out naming the
subclass here directly.

:class:`~molix.schema.CellBound` and :class:`~molix.schema.CoreCompilable` are
re-exported here for MD-side readers; both are declared in
:mod:`molix.schema` (which molpot already imports), and that module's class
docstrings are the sole home of their contracts.
"""

from typing import Protocol, runtime_checkable

from molpy import ForceField

from molix.schema import CellBound, CoreCompilable

__all__ = ["CellBound", "CoreCompilable", "ForceField", "MDForceField"]


@runtime_checkable
class MDForceField(Protocol):
    """A force field :class:`~molix.md.driver.MD` can evaluate.

    Attributes:
        units: LAMMPS-style unit label (``"real"`` / ``"metal"``), resolved by
            :func:`molix.units.preset_energy_to_md`.
    """

    units: str

    def to_potentials(self) -> object:
        """Realized torch potentials — callable with no frame argument."""
        ...
