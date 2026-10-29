"""A residual potential with its elemental dress added back, for export and absolute energies.

A network trained on ``graphs.energy_residual`` (label minus
:class:`~molix.data.AtomicDress`, subtracted by
:class:`~molix.data.EnergyResidual`) writes the residual. An engine that loads
the exported package has no Python seam to add ``c[Z]`` afterwards, and
:class:`~molix.deploy.AtomEnergyAdapter` differentiates the masked sum of
``atoms.energy`` rather than a restored total — so the dress has to be on the
atom energies inside the module that is packaged. :class:`DressedPotential` is
that module: the residual potential's ``energy``, then ``c[Z]`` onto every real
atom. ``c[Z]`` is constant in positions, so forces and virial are the residual
potential's.

Compose it for export only, never on a training step that also runs
:class:`~molix.data.EnergyResidual` (that counts ``c[Z]`` twice)::

    export_dress = copy.deepcopy(dress).to(device)  # never the loader's instance
    adapter = AtomEnergyAdapter(DressedPotential(potential, export_dress))

Reference:
    Shao et al. "PiNN: A Python Library for Building Atomic Neural Networks of
    Molecules and Materials" J. Chem. Inf. Model. 60, 1184 (2020).
    https://doi.org/10.1021/acs.jcim.9b00994 (``potential_model``: the dress is
    added only to the reported energy, never inside the trained network).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDictBase

from molix.autograd import Derivative, Potential
from molix.data.dress import AtomicDress
from molix.F import batch_add
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY

__all__ = ["DressedPotential"]


class DressedPotential(Potential):
    """``potential.energy`` plus ``c[Z]`` on each real atom's energy.

    The add runs in float64 and the result is stored in the inner potential's
    atom-energy dtype; a float32 store therefore rounds the absolute energy at
    the ulp of ``|c[Z]|``. ``state_dict`` is nested (``potential.*``,
    ``dress.e0``, ``dress.support``), so the exported graph reads the dress as
    the buffer ``dress.e0``; it is not the training checkpoint.

    Args:
        potential: The residual potential. Its ``energy`` is called, not its
            ``forward``.
        dress: The table to add, on the potential's device; a copy of the
            loader's, never the same instance.
        wrt: Variables :class:`~molix.autograd.Derivative` differentiates in
            :meth:`forward`. ``()`` under
            :class:`~molix.deploy.AtomEnergyAdapter`, which differentiates
            itself.

    Raises:
        ValueError: If *dress* has a nonzero intercept: a per-graph constant
            has no atom to carry it, so the atom-energy partition an engine
            sums would lose it.
    """

    def __init__(
        self, potential: Potential, dress: AtomicDress, *, wrt: Sequence[str] = ()
    ) -> None:
        if float(dress.intercept) != 0.0:
            raise ValueError(
                f"DressedPotential needs an elemental dress; this one has intercept "
                f"{float(dress.intercept)!r}, which no atom energy can carry"
            )
        super().__init__()
        self.potential = potential
        self.dress = dress
        self._derivative = Derivative(self.energy, wrt=wrt)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write the absolute ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)``.

        A row is dressed in proportion to ``atoms.mask`` — real atoms in full,
        padding ghosts not at all. The column is required: the padder writes
        it, and so does :meth:`~molix.deploy.AtomEnergyAdapter.build_batch`
        (all true); a frame without it is refused (``KeyError``) rather than
        dressed on every row. An atomic number outside the table is undefined
        here.

        Args:
            frame: Post-collate batch the inner potential reads, with the
                ``graphs`` namespace at ``batch_size=[B]``.

        Returns:
            The same ``frame``, both energy columns absolute.
        """
        frame = self.potential.energy(frame)
        atoms = frame["atoms"]
        residual = atoms["energy"]
        dress = self.dress.per_atom(atoms["Z"]) * atoms["mask"].to(dtype=torch.float64)
        per_atom = (residual.to(dtype=torch.float64) + dress).to(dtype=residual.dtype)
        frame[ATOMIC_ENERGY_KEY] = per_atom
        frame[ENERGY_KEY] = batch_add(
            per_atom, atoms["batch"], dim_size=frame["graphs"].batch_size[0]
        )
        return frame
