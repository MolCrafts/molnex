"""Write LAMMPS data and settings via :mod:`molpy.io`.

The Class-I :class:`~molpot.forcefield.ForceField` **is** a molpy
``ForceField``. This type builds a :class:`~molpy.core.atomistic.Atomistic`
frame, calls :meth:`~molpot.forcefield.ForceField.realize` to fill native
types, then delegates files to molpy. Independent of ``pair_style molnex``.

Required system contract (post-collate nested TensorDict)::

    system["atoms"]["pos"]                 # (N, 3) Å
    system["atoms"]["q"]                   # optional (N,)
    system["bonds"]["atomi" / "atomj"]     # optional
    system["angles"]["atomi" / "atomj" / "atomk"]
    system["propers"]["atomi" … "atoml"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from molpot.forcefield.container import ForceField

__all__ = ["LammpsWriter"]


class LammpsWriter:
    """Serialize a Class-I :class:`~molpot.forcefield.ForceField` through molpy IO.

    Identity types: one atom type per atom, one bonded type per interaction.
    """

    def write_data(
        self,
        path: str | Path,
        system: TensorDict,
        *,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Path:
        """Write an ``atom_style full`` data file via :func:`molpy.io.write_lammps_data`.

        Args:
            path: Destination file.
            system: Nested TensorDict with ``atoms.pos`` and optional bonded
                namespaces (``atomi`` … columns).
            masses: Per-atom mass in amu ``(N,)``. Default: 12.011.
            charges: Per-atom charge ``(N,)``. Default: ``atoms.q`` or zeros.
            box_pad_A: Padding around the molecule for an open box (Å).

        Returns:
            The written path.
        """
        from molpy.io.writers import write_lammps_data

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_lammps_data(
            dest, self.to_frame(system, masses=masses, charges=charges, box_pad_A=box_pad_A)
        )
        return dest

    def write_settings(
        self,
        path: str | Path,
        forcefield: ForceField,
        system: TensorDict,
        *,
        masses: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Path:
        """Write a LAMMPS include via :func:`molpy.io.write_lammps_forcefield`.

        Args:
            path: Destination file.
            forcefield: Filled Class-I container.
            system: Topology used to assign identity types.
            masses: Per-atom mass in amu (must match :meth:`write_data`).
            box_pad_A: Box padding (must match :meth:`write_data`).

        Returns:
            The written path.
        """
        from molpy.io.writers import write_lammps_forcefield

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame(system, masses=masses, box_pad_A=box_pad_A)
        ff = forcefield.clone().realize(system, masses=masses)
        write_lammps_forcefield(dest, ff, frame=frame, units=ff.units)
        return dest

    def to_frame(
        self,
        system: TensorDict,
        *,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Any:
        """Build a molpy ``Frame`` (Atomistic → ``to_frame``) from ``system``."""
        from molpy import Box
        from molpy.core.atomistic import Atomistic

        pos = self._tensor(system, "atoms", "pos")
        xyz = pos.detach().cpu().double().numpy()
        n_atoms = int(xyz.shape[0])
        mass_list = ForceField._mass_list(masses, n_atoms)
        if charges is not None:
            charge_list = [float(x) for x in torch.as_tensor(charges).reshape(-1).tolist()]
        elif ("atoms", "q") in system.keys(include_nested=True):
            charge_list = [
                float(x)
                for x in self._tensor(system, "atoms", "q").detach().cpu().reshape(-1).tolist()
            ]
        else:
            charge_list = [0.0] * n_atoms

        world = Atomistic()
        handles = []
        for i in range(n_atoms):
            handles.append(
                world.def_atom(
                    {
                        "type": str(i + 1),
                        "x": float(xyz[i, 0]),
                        "y": float(xyz[i, 1]),
                        "z": float(xyz[i, 2]),
                        "charge": charge_list[i],
                        "mol": 1,
                        "mass": mass_list[i],
                    }
                )
            )
        for i, j in ForceField._arity(system, "bonds", 2):
            world.def_bond(handles[i], handles[j], type=f"{i + 1}-{j + 1}")
        for i, j, k in ForceField._arity(system, "angles", 3):
            world.def_angle(handles[i], handles[j], handles[k], type=f"{i + 1}-{j + 1}-{k + 1}")
        for i, j, k, l in ForceField._arity(system, "propers", 4):
            world.def_dihedral(
                handles[i],
                handles[j],
                handles[k],
                handles[l],
                type=f"{i + 1}-{j + 1}-{k + 1}-{l + 1}",
            )
        frame = world.to_frame()
        frame.box = Box.from_bounds(np.asarray(xyz, dtype=float), padding=box_pad_A)
        return frame

    @staticmethod
    def _tensor(system: TensorDict, *keys: str) -> torch.Tensor:
        value = system[keys] if len(keys) > 1 else system[keys[0]]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"system{list(keys)} must be a tensor, got {type(value)!r}")
        return value
