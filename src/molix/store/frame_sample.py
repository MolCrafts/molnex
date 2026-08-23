"""One molpy ``Frame`` as the pre-collate sample dict ``collate_molecules`` reads.

Does not import molhub. Does not invent topology: bonds / angles / propers /
impropers are copied when the frame already carries them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["FrameSample"]

_XYZ = ("x", "y", "z")
_FORCES = ("fx", "fy", "fz")
_META_SCALARS = ("energy", "mm_energy", "torsion_angle")
_VALENCE = (
    ("angles", ("atomi", "atomj", "atomk")),
    ("propers", ("atomi", "atomj", "atomk", "atoml")),
    ("impropers", ("atomi", "atomj", "atomk", "atoml")),
)


def _column(block: Any, name: str) -> np.ndarray:
    return np.asarray(block[name])


class FrameSample:
    """A single geometry, ready to collate.

    Args:
        frame: A molpy ``Frame`` with an ``atoms`` block that has ``number``
            and ``x`` / ``y`` / ``z``.
    """

    def __init__(self, frame: Any) -> None:
        if "atoms" not in frame:
            raise KeyError("FrameSample needs an atoms block")
        self._frame = frame

    @property
    def n_atoms(self) -> int:
        """Atom count of this geometry."""
        return int(np.asarray(self._frame["atoms"]["number"]).shape[0])

    def as_dict(self) -> dict[str, Any]:
        """Flat pre-collate sample: ``Z``, ``pos``, optional bonds and targets."""
        atoms = self._frame["atoms"]
        z = torch.as_tensor(_column(atoms, "number"), dtype=torch.long)
        pos = torch.stack(
            [torch.as_tensor(_column(atoms, axis), dtype=torch.float32) for axis in _XYZ],
            dim=-1,
        )
        sample: dict[str, Any] = {"Z": z, "pos": pos}
        if "bonds" in self._frame:
            bonds = self._frame["bonds"]
            sample["bond_index"] = torch.stack(
                [
                    torch.as_tensor(_column(bonds, "atomi"), dtype=torch.long),
                    torch.as_tensor(_column(bonds, "atomj"), dtype=torch.long),
                ],
                dim=0,
            )
        for name, cols in _VALENCE:
            if name not in self._frame:
                continue
            block = self._frame[name]
            sample[name] = {
                col: torch.as_tensor(_column(block, col), dtype=torch.long) for col in cols
            }
        targets = self._targets(atoms)
        if targets:
            sample["targets"] = targets
        return sample

    def _targets(self, atoms: Any) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        meta = getattr(self._frame, "meta", None) or {}
        for key in _META_SCALARS:
            if key in meta:
                targets[key] = float(meta[key])
        if all(col in atoms for col in _FORCES):
            targets["forces"] = torch.stack(
                [torch.as_tensor(_column(atoms, col), dtype=torch.float32) for col in _FORCES],
                dim=-1,
            )
        if "atomic_charge" in atoms:
            targets["atomic_charge"] = torch.as_tensor(
                _column(atoms, "atomic_charge"), dtype=torch.float32
            )
        return targets
