"""Tests for :class:`molix.store.frame_sample.FrameSample`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from molpy import Block, Frame

from molix.store.frame_sample import FrameSample


def _water(*, energy: float | None = -1.5, forces: bool = False) -> Frame:
    atoms = Block()
    atoms["element"] = np.array(["O", "H", "H"], dtype="U2")
    atoms["number"] = np.array([8, 1, 1], dtype=np.int64)
    atoms["x"] = np.array([0.0, 0.96, -0.24], dtype=np.float64)
    atoms["y"] = np.array([0.0, 0.0, 0.93], dtype=np.float64)
    atoms["z"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if forces:
        atoms["fx"] = np.array([0.1, -0.05, -0.05], dtype=np.float64)
        atoms["fy"] = np.zeros(3, dtype=np.float64)
        atoms["fz"] = np.zeros(3, dtype=np.float64)
    bonds = Block()
    bonds["atomi"] = np.array([0, 0], dtype=np.int64)
    bonds["atomj"] = np.array([1, 2], dtype=np.int64)
    frame = Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    if energy is not None:
        frame.meta = {**frame.meta, "energy": energy}
    frame.meta = {**frame.meta, "molecule_id": "water"}
    return frame


class TestFrameSample:
    def test_n_atoms(self):
        assert FrameSample(_water()).n_atoms == 3

    def test_as_dict_has_z_and_pos(self):
        sample = FrameSample(_water()).as_dict()
        assert sample["Z"].tolist() == [8, 1, 1]
        assert sample["pos"].shape == (3, 3)

    def test_as_dict_copies_bonds(self):
        sample = FrameSample(_water()).as_dict()
        assert sample["bond_index"].shape == (2, 2)
        assert sample["bond_index"][0].tolist() == [0, 0]
        assert sample["bond_index"][1].tolist() == [1, 2]

    def test_energy_is_a_graph_target(self):
        sample = FrameSample(_water(energy=-2.0)).as_dict()
        assert sample["targets"]["energy"] == -2.0

    def test_forces_are_an_atom_target(self):
        sample = FrameSample(_water(forces=True)).as_dict()
        assert sample["targets"]["forces"].shape == (3, 3)
        assert torch.allclose(sample["targets"]["forces"][0], torch.tensor([0.1, 0.0, 0.0]))

    def test_missing_atoms_block_raises(self):
        with pytest.raises(KeyError, match="atoms"):
            FrameSample(Frame())
