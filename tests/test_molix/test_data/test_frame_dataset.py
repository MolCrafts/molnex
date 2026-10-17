"""Tests for :class:`molix.data.frame_dataset.FrameDataset`."""

from __future__ import annotations

import numpy as np
from molpy import Block, Frame

from molix.data import collate_molecules
from molix.data.frame_dataset import FrameDataset


def _geom(z: list[int], *, molecule_id: str, energy: float = 0.0) -> Frame:
    n = len(z)
    atoms = Block()
    atoms["atomic_number"] = np.array(z, dtype=np.uint64)
    atoms["x"] = np.arange(n, dtype=np.float64)
    atoms["y"] = np.zeros(n, dtype=np.float64)
    atoms["z"] = np.zeros(n, dtype=np.float64)
    frame = Frame()
    frame["atoms"] = atoms
    frame.meta = {**frame.meta, "energy": energy}
    frame.meta = {**frame.meta, "molecule_id": molecule_id}
    return frame


def _one(frame: Frame) -> tuple[Frame, ...]:
    return (frame,)


class TestFrameDataset:
    def test_from_records_labels_every_collection(self):
        ds = FrameDataset.from_records(
            [
                _one(_geom([8, 1], molecule_id="m0")),
                _one(_geom([6, 1, 1], molecule_id="m1")),
            ],
            task="equilibrium",
        )
        assert len(ds) == 2
        assert ds.task_labels == ("equilibrium", "equilibrium")
        assert ds.group_ids == ("m0", "m1")
        from molix.data.dataset import BaseDataset

        assert BaseDataset.nrows_of(ds, "atoms").tolist() == [2, 3]
        assert len(ds[0]) == 1

    def test_scan_stays_one_index(self):
        frames = [_geom([8, 1], molecule_id="scan-a", energy=float(i)) for i in range(3)]
        for frame in frames:
            frame.meta = {**frame.meta, "scan_id": "scan-a"}
        ds = FrameDataset.from_records([tuple(frames)], task="torsion")
        assert len(ds) == 1
        assert ds.group_ids == ("scan-a",)
        from molix.data.dataset import BaseDataset

        assert BaseDataset.nrows_of(ds, "atoms").tolist() == [6]
        assert [float(frame.meta["energy"]) for frame in ds[0]] == [0.0, 1.0, 2.0]

    def test_getitem_is_collateable(self):
        ds = FrameDataset.from_records(
            [_one(_geom([8, 1, 1], molecule_id="w"))], task="equilibrium"
        )
        batch = collate_molecules([ds[0]])
        assert batch["atoms", "Z"].tolist() == [8, 1, 1]
        assert batch["graphs", "energy"].shape == (1,)

    def test_empty_collection_is_rejected(self):
        try:
            FrameDataset.from_records([()], task="equilibrium")
        except ValueError as error:
            assert "at least one Frame" in str(error)
        else:
            raise AssertionError("an empty record must raise")
