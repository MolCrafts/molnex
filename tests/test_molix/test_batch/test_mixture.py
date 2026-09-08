"""Tests for :class:`molix.batch.mixture.DataMixture`."""

from __future__ import annotations

import numpy as np
from molpy import Block, Frame
from torch.utils.data import Dataset

from molix.batch import DataMixture, DataModule
from molix.store.frame_dataset import FrameCollection, FrameDataset


def _geom(z: list[int], *, molecule_id: str, energy: float = 0.0) -> Frame:
    n = len(z)
    atoms = Block()
    atoms["number"] = np.array(z, dtype=np.int64)
    atoms["x"] = np.arange(n, dtype=np.float64)
    atoms["y"] = np.zeros(n, dtype=np.float64)
    atoms["z"] = np.zeros(n, dtype=np.float64)
    frame = Frame()
    frame["atoms"] = atoms
    frame.meta = {**frame.meta, "energy": energy}
    frame.meta = {**frame.meta, "molecule_id": molecule_id}
    return frame


def _one(frame: Frame) -> FrameCollection:
    return (frame,)


class TestDataMixture:
    def test_is_a_dataset(self):
        eq = FrameDataset.from_records([_one(_geom([8], molecule_id="h"))], task="equilibrium")
        mix = DataMixture(
            {"equilibrium": eq},
            mixture={"equilibrium": 1.0},
            max_atoms=8,
        )
        assert isinstance(mix, Dataset)
        assert len(mix) == 1
        assert len(mix[0]) == 1

    def test_task_name_is_the_source_key(self):
        eq = FrameDataset.from_records(
            [_one(_geom([8, 1], molecule_id="a")), _one(_geom([6], molecule_id="b"))],
            task="ignored",
        )
        tor = FrameDataset.from_records(
            [tuple(_geom([6, 1], molecule_id="s", energy=float(i)) for i in range(2))],
            task="also-ignored",
        )
        mix = DataMixture(
            {"equilibrium": eq, "torsion": tor},
            mixture={"equilibrium": 0.5, "torsion": 0.5},
            max_atoms=16,
        )
        assert mix.task_labels == ("equilibrium", "equilibrium", "torsion")
        assert mix.group_ids == ("equilibrium:0", "equilibrium:1", "torsion:0")
        assert mix.atom_counts.tolist() == [2, 1, 4]

    def test_getitem_returns_the_source_collection(self):
        scan = tuple(_geom([6, 1], molecule_id="s", energy=float(i)) for i in range(2))
        tor = FrameDataset.from_records([scan], task="torsion")
        eq = FrameDataset.from_records([_one(_geom([8], molecule_id="h"))], task="equilibrium")
        mix = DataMixture(
            {"equilibrium": eq, "torsion": tor},
            mixture={"equilibrium": 0.5, "torsion": 0.5},
            max_atoms=16,
        )
        assert len(mix) == 2
        assert len(mix[0]) == 1
        assert len(mix[1]) == 2
        assert [float(frame.meta["energy"].value) for frame in mix[1]] == [0.0, 1.0]

    def test_empty_sources_rejected(self):
        try:
            DataMixture({}, mixture={}, max_atoms=8)
        except ValueError as error:
            assert "at least one source" in str(error)
        else:
            raise AssertionError("empty sources must raise")

    def test_mixture_keys_must_match_sources(self):
        eq = FrameDataset.from_records([_one(_geom([8], molecule_id="h"))], task="equilibrium")
        try:
            DataMixture({"equilibrium": eq}, mixture={"torsion": 1.0}, max_atoms=8)
        except ValueError as error:
            assert "same tasks" in str(error)
        else:
            raise AssertionError("mismatched mixture keys must raise")

    def test_empty_source_rejected(self):
        try:
            DataMixture(
                {"equilibrium": FrameDataset.from_records([], task="equilibrium")},
                mixture={"equilibrium": 1.0},
                max_atoms=8,
            )
        except ValueError as error:
            assert "empty" in str(error)
        else:
            raise AssertionError("empty source must raise")

    def test_rejects_competing_batch_sampler_source(self):
        eq = FrameDataset.from_records([_one(_geom([8], molecule_id="h"))], task="equilibrium")
        mix = DataMixture(
            {"equilibrium": eq},
            mixture={"equilibrium": 1.0},
            max_atoms=8,
        )
        try:
            DataModule(mix, mix, batch_sampler_source=mix, num_workers=0)
        except ValueError as error:
            assert "already is the train mix policy" in str(error)
        else:
            raise AssertionError("DataMixture + batch_sampler_source must raise")

    def test_feeds_datamodule(self):
        eq = FrameDataset.from_records(
            [
                _one(_geom([8, 1], molecule_id="a")),
                _one(_geom([8, 1], molecule_id="b")),
            ],
            task="equilibrium",
        )
        scan = [
            _geom([6, 1], molecule_id="s", energy=0.0),
            _geom([6, 1], molecule_id="s", energy=1.0),
        ]
        tor = FrameDataset.from_records([tuple(scan)], task="torsion")
        mix = DataMixture(
            {"equilibrium": eq, "torsion": tor},
            mixture={"equilibrium": 0.5, "torsion": 0.5},
            max_atoms=16,
        )
        dm = DataModule(mix, mix, batch_size=1, num_workers=0)
        dm.setup()
        batches = list(dm.train_dataloader())
        assert batches
        first = batches[0]
        assert "Z" in first["atoms"]
        assert first["atoms", "pos"].shape[-1] == 3
        val = list(dm.val_dataloader())
        assert val
