"""Tests for :class:`molix.data.FrameCollator`."""

from __future__ import annotations

import pickle

import pytest
import torch

from molix.data import FrameCollator, MoleculeCollator, MoleculePadder
from tests.conftest import frame


def _two_water() -> list:
    return [
        frame(Z=[8, 1, 1], pos=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        frame(Z=[8, 1, 1], pos=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    ]


def test_ragged_occupancy_is_one() -> None:
    collate = FrameCollator(collator=MoleculeCollator())
    collate(_two_water())
    assert collate.occupancy_atoms() == 1.0
    assert collate.occupancy_pairs() == 1.0
    assert collate.sparsity() == {"atoms": 0.0, "pairs": 0.0, "graphs": 0.0}
    assert collate.pad_overhead()["bytes"] == 0.0


def test_padder_occupancy_is_n_real_over_cap() -> None:
    collate = FrameCollator(
        collator=MoleculeCollator(),
        padder=MoleculePadder(max_atoms=16, max_pairs=64, cutoff=5.0),
    )
    batch = collate(_two_water())
    assert batch["atoms"].batch_size[0] == 16
    assert collate.occupancy_atoms() == 6 / 16
    assert collate.occupancy_valence() == {}
    assert collate.nbytes() > collate.nbytes_payload()
    assert collate.pad_overhead()["bytes"] > 0


def test_graph_schema_compiles_from_the_first_batch_and_reorders_later_ones() -> None:
    from tensordict import TensorDict

    from molix.data.collate import freeze_graph_columns, graph_schema_from

    collate = FrameCollator(collator=MoleculeCollator())
    first = TensorDict(
        graphs=TensorDict(
            {
                "task": torch.zeros(1, dtype=torch.long),
                "energy": torch.zeros(1),
                "num_atoms": torch.ones(1, dtype=torch.long),
            },
            batch_size=[1],
        ),
        batch_size=[],
    )
    assert collate.compile_graph_schema(first) == graph_schema_from(["task", "energy", "num_atoms"])
    second = TensorDict(
        graphs=TensorDict(
            {
                "energy": torch.ones(1),
                "num_atoms": torch.ones(1, dtype=torch.long),
                "task": torch.ones(1, dtype=torch.long),
                "spin": torch.zeros(1),
            },
            batch_size=[1],
        ),
        batch_size=[],
    )
    out = collate.apply_graph_schema(second)
    assert list(out["graphs"].keys()) == list(collate._graph_schema)
    assert "spin" not in out["graphs"].keys()
    third = TensorDict(
        graphs=TensorDict({"num_atoms": torch.ones(1, dtype=torch.long)}, batch_size=[1]),
        batch_size=[],
    )
    with pytest.raises(ValueError, match="missing"):
        freeze_graph_columns(third, collate._graph_schema)


def test_frame_collator_pickles_without_last_batch() -> None:
    collate = FrameCollator(
        collator=MoleculeCollator(),
        padder=MoleculePadder(max_atoms=16, max_pairs=64, cutoff=5.0),
    )
    collate(_two_water())
    schema = collate._graph_schema
    clone = pickle.loads(pickle.dumps(collate))
    assert clone._last is None
    assert clone._graph_schema == schema
    out = clone(_two_water())
    assert out["atoms"].batch_size[0] == 16


class _Tagging:
    """A ``Prepare``: tags every frame of a record with the record's size."""

    def __init__(self) -> None:
        self.records: list[int] = []

    def execute_record(self, frames):
        self.records.append(len(frames))
        for sample in frames:
            sample.meta["tag"] = float(len(frames))
        return tuple(frames)


def test_prepare_runs_execute_record_once_per_sample() -> None:
    """A single frame and a multi-frame record are each one ``execute_record`` call."""
    prepare = _Tagging()
    collator = FrameCollator(collator=MoleculeCollator(), prepare=prepare)
    single = frame(Z=[1], pos=[[0.0, 0.0, 0.0]])
    pair = (frame(Z=[1], pos=[[0.0, 0.0, 0.0]]), frame(Z=[1], pos=[[1.0, 0.0, 0.0]]))

    batch = collator([single, pair])

    assert prepare.records == [1, 2]
    assert batch["graphs", "tag"].tolist() == [1.0, 2.0, 2.0]
