"""JointDataModule — mirrors ``src/molix/data/joint.py``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from tensordict import TensorDict

from molix.data import DataModule, JointDataModule
from molix.data.dataset import BaseDataset
from tests.conftest import frame


def _batch(i: int) -> TensorDict:
    return TensorDict(
        {"graphs": TensorDict({"i": torch.tensor([i])}, batch_size=[1])}, batch_size=[]
    )


class _Source:
    """A data module whose loaders are lists: ``Sized`` and re-iterable."""

    def __init__(self, n_train: int, n_val: int) -> None:
        self.n_train, self.n_val = n_train, n_val
        self.stages: list[str] = []
        self.epochs: list[int] = []
        self.train_calls = 0

    def setup(self, stage: str = "fit") -> None:
        self.stages.append(stage)

    def on_epoch_start(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def train_dataloader(self) -> list[TensorDict]:
        self.train_calls += 1
        return [_batch(i) for i in range(self.n_train)]

    def val_dataloader(self) -> list[TensorDict]:
        return [_batch(i) for i in range(self.n_val)]


class _PersistentLoader:
    """What a persistent-worker DataLoader does: one iterator, handed to every
    ``iter()`` call and reset there."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._iterator: Iterator[TensorDict] | None = None

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[TensorDict]:
        self._iterator = (_batch(i) for i in range(self.n))
        return self._iterator


class _PersistentSource(_Source):
    """A data module that caches its loaders, as ``DataModule`` does."""

    def __init__(self, n_train: int, n_val: int) -> None:
        super().__init__(n_train, n_val)
        self._train = _PersistentLoader(n_train)
        self._val = _PersistentLoader(n_val)

    def train_dataloader(self) -> _PersistentLoader:
        self.train_calls += 1
        return self._train

    def val_dataloader(self) -> _PersistentLoader:
        return self._val


def _indices(batches: list[TensorDict], name: str) -> list[int]:
    return [int(batch[name, "graphs", "i"]) for batch in batches]


class TestJointDataModule:
    def test_a_batch_is_keyed_by_source(self) -> None:
        dm = JointDataModule({"short": _Source(2, 1), "long": _Source(3, 3)})
        batch = next(iter(dm.train_dataloader()))
        assert set(batch.keys()) == {"short", "long"}
        assert int(batch["long", "graphs", "i"]) == 0

    def test_a_batch_moves_as_one_tensordict(self) -> None:
        dm = JointDataModule({"short": _Source(2, 1), "long": _Source(3, 3)})
        moved = next(iter(dm.train_dataloader())).to("cpu")
        assert isinstance(moved, TensorDict)
        assert set(moved.keys()) == {"short", "long"}

    def test_the_epoch_is_the_shortest_source(self) -> None:
        dm = JointDataModule({"short": _Source(2, 1), "long": _Source(3, 3)})
        train = dm.train_dataloader()
        assert len(train) == 2
        assert _indices(list(train), "long") == [0, 1]

    def test_steps_per_epoch_cycles_an_exhausted_source(self) -> None:
        dm = JointDataModule({"short": _Source(2, 1), "long": _Source(3, 3)}, steps_per_epoch=5)
        train = dm.train_dataloader()
        batches = list(train)
        assert len(train) == len(batches) == 5
        assert _indices(batches, "short") == [0, 1, 0, 1, 0]
        assert _indices(batches, "long") == [0, 1, 2, 0, 1]

    def test_validation_zips_to_the_shortest_and_never_cycles(self) -> None:
        dm = JointDataModule({"short": _Source(2, 1), "long": _Source(3, 3)}, steps_per_epoch=5)
        val = dm.val_dataloader()
        assert len(val) == 1
        assert _indices(list(val), "long") == [0]

    def test_setup_and_epoch_start_reach_every_source(self) -> None:
        sources = {"short": _Source(2, 1), "long": _Source(3, 3)}
        dm = JointDataModule(sources)
        dm.setup("fit")
        dm.on_epoch_start(7)
        assert all(source.stages == ["fit"] for source in sources.values())
        assert all(source.epochs == [7] for source in sources.values())

    def test_a_source_handing_out_fresh_loaders_gives_each_name_a_stream(self) -> None:
        source = _Source(3, 1)
        dm = JointDataModule({"x": source, "y": source})
        batches = list(dm.train_dataloader())
        assert source.train_calls == 2
        assert _indices(batches, "x") == _indices(batches, "y") == [0, 1, 2]

    def test_a_source_handing_out_one_loader_is_read_once_and_copied(self) -> None:
        """A one-batch validation set under two names is one nested batch, not
        none: the shared iterator is advanced once per step and the second name
        gets a copy of the same batch, with its own storage."""
        source = _PersistentSource(3, 1)
        dm = JointDataModule({"x": source, "y": source})
        val = list(dm.val_dataloader())
        assert _indices(val, "x") == _indices(val, "y") == [0]
        assert val[0]["x", "graphs", "i"].data_ptr() != val[0]["y", "graphs", "i"].data_ptr()
        train = list(dm.train_dataloader())
        assert _indices(train, "x") == _indices(train, "y") == [0, 1, 2]

    @pytest.mark.parametrize(
        ("sources", "steps"),
        [({}, None), ({"a": _Source(1, 1)}, 0)],
        ids=["no-sources", "zero-steps"],
    )
    def test_rejects_a_malformed_declaration(self, sources: dict, steps: int | None) -> None:
        with pytest.raises(ValueError):
            JointDataModule(sources, steps_per_epoch=steps)


class _Frames(BaseDataset):
    def __init__(self, n: int) -> None:
        self._frames = [
            frame(Z=[1, 6], pos=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], energy=float(i))
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        return self._frames[idx]


def test_real_data_modules_run_in_lockstep() -> None:
    """A ``DataModule`` loader is ``Sized`` and yields the batch schema under its name."""
    short = DataModule(_Frames(4), _Frames(2), batch_size=2, num_workers=0, pin_memory=False)
    long = DataModule(_Frames(6), _Frames(2), batch_size=2, num_workers=0, pin_memory=False)
    dm = JointDataModule({"short": short, "long": long})
    train = dm.train_dataloader()
    batches = list(train)
    assert len(train) == len(batches) == 2
    assert batches[0]["long", "atoms", "Z"].shape == (4,)
