"""Several data modules in lockstep: one batch of each per step, as a nested batch.

ByteFF's joint training draws one batch from every dataset per optimizer
step and defines an epoch by the shortest of them (``trainer.py:316-318``,
``min(epoch_steps)``). :class:`JointDataModule` is that policy over molnex
data modules: each source keeps its own loader — its own padder capacity,
batch tasks, workers and per-epoch reseed — and the joint module only
interleaves what they yield.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tensordict import TensorDict

__all__ = ["JointDataModule"]


class _Lockstep:
    """K loaders iterated side by side; one nested batch per step.

    Args:
        loaders: Name → an iterable of batches, re-``iter()``-able.
        steps: Batches per pass. ``None`` is zip semantics — the pass ends when
            the first source is exhausted and nothing is re-iterated. A count
            runs exactly that many steps, re-``iter()``-ing a source that runs
            out before then.
    """

    def __init__(self, loaders: Mapping[str, Iterable], *, steps: int | None) -> None:
        self._loaders = dict(loaders)
        self._steps = steps

    def __len__(self) -> int:
        """*steps* when set, else the shortest source's length.

        Raises:
            TypeError: *steps* is unset and a source has no ``len``.
        """
        if self._steps is not None:
            return self._steps
        return min(len(loader) for loader in self._loaders.values())  # ty: ignore[invalid-argument-type]

    def __iter__(self) -> Iterator[TensorDict]:
        # One iterator per distinct loader object. A persistent-worker DataLoader
        # hands its single iterator to every `iter()` call, so two names on one
        # loader would split its batches between them — and a validation set
        # that packs into one batch would score as none (GH200 3680093: every
        # eval "missing", no best.pt). Names that share a loader get the same
        # batch, each its own copy, so the step bodies' in-place writes stay
        # apart; that is also ByteFF's step, every loss of a dataset on the one
        # batch drawn from it.
        groups: dict[int, list[str]] = {}
        loaders: dict[int, Iterable] = {}
        for name, loader in self._loaders.items():
            groups.setdefault(id(loader), []).append(name)
            loaders[id(loader)] = loader
        iterators = {key: iter(loader) for key, loader in loaders.items()}
        produced = 0
        while self._steps is None or produced < self._steps:
            step: dict[str, Any] = {}
            for key, it in iterators.items():
                try:
                    batch = next(it)
                except StopIteration:
                    if self._steps is None:
                        return
                    iterators[key] = it = iter(loaders[key])
                    try:
                        batch = next(it)
                    except StopIteration:
                        raise ValueError(
                            f"joint source {groups[key][0]!r} yields no batches"
                        ) from None
                first, *rest = groups[key]
                step[first] = batch
                for name in rest:
                    step[name] = batch.clone()
            yield TensorDict(step, batch_size=[])
            produced += 1


class JointDataModule:
    """Data modules run in lockstep; every batch is ``TensorDict({name: sub_batch})``.

    ``setup`` and ``on_epoch_start`` are forwarded to each source that has
    them, so every source reseeds its own shuffle per epoch. The same source
    object may sit under two names: when its ``*_dataloader()`` returns the one
    (cached, persistent-worker) loader for both, that loader is read once per
    step and every name gets the same batch (its own copy), so two objectives
    score one dataset's batch; a source that returns a fresh loader per call
    gives each name its own stream.

    **Epoch semantics.** By default an epoch is the **shortest** source's
    (ByteFF ``min(epoch_steps)``): the training pass ends when the first
    source is exhausted, and the longer sources pick up fresh — reshuffled —
    the next epoch, so a large source is sampled a different slice every
    epoch rather than exhausted once. With *steps_per_epoch* set the pass runs
    that many steps, re-iterating a source that runs out (its own
    ``train_dataloader()`` is re-entered, so a reseed-per-call source serves a
    new permutation). Validation always zips to the shortest val loader and
    never cycles — the same prefix of every source is scored each time.

    ``len(train_dataloader())`` is *steps_per_epoch*, or the shortest source's
    ``len`` when every source can report one.

    Args:
        sources: Name → data module (``train_dataloader`` / ``val_dataloader``,
            optionally ``setup`` / ``on_epoch_start``). The names are the keys
            of the nested batch and of the objectives that read it.
        steps_per_epoch: Training steps per epoch; ``None`` is the shortest
            source.

    Raises:
        ValueError: No sources, or ``steps_per_epoch <= 0``.
    """

    def __init__(self, sources: Mapping[str, Any], *, steps_per_epoch: int | None = None) -> None:
        if not sources:
            raise ValueError("JointDataModule needs at least one source")
        if steps_per_epoch is not None and steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
        self.sources = dict(sources)
        self.steps_per_epoch = steps_per_epoch

    def setup(self, stage: str = "fit") -> None:
        """Forward to every source that has ``setup``."""
        for source in self.sources.values():
            setup = getattr(source, "setup", None)
            if callable(setup):
                setup(stage)

    def on_epoch_start(self, epoch: int) -> None:
        """Forward to every source that has ``on_epoch_start``."""
        for source in self.sources.values():
            hook = getattr(source, "on_epoch_start", None)
            if callable(hook):
                hook(epoch)

    def train_dataloader(self) -> Iterable:
        """The lockstep over every source's training loader (see the class)."""
        return _Lockstep(
            {name: source.train_dataloader() for name, source in self.sources.items()},
            steps=self.steps_per_epoch,
        )

    def val_dataloader(self) -> Iterable:
        """The lockstep over every source's validation loader; zip, never cycle."""
        return _Lockstep(
            {name: source.val_dataloader() for name, source in self.sources.items()}, steps=None
        )
