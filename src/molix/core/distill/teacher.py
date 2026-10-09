"""A frozen teacher that stamps TensorDict columns onto a student batch.

A molnex forward receives a ``TensorDict`` and returns one. Predictions are
nested columns on that batch (``("graphs", "energy")``, ``("atoms",
"forces")``, ``("atoms", "node_features")``). This teacher clones the student
batch, runs that forward, and copies the declared columns onto the original
under caller-chosen destination keys. A ``dict`` return is refused: reading
it would be a second forward contract.

The stamp runs in the eager prologue
(:class:`~molix.core.distill.step.DistillTrainStep`). ``torch.no_grad``
rather than ``inference_mode``: an inference tensor cannot be saved for the
student backward. Each stamped value is ``detach().clone()`` so the student
graph holds a constant.

Reference:
    Kelvinius et al. "Accelerating Molecular Graph Neural Networks via
    Knowledge Distillation." NeurIPS 2023.
    https://arxiv.org/abs/2306.14818
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDictBase

NestedKey = tuple[str, str]


def _nested(key: object, *, what: str) -> NestedKey:
    """A ``(namespace, column)`` pair, or a :class:`TypeError` naming *what*."""
    if isinstance(key, tuple) and len(key) == 2:
        namespace, column = key
        if isinstance(namespace, str) and isinstance(column, str):
            return (namespace, column)
    raise TypeError(f"{what} must be a (namespace, column) pair of strings, got {key!r}")


def _tensor(value: object, *, what: str) -> torch.Tensor:
    """*value* when it is a tensor, otherwise a :class:`TypeError` naming *what*."""
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"{what} produced {type(value).__name__}, expected a tensor")


def _unique_destinations(dests: Sequence[NestedKey]) -> tuple[NestedKey, ...]:
    """*dests* with duplicates rejected, so a later column cannot hide an earlier one."""
    frozen = tuple(dests)
    if len(frozen) != len(set(frozen)):
        raise ValueError(f"Teacher destination columns overlap: {frozen}")
    return frozen


class Teacher(nn.Module):
    """Frozen module that copies nested columns off its ``TensorDict`` forward.

    Construction sets the module to ``eval`` and freezes every parameter in
    place. Pass a module the student does not share: the freeze reaches the
    caller's object.

    Args:
        model: ``forward(batch) -> TensorDict``. A derivative wrapper the
            teacher needs is already part of this module.
        columns: ``(source, destination)`` pairs. Both ends are
            ``(namespace, column)``. Source is read from the returned batch.
            Destination is written on the student batch. At least one pair.
            Destinations unique.
    """

    def __init__(
        self,
        model: nn.Module,
        columns: Sequence[tuple[NestedKey, NestedKey]],
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(f"teacher model must be an nn.Module, got {type(model).__name__}")
        if not columns:
            raise ValueError("Teacher stamps at least one column")
        parsed = tuple(
            (
                _nested(src, what="Teacher source"),
                _nested(dest, what="Teacher destination"),
            )
            for src, dest in columns
        )
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self._columns = parsed
        self._dest_keys = _unique_destinations(tuple(dest for _, dest in parsed))

    @property
    def dest_keys(self) -> tuple[NestedKey, ...]:
        """Destination columns this teacher writes, in declaration order."""
        return self._dest_keys

    def write(self, batch: TensorDictBase) -> None:
        """Stamp this teacher's columns onto *batch*.

        The model runs on a clone, under ``no_grad``. Columns on *batch* stay
        as they were except the declared destinations.

        Args:
            batch: The student batch. It must carry ``atoms.pos``, which is
                the device this teacher's parameters are checked against.

        Raises:
            RuntimeError: Called from inside a compiled region. The stamp
                belongs in the eager prologue, so a teacher that host-syncs
                cannot break the student graph.
            ValueError: A parameter or buffer sits on a different device from
                ``atoms.pos``.
            KeyError: ``atoms.pos`` is absent, a source column is absent, or
                a destination namespace is absent.
            TypeError: ``forward`` did not return a ``TensorDict``, or a
                source value is not a tensor.
        """
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "Teacher.write runs in the eager prologue, outside the compiled student region"
            )
        self._check_device(batch)
        # A Trainer.train() on some parent must not turn dropout back on here.
        self.model.eval()
        with torch.no_grad():
            # Full clone: the teacher may overwrite columns on its copy.
            # The student's leaves stay on *batch*, so this graph dies with the clone.
            work = batch.clone()
            returned = self.model(work)
            stamped = self._read(returned)
        for dest, value in stamped:
            self._assign(batch, dest, value)

    def _read(self, returned: object) -> tuple[tuple[NestedKey, torch.Tensor], ...]:
        """Source columns from the ``TensorDict`` the forward returned."""
        if not isinstance(returned, TensorDictBase):
            raise TypeError(
                f"{type(self.model).__name__}.forward must return a TensorDict, "
                f"got {type(returned).__name__}"
            )
        found: list[tuple[NestedKey, torch.Tensor]] = []
        for src, dest in self._columns:
            try:
                value = returned[src]
            except KeyError as exc:
                raise KeyError(f"Teacher source {src!r} is absent on the forwarded batch") from exc
            found.append((dest, _tensor(value, what=f"Teacher source {src!r}")))
        return tuple(found)

    def _check_device(self, batch: TensorDictBase) -> None:
        """Raise when the first parameter or buffer is off the batch device."""
        try:
            pos = batch["atoms", "pos"]
        except KeyError as exc:
            raise KeyError("Teacher.write needs batch['atoms', 'pos']") from exc
        param = next(self.model.parameters(), None)
        if param is None:
            param = next(self.model.buffers(), None)
        if param is not None and param.device != pos.device:
            raise ValueError(
                f"Teacher is on {param.device}, batch atoms.pos is on {pos.device}. "
                "Move the teacher before the step."
            )

    def _assign(self, batch: TensorDictBase, dest: NestedKey, value: torch.Tensor) -> None:
        """Write one detached constant. Never creates a missing namespace."""
        namespace, leaf = dest
        if namespace not in batch.keys():
            raise KeyError(
                f"Teacher destination namespace {namespace!r} is absent; "
                f"refusing to create a schema for {dest!r}"
            )
        batch[namespace, leaf] = value.detach().clone()
