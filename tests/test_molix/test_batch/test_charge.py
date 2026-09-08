"""Unit tests for ``molix.batch.charge.TotalCharge`` — the per-graph charge column.

One post-collate task, five concerns: it fills ``graphs.total_charge`` in the
batch's float dtype when the dataset declares no charge, it casts an integral
column (an integer meta scalar collates to ``int64``; the EEM solver wants the
float) to that same dtype, it casts a float column of another *precision* to
it, it hands back the column it was given when that column is already at the
batch dtype, and its ``task_id`` names the declared charge so two declarations
never share a cache.

Pure ``execute()`` tests on two molpy ``Frame``s put through
``collate_molecules`` — no DataModule, no DataLoader, no model. Every charge a
frame declares is a ``meta`` scalar, which is where a dataset states one, so
the three column dtypes under test are the three the collate itself produces:
an integral meta scalar becomes ``int64``, a float one ``float32``, and a frame
that declares nothing leaves the column absent. The batch's own float dtype is
read from ``atoms.pos``, so the fill test uses an fp64 batch and the cast test
an fp32 one: a hard-coded ``torch.get_default_dtype()`` would pass one and fail
the other.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.batch import TensorFrame, TotalCharge
from tests.conftest import collate_frames, frame

_Z = (1, 8)
_POS = (
    ((0.0, 0.0, 0.0), (0.96, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.96, 0.0)),
)


def _batch(
    dtype: torch.dtype,
    total_charge: tuple[float, float] | tuple[int, int] | None = None,
) -> TensorDict:
    """Two 2-atom graphs (``B = 2``) with ``pos`` in *dtype*.

    Args:
        dtype: Floating dtype of ``atoms.pos`` — the batch's float dtype.
        total_charge: Optional charge each frame declares in its ``meta``;
            omitted entirely when ``None`` (the absent-column case). Integers
            collate to an ``int64`` column, floats to a ``float32`` one.

    Returns:
        The collated batch.
    """
    frames = [
        frame(
            Z=_Z,
            pos=pos,
            **({} if total_charge is None else {"total_charge": total_charge[graph]}),
        )
        for graph, pos in enumerate(_POS)
    ]
    return collate_frames(frames, dtype=dtype)


class TestTotalCharge:
    """``TotalCharge`` — the ``BatchTask`` that owns ``graphs.total_charge``."""

    def test_fills_an_absent_column_in_the_batch_float_dtype(self) -> None:
        batch = _batch(torch.float64)
        assert ("graphs", "total_charge") not in batch

        out = TotalCharge(charge=-1.0).execute(batch)

        column = out["graphs", "total_charge"]
        assert column.dtype == torch.float64
        assert column.shape == (2,)
        assert torch.equal(column, torch.full((2,), -1.0, dtype=torch.float64))

    def test_casts_an_integral_column_to_the_batch_float_dtype(self) -> None:
        batch = _batch(torch.float32, (-1, 2))

        out = TotalCharge().execute(batch)

        column = out["graphs", "total_charge"]
        assert column.dtype == torch.float32
        assert torch.equal(column, torch.tensor([-1.0, 2.0], dtype=torch.float32))

    def test_a_float_column_of_another_precision_is_cast_to_the_batch_dtype(self) -> None:
        """fp32 charges in an fp64 batch come out fp64, with their values kept.

        The normalisation used to be conditioned on the column being
        *integral*, which reads "an int64 meta scalar is not what the solver
        wants" — but the actual contract is that the column is at the batch's
        float dtype, and a dataset that stored its declared charges as fp32 is
        as far from it as one that stored them as ints. What the mismatch buys
        downstream is a promotion nobody chose: ``bind_topology`` compares the
        per-graph sum of ``atoms.q`` against this column at 1e-6 e, and a
        constraint checked across two precisions is a tolerance nobody wrote.
        The values are exact halves, so a cast that changed one would show.
        """
        batch = _batch(torch.float64, (0.5, -0.5))

        out = TotalCharge().execute(batch)

        column = out["graphs", "total_charge"]
        assert column.dtype == torch.float64
        assert torch.equal(column, torch.tensor([0.5, -0.5], dtype=torch.float64))

    def test_a_column_already_at_the_batch_dtype_is_the_same_tensor(self) -> None:
        """No copy where there is nothing to change — identity, not equality.

        The declared column is the dataset's own storage — here the tensor the
        collate built from the two frames' ``meta`` scalars, in an fp32 batch
        so that column already is at the batch's float dtype. A task that
        rebuilt it every epoch would be a per-batch allocation for no change at
        all, and the identity check is the only assertion that can tell the two
        apart.
        """
        batch = _batch(torch.float32, (0.5, -0.5))
        declared = batch["graphs", "total_charge"]
        assert declared.dtype == torch.float32

        out = TotalCharge().execute(batch)

        assert out["graphs", "total_charge"] is declared

    def test_a_typed_view_writes_the_column_into_the_batch_it_wraps(self) -> None:
        """``TensorFrame`` is a view, so a task run on it fills the caller's batch.

        The typed read of the post-collate batch (CLAUDE.md § TensorDict
        Contract) adds no storage: ``from_tensordict`` wraps the leaves that are
        already there. A ``BatchTask`` handed that view must therefore be
        indistinguishable from one handed the batch — same column, same dtype,
        and it has to land in the batch the caller still holds, because the view
        is not where the data lives.
        """
        batch = _batch(torch.float64)
        view = TensorFrame.from_tensordict(batch)

        out = TotalCharge(charge=-1.0).execute(view)

        assert ("graphs", "total_charge") in batch.keys(include_nested=True)
        column = batch["graphs", "total_charge"]
        assert column.dtype == torch.float64
        assert torch.equal(column, torch.full((2,), -1.0, dtype=torch.float64))
        assert out["graphs", "total_charge"] is column

    def test_a_plain_dict_is_refused_rather_than_written_nowhere(self) -> None:
        """A dict of namespaces would take the write into a container the caller never sees.

        Wrapping it would build a fresh ``TensorDict`` and fill *that*; leaving
        it unwrapped would store the column under a tuple key of the dict. Both
        come back looking processed, so the task fails closed instead.
        """
        batch = _batch(torch.float64)
        mapping = {name: batch[name] for name in batch.keys()}

        with pytest.raises(TypeError, match=r"collate_molecules"):
            TotalCharge().execute(mapping)

    def test_task_id_names_the_declared_charge(self) -> None:
        assert TotalCharge().task_id == "total_charge:0.0"
        assert TotalCharge(charge=-1.0).task_id == "total_charge:-1.0"
