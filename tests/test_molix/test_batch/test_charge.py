"""Unit tests for ``molix.batch.charge.TotalCharge`` — the per-graph charge column.

One post-collate task, five concerns: it fills ``graphs.total_charge`` in the
batch's float dtype when the dataset declares no charge, it casts an integral
column (an integer meta scalar collates to ``int64``; the EEM solver wants the
float) to that same dtype, it casts a float column of another *precision* to
it, it hands back the column it was given when that column is already at the
batch dtype, and its ``task_id`` names the declared charge so two declarations
never share a cache.

Pure ``execute()`` tests on a hand-collated two-graph batch — no DataModule, no
DataLoader, no model. The float dtype is read from ``atoms.pos``, so the fill
test uses an fp64 batch and the cast test an fp32 one: a hard-coded
``torch.get_default_dtype()`` would pass one and fail the other.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.batch import TotalCharge
from tests.conftest import make_graph_batch

_Z = torch.tensor([1, 8, 1, 8], dtype=torch.long)
_EDGE_INDEX = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
_BATCH = torch.tensor([0, 0, 1, 1], dtype=torch.long)
_POS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.96, 0.0],
    ]
)


def _batch(
    dtype: torch.dtype,
    total_charge: torch.Tensor | None = None,
) -> TensorDict:
    """Two 2-atom graphs (``B = 2``) with ``pos`` in *dtype*.

    Args:
        dtype: Floating dtype of ``atoms.pos`` — the batch's float dtype.
        total_charge: Optional ``(2,)`` ``graphs.total_charge`` column; omitted
            entirely when ``None`` (the absent-column case).

    Returns:
        The collated batch.
    """
    graphs = None if total_charge is None else {"total_charge": total_charge}
    return make_graph_batch(_POS.to(dtype), _Z, _EDGE_INDEX, _BATCH, graphs=graphs)


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
        batch = _batch(torch.float32, torch.tensor([-1, 2], dtype=torch.int64))

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
        batch = _batch(torch.float64, torch.tensor([0.5, -0.5], dtype=torch.float32))

        out = TotalCharge().execute(batch)

        column = out["graphs", "total_charge"]
        assert column.dtype == torch.float64
        assert torch.equal(column, torch.tensor([0.5, -0.5], dtype=torch.float64))

    def test_a_column_already_at_the_batch_dtype_is_the_same_tensor(self) -> None:
        """No copy where there is nothing to change — identity, not equality.

        The declared column is the dataset's own storage; a task that rebuilt
        it every epoch would be a per-batch allocation for no change at all,
        and the identity check is the only assertion that can tell the two
        apart.
        """
        declared = torch.tensor([0.5, -0.5], dtype=torch.float64)
        batch = _batch(torch.float64, declared)

        out = TotalCharge().execute(batch)

        assert out["graphs", "total_charge"] is declared

    def test_task_id_names_the_declared_charge(self) -> None:
        assert TotalCharge().task_id == "total_charge:0.0"
        assert TotalCharge(charge=-1.0).task_id == "total_charge:-1.0"
