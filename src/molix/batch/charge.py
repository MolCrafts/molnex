"""Per-graph total charge, declared once as a post-collate batch task.

``graphs.total_charge`` ``(B,)`` is the S5 total-charge constraint of a charge
model: the net charge, in elementary charges ``e``, that a graph's per-atom
partial charges must sum to. Since a0-02 the
:class:`~molpot.forcefield.parameterizer.Parameterizer` is *fail-closed* on it
— a charge model never assumes neutrality, because equilibrating an ion as a
neutral molecule shifts every partial charge by a constant with nothing in the
run record to say so.

Whether a pipeline's graphs are neutral is therefore a **declaration**, made
once where the batch tasks are mounted, rather than a float pasted into every
collate. :class:`TotalCharge` is that declaration, and a dataset that carries
its own charge column keeps it.

Reference:
    ``.claude/notes/composable-potentials.md`` F12 — the workflow that raised
    at its first forward for want of this column.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.batch.task import BatchTask


class TotalCharge(BatchTask):
    """Write (or float-normalise) ``graphs.total_charge`` on a collated batch.

    Mounted on a :class:`~molix.batch.datamodule.DataModule` as
    ``batch_tasks=[..., TotalCharge()]``. The default ``0.0`` reads as *the
    graphs of this pipeline are neutral* — one statement for the whole run,
    which is auditable, instead of a zero re-spelled at every collate site.

    A dataset that already carries ``total_charge`` keeps its values: the
    column is then only cast to the batch's float dtype, never overwritten with
    ``charge``. So a mixed-charge dataset is served by its own column and
    ``charge`` only fills the gap where there is none.

    "The batch's float dtype" means ``atoms.pos``'s, and the cast is on
    **inequality**, not on integrality. An integral meta scalar collates to
    ``int64`` and the EEM — *electronegativity equalization method*, the linear
    solve that distributes a molecule's charge over its atoms — wants the float
    it solves in; but so does an fp32 column that arrives in an fp64 batch,
    which is the ordinary shape of a dataset written at single precision and
    read at double. Normalising only the non-float case leaves that one to be
    caught (or promoted, or refused) somewhere further down, per kernel.

    What consumes the column: a charge model's total-charge constraint, and
    :meth:`molpot.forcefield.Potentials.bind_topology`, which cross-checks the
    per-graph sum of ``atoms.q`` against it once per topology and refuses a
    field whose charges have drifted off the declaration. Declaring a charge
    the data does not have is therefore caught, not absorbed.

    Args:
        charge: Net charge in ``e`` (elementary charges) written for every graph
            of a batch that carries no ``total_charge`` column. ``0.0``
            (neutral) by default.
    """

    def __init__(self, charge: float = 0.0) -> None:
        self.charge = float(charge)

    @property
    def task_id(self) -> str:
        """Cache-key identity ``total_charge:<charge>``, the declared charge in ``e``.

        The charge is part of the identity because it is part of the model: a
        pipeline declared neutral and one declared at ``-1`` produce different
        batches, so they must not share a cache entry.
        """
        return f"total_charge:{self.charge}"

    def execute(self, data: TensorDict | dict) -> TensorDict | dict:
        """Fill an absent ``graphs.total_charge``, or float-cast a present one.

        Args:
            data: Collated batch carrying an ``atoms`` namespace — whose ``pos``
                ``(N, 3)`` in Å fixes the batch float dtype and device — and a
                ``graphs`` namespace sized ``[B]``, one row per graph.

        Returns:
            The same batch, mutated in place, with ``graphs.total_charge``
            ``(B,)`` in ``e`` and in the batch float dtype. A present column
            already in that dtype is left as the *same tensor object* — no copy
            is made to prove a point.

        Raises:
            KeyError: The batch has no ``atoms`` or ``graphs`` namespace —
                there is then neither a dtype to write in nor a set of graphs to
                write for. Both are post-collate guarantees
                (:func:`~molix.batch.collate.collate_molecules`), so this means
                the task was mounted somewhere other than after a collate.
            IndexError: ``graphs`` carries no batch dimension (``batch_size=[]``
                rather than ``[B]``), so the graph count is unknown.
        """
        pos = data["atoms"]["pos"]
        graphs = data["graphs"]
        if "total_charge" in graphs:
            declared = graphs["total_charge"]
            if declared.dtype != pos.dtype:
                data["graphs", "total_charge"] = declared.to(pos.dtype)
            return data
        n_graphs = int(graphs.batch_size[0])
        data["graphs", "total_charge"] = torch.full(
            (n_graphs,), self.charge, dtype=pos.dtype, device=pos.device
        )
        return data
