"""A batch-local dense index for the conformer profiles a batch happens to hold.

A *profile* is a set of graphs that must be compared only against each other: the
24 constrained conformers of one THEMol torsion scan, or the frames of one
relaxation. Their absolute energies are meaningless — a THEMol record publishes
-503 415.87 kcal/mol against a ~10 kcal/mol barrier — so a profile objective
compares each conformer to its own profile's minimum and lets a per-profile
constant offset go free.

Doing that needs a per-profile reduction, which needs somewhere to reduce *into*.
``graphs.group`` cannot be that index: it is dense over the whole dataset (about
80 000 codes for one shard), so a buffer sized by it is both enormous and
dataset-dependent. This task renumbers whichever profiles a batch holds into
``[0, max_graphs)``, so the reduction buffer is the graph axis the padder already
fixed — a static shape, no host sync in the step, and one graph.

It is a *host-side* task, mounted beside
:class:`~molix.data.charge.TotalCharge`, precisely because the renumbering is the
part that cannot be traced: it is a ``torch.unique`` over the batch's own codes.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.task import BatchTask
from molix.frame import as_tensor_frame

__all__ = ["ProfileIndex"]


class ProfileIndex(BatchTask):
    """Renumber ``graphs.<source>`` into a batch-local ``graphs.profile``.

    Dead graph slots are sent to the **last** index rather than to 0, which is a
    live profile. The padder reserves that slot for the ghost atoms' sink
    (``max_graphs`` = live graphs + 1), and a batch holds at most that many
    distinct profiles minus the sink, so the last index is never a live
    profile's.

    Args:
        source: The ``graphs`` column holding the dataset-wide profile code.
            ``"group"`` is what the Class-I path writes (one code per scan).
        out: The column to write.

    Example::

        DataModule(..., batch_tasks=[neighbor_list, TotalCharge(), ProfileIndex()])
    """

    def __init__(self, source: str = "group", out: str = "profile") -> None:
        self.source = str(source)
        self.out = str(out)

    @property
    def task_id(self) -> str:
        """Cache-key identity — both column names, since both shape the batch."""
        return f"profile_index:{self.source}->{self.out}"

    def execute(self, data: TensorDict | dict) -> TensorDict | dict:
        """Write ``graphs.<out>``, a dense profile index over this batch's graphs.

        Args:
            data: Collated (and usually padded) batch whose ``graphs`` namespace
                is sized ``[B]`` and carries :attr:`source`, plus ``mask`` when
                it has been padded.

        Returns:
            The same batch, mutated in place, with an integer ``(B,)`` column in
            ``[0, B)``: live graphs of one profile share a value, and every dead
            slot holds ``B - 1``.

        Raises:
            TypeError: *data* is a plain ``dict`` — the write would land in a
                container the caller does not hold.
            KeyError: ``graphs`` is absent, or carries no :attr:`source` column.
            IndexError: ``graphs`` carries no batch dimension.
        """
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                f"ProfileIndex.execute needs the batch (a TensorDict), got {type(data).__name__}; "
                "a write into a plain dict would not reach the caller's batch"
            )
        frame = as_tensor_frame(data)
        graphs = frame["graphs"]
        if not graphs.batch_size:
            raise IndexError(
                "ProfileIndex: graphs has no batch dimension (batch_size=[] rather than [B]); "
                "the graph count is what the profile index is dense over"
            )
        n_graphs = int(graphs.batch_size[0])
        codes = graphs.get(self.source)
        if codes is None:
            raise KeyError(
                f"ProfileIndex: graphs carries no {self.source!r} column. It is the "
                "dataset-wide profile code a batch's profiles are renumbered from — the "
                "Class-I path writes it as a per-frame meta scalar."
            )
        codes = codes.reshape(-1)
        mask = graphs.get("mask")
        live = torch.ones(n_graphs, dtype=torch.bool, device=codes.device)
        if mask is not None:
            live = mask.reshape(-1).to(torch.bool)
        # The sink is the last slot, which the padder keeps for the ghosts, so a
        # dead graph can never share a profile with a live one.
        index = torch.full((n_graphs,), n_graphs - 1, dtype=torch.long, device=codes.device)
        if bool(live.any()):
            index[live] = torch.unique(codes[live], return_inverse=True)[1]
        graphs.set(self.out, index)
        return data
