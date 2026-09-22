"""``ProfileIndex`` — renumbering a batch's profiles into the graph axis.

Mirrors ``src/molix/data/profile.py``. The point of the column is that a
per-profile reduction has a static buffer to reduce into; these assert the two
properties that makes true — density over the batch, and a dead slot that can
never join a live profile.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data import ProfileIndex


def _batch(codes, mask=None, column="row"):
    graphs = {column: torch.tensor(codes, dtype=torch.float64)}
    if mask is not None:
        graphs["mask"] = torch.tensor(mask)
    return TensorDict(
        {
            "atoms": TensorDict({"pos": torch.zeros(1, 3)}, batch_size=[1]),
            "graphs": TensorDict(graphs, batch_size=[len(codes)]),
        },
        batch_size=[],
    )


class TestProfileIndex:
    """The batch-local dense profile index."""

    def test_dataset_wide_codes_become_dense_over_this_batch(self) -> None:
        """A dataset-wide code is dense over the *dataset* — 80 000 for a shard.

        A reduction buffer sized by that is both enormous and dataset-dependent,
        which is what this column exists to avoid: the values it writes index the
        graph axis the padder already fixed.
        """
        batch = _batch([4071.0, 4071.0, 9.0, 4071.0, 9.0])
        ProfileIndex().execute(batch)
        index = batch["graphs", "profile"]
        assert index.dtype == torch.long
        assert int(index.max()) < index.numel()
        # Graphs of one scan share a value; different scans do not.
        assert index[0] == index[1] == index[3]
        assert index[2] == index[4]
        assert index[0] != index[2]

    def test_dead_slots_take_the_sink_index_never_a_live_profile(self) -> None:
        """Index 0 is a live profile; the padder's sink is the last slot.

        A dead graph pointed at 0 would join a real scan's minimum and its mean.
        """
        batch = _batch([7.0, 7.0, 0.0, 0.0], mask=[True, True, False, False])
        ProfileIndex().execute(batch)
        index = batch["graphs", "profile"]
        assert index[2] == index[3] == index.numel() - 1
        assert index[0] == index[1] == 0

    def test_a_batch_without_the_source_column_is_refused(self) -> None:
        """A silently absent profile column would make every graph its own scan."""
        batch = TensorDict(
            {
                "atoms": TensorDict({"pos": torch.zeros(1, 3)}, batch_size=[1]),
                "graphs": TensorDict({"num_atoms": torch.ones(2)}, batch_size=[2]),
            },
            batch_size=[],
        )
        with pytest.raises(KeyError, match="no 'row' column"):
            ProfileIndex().execute(batch)

    def test_a_plain_dict_is_refused(self) -> None:
        """The write would land in a container the caller does not hold."""
        with pytest.raises(TypeError, match="needs the batch"):
            ProfileIndex().execute({"graphs": {}})

    def test_the_task_id_names_both_columns(self) -> None:
        """It is a cache key: two spellings produce different batches."""
        assert ProfileIndex().task_id != ProfileIndex(source="molecule_id").task_id

    def test_the_default_source_is_the_row_not_a_group_code(self) -> None:
        """The default decides what a "profile" means, so it is asserted, not assumed.

        It read ``graphs.group`` before, which the Class-I path wrote as a constant
        ``0.0``: every conformer of every molecule collapsed into profile 0, and a
        profile-relative loss then aligned to a minimum taken across different
        molecules. Nothing raised — the column was there, it just meant nothing.
        """
        assert ProfileIndex().source == "row"
        rowed = _batch([0.0, 0.0, 1.0, 1.0, 1.0])
        ProfileIndex().execute(rowed)
        index = rowed["graphs", "profile"]
        assert index[0] == index[1]
        assert index[2] == index[3] == index[4]
        assert index[0] != index[2]

    def test_a_constant_source_column_collapses_every_graph_into_one_profile(self) -> None:
        """What the old default did, kept as the shape of the failure.

        A constant code is not an error to ``ProfileIndex`` — renumbering it is
        correct — so this is the one place the consequence is written down: with
        one profile, a profile-relative objective subtracts a minimum taken over
        unrelated molecules and its residual is their absolute-energy spread.
        """
        batch = _batch([0.0, 0.0, 0.0, 0.0], column="group")
        ProfileIndex(source="group").execute(batch)
        assert int(batch["graphs", "profile"].unique().numel()) == 1
