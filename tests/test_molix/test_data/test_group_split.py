"""Tests for group_split_indices."""

from __future__ import annotations

from molix.data.group_split import group_split_indices


class TestGroupSplitIndices:
    def test_no_leakage_801010(self):
        # 10 mols × 2 confs
        ids = [f"m{m}" for m in range(10) for _ in range(2)]
        train, val, test = group_split_indices(ids, ratios=(0.8, 0.1, 0.1), seed=0)
        assert len({ids[i] for i in train}) == 8
        assert len({ids[i] for i in val}) == 1
        assert len({ids[i] for i in test}) == 1
        assert set(ids[i] for i in train).isdisjoint(ids[i] for i in val)
        assert set(ids[i] for i in train).isdisjoint(ids[i] for i in test)
