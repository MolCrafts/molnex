"""Tests for molzoo.mace.utils — official archive helpers."""

from __future__ import annotations

import torch

from molzoo.mace.utils import e3nn2cueq


class TestE3nn2cueq:
    def test_state_dict_mapping_passes_through(self) -> None:
        """A plain tensor dict is returned as ``state`` with an empty config."""
        weight = torch.arange(4, dtype=torch.float64)
        archive = e3nn2cueq({"linear.weight": weight})
        assert set(archive) == {"state", "config"}
        assert archive["config"] == {}
        assert torch.equal(archive["state"]["linear.weight"], weight)
