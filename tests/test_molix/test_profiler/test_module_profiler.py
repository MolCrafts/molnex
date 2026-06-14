"""Tests for ModuleProfiler, focused on the submodule breakdown enhancement."""

from __future__ import annotations

import torch.nn as nn
from tensordict import TensorDict

from molix.profiler import MockBatch, ModuleProfiler


class _Composite(nn.Module):
    """Tiny encoder-shaped model: a named child + a ModuleList of blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(3, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, batch: TensorDict) -> TensorDict:
        x = self.embed(batch["atoms", "pos"])
        for blk in self.blocks:
            x = blk(x)
        batch["atoms", "node_features"] = x.unsqueeze(1)
        return batch


def _loss(out, batch=None):
    return out["atoms", "node_features"].sum()


def _factory():
    return MockBatch(n_atoms=64, n_edges=128, n_graphs=4, atomic_numbers=7, device="cpu", seed=0)


def test_run_basic_component_breakdown():
    prof = ModuleProfiler(_Composite(), loss_fn=_loss, device="cpu")
    result = prof.run(_factory(), n_steps=8, n_warmup=2)
    assert result.forward_ms.mean_ms > 0
    assert result.backward_ms is not None
    assert result.n_params > 0
    assert result.submodule_table is None  # off by default


def test_submodule_breakdown_expands_modulelist():
    prof = ModuleProfiler(_Composite(), loss_fn=_loss, device="cpu")
    result = prof.run(_factory(), n_steps=8, n_warmup=2, submodules=True)
    table = result.submodule_table
    assert table is not None
    # named child shows up, and the ModuleList is expanded into its entries
    assert "embed" in table
    assert "blocks.0" in table
    assert "blocks.1" in table
    assert "Σ captured" in table


def test_submodule_breakdown_none_without_children():
    """A leaf module (no named children) yields no submodule table."""
    prof = ModuleProfiler(nn.Identity(), loss_fn=None, device="cpu")
    # Identity ignores the batch; use run_fn-free path via a wrapper is overkill —
    # just assert the breakdown helper returns None for a childless module.
    assert prof._submodule_breakdown(nn.Identity(), _factory(), 2, 1, False) is None
