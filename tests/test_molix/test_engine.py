"""Tests for the generic ``pair_style molnex`` interface (molix.engine).

Covers the Python export side end-to-end: adapter registry + flat-convention
calling, the molnex ``TensorDict`` construction (edge convention!), and the
AOTI export round-trip that stamps the ``lammps`` meta block a ``pair_style
molnex`` C++ run reads. The C++ pair style itself is exercised by an actual
LAMMPS run (see ``interface/lammps/README.md``), not here.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from molix.engine import (
    EngineAdapter,
    EngineForward,
    FlatTensorAdapter,
    MolnexTensorDictAdapter,
    export_for_lammps,
)
from molix.export import Exporter


class _DummyTDPotential(nn.Module):
    """Mimics a molnex potential: nested-TensorDict in, ``{energy, forces}`` out."""

    def forward(self, batch, *, compute_forces: bool = False):
        pos = batch["atoms", "pos"]
        bd = batch["edges", "edge_dist"]
        out = {"energy": (bd**2).sum().reshape(1), "atomic_energy": torch.zeros(pos.shape[0])}
        if compute_forces:
            out["forces"] = torch.zeros_like(pos)
        return out


class _FlatEFPotential(nn.Module):
    """Exportable flat-convention potential with analytic forces (no functorch)."""

    def forward(self, Z, pos, edge_index):
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        d = pos[tgt] - pos[src]
        r = d.norm(dim=-1).clamp(min=1e-6)
        e = ((r - 1.0) ** 2).sum().reshape(1)
        g = (2 * (r - 1.0) / r).unsqueeze(-1) * d
        f = torch.zeros_like(pos)
        f.index_add_(0, src, g)
        f.index_add_(0, tgt, -g)
        return e, f


@pytest.fixture
def graph():
    Z = torch.tensor([1, 6, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    return Z, pos, edge_index


def test_registry_has_builtin_adapters():
    assert "molnex-tensordict" in EngineAdapter.names()
    assert "flat" in EngineAdapter.names()
    assert isinstance(EngineAdapter.from_name("flat"), FlatTensorAdapter)


def test_unknown_adapter_raises():
    with pytest.raises(ValueError, match="unknown adapter"):
        EngineAdapter.from_name("does-not-exist")


def test_tensordict_adapter_edge_convention(graph):
    """edge_diff = pos[target] - pos[source]; single-graph batch metadata."""
    Z, pos, edge_index = graph
    batch = MolnexTensorDictAdapter().build_inputs(_DummyTDPotential(), Z, pos, edge_index)
    expected = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    assert torch.allclose(batch["edges", "edge_diff"], expected)
    assert torch.allclose(
        batch["edges", "edge_dist"], expected.norm(dim=-1).clamp(min=1e-6)
    )
    assert (batch["atoms", "batch"] == 0).all()
    assert batch["graphs", "num_atoms"].item() == 3


def test_lammps_forward_shapes_tensordict(graph):
    Z, pos, edge_index = graph
    e, f = EngineForward(_DummyTDPotential(), "molnex-tensordict")(Z, pos, edge_index)
    assert e.shape == ()                       # scalar total energy
    assert f.shape == (3, 3)


def test_lammps_forward_shapes_flat(graph):
    Z, pos, edge_index = graph
    e, f = EngineForward(_FlatEFPotential(), "flat")(Z, pos, edge_index)
    assert e.shape == ()
    assert f.shape == (3, 3)


def test_export_for_lammps_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="species"):
        export_for_lammps(_FlatEFPotential(), tmp_path / "x", species=[], cutoff=4.5)
    with pytest.raises(ValueError, match="units"):
        export_for_lammps(
            _FlatEFPotential(), tmp_path / "x", species=[1], cutoff=4.5, units="lj"
        )


def test_export_for_lammps_cpu_refused(tmp_path):
    """CPU force export is disabled (torch 2.12 miscompiles the backward scatter)."""
    with pytest.raises(RuntimeError, match="disabled on CPU"):
        export_for_lammps(
            _FlatEFPotential(),
            tmp_path / "x",
            species=[1, 6, 8],
            cutoff=4.5,
            units="real",
            adapter="flat",
            device="cpu",
        )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="force export requires CUDA")
def test_export_for_lammps_roundtrip(tmp_path):
    """AOTI export writes the lammps meta block and a dynamic-shape ``.pt2``."""
    outdir = export_for_lammps(
        _FlatEFPotential(),
        tmp_path / "flat_export",
        species=[8, 1, 6],          # deliberately unsorted → stored sorted
        cutoff=4.5,
        units="real",
        adapter="flat",
        device="cuda",
    )
    meta = json.loads((outdir / "model.meta.json").read_text())
    assert (outdir / "model.pt2").exists()
    lm = meta["lammps"]
    assert lm["cutoff"] == 4.5
    assert lm["units"] == "real"
    assert lm["species"] == [1, 6, 8]
    assert lm["adapter"] == "flat"
    assert lm["inputs"] == ["Z", "pos", "edge_index"]
    assert lm["outputs"] == ["energy", "forces"]

    # the exported .pt2 serves N/E different from the trace (MD needs this)
    runner = Exporter.load(outdir / "model.pt2")
    Z = torch.tensor([1, 6, 8, 1], device="cuda")
    pos = torch.randn(4, 3, device="cuda")
    ei = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], device="cuda")
    e, f = runner(Z, pos, ei)
    assert f.shape == (4, 3)
