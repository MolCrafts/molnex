"""Tests for FukuiEquilibration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molix import config
from molpot.potentials.gto.fukui import FukuiEquilibration


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestFukuiEquilibration:
    """ASE multiplicity M=2S+1; targets (Q±(M-1))/2 per spin channel."""

    def test_closed_shell_zero_charge_is_a_no_op_when_already_neutral(self):
        """Q=0, M=1, monopoles already 0 → density unchanged."""
        eq = FukuiEquilibration()
        density = torch.zeros(2, 2, 4, dtype=torch.float64)
        fukui = torch.ones(2, 2, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = eq(
            density,
            fukui,
            batch,
            total_charge=torch.zeros(1, dtype=torch.float64),
            total_spin=torch.ones(1, dtype=torch.float64),
        )
        assert torch.equal(out, density)

    def test_cation_singlet_splits_charge_equally_across_spins(self):
        """Q=1, M=1 → n_up = n_down = 0.5."""
        eq = FukuiEquilibration()
        density = torch.zeros(2, 2, 4, dtype=torch.float64)
        fukui = torch.ones(2, 2, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = eq(
            density,
            fukui,
            batch,
            total_charge=torch.ones(1, dtype=torch.float64),
            total_spin=torch.ones(1, dtype=torch.float64),
        )
        assert out[:, 0, 0].sum().item() == pytest.approx(0.5)
        assert out[:, 1, 0].sum().item() == pytest.approx(0.5)
        assert torch.equal(out[..., 1:], torch.zeros(2, 2, 3, dtype=torch.float64))

    def test_neutral_doublet_puts_unpaired_electron_on_spin_up(self):
        """Q=0, M=2 → n_up=0.5, n_down=-0.5 (S_paper = M-1 = 1)."""
        eq = FukuiEquilibration()
        density = torch.zeros(2, 2, 4, dtype=torch.float64)
        fukui = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = eq(
            density,
            fukui,
            batch,
            total_charge=torch.zeros(1, dtype=torch.float64),
            total_spin=torch.full((1,), 2.0, dtype=torch.float64),
        )
        assert out[:, 0, 0].sum().item() == pytest.approx(0.5)
        assert out[:, 1, 0].sum().item() == pytest.approx(-0.5)

    def test_fukui_normalizes_per_graph(self):
        """Each spin channel of fukui sums to 1 inside a graph after norm."""
        eq = FukuiEquilibration()
        density = torch.zeros(3, 2, 4, dtype=torch.float64)
        fukui = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], dtype=torch.float64)
        batch = torch.tensor([0, 0, 1], dtype=torch.long)
        out = eq(
            density,
            fukui,
            batch,
            total_charge=torch.tensor([1.0, 0.0], dtype=torch.float64),
            total_spin=torch.ones(2, dtype=torch.float64),
        )
        assert out[:2, 0, 0].sum().item() == pytest.approx(0.5)
        assert out[2, 0, 0].item() == pytest.approx(0.0)

    def test_scatter_import_is_compile_safe(self) -> None:
        """``scatter_sum`` hides an ``.item()`` on its ``dim_size=None`` path.

        The equilibration runs inside the compiled charge path, so it must use
        the compile-safe reduction its GTO siblings already use — the same
        assertion as ``test_scatter_import_is_compile_safe`` on
        ``hartree.py`` / ``features.py``.
        """
        path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "molpot"
            / "potentials"
            / "gto"
            / "fukui.py"
        )
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("from molix.F.scatter import")
        ]
        assert lines == ["from molix.F.scatter import scatter_sum_compile_safe"]
