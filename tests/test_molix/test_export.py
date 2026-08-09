"""Tests for molix.export.Exporter — make_fx → torch.export → AOTInductor ``.pt2``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from molix.export import Exporter


@pytest.fixture
def small_mlp() -> nn.Sequential:
    """A small 3-layer MLP (10 -> 32 -> 16 -> 5), eval mode, seeded."""
    torch.manual_seed(42)
    m = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )
    m.eval()
    return m


class _ForceModel(nn.Module):
    """Energy MLP whose forward returns ``F = -dE/dx`` via ``autograd.grad``.

    This is exactly the in-forward-gradient case that defeats a naive
    ``torch.export`` (trainable weights mis-lifted as fake constants) and is
    unblocked by :class:`Exporter`'s ``make_fx`` pretrace.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.requires_grad_(True)
        energy = self.net(pos).sum()
        return -torch.autograd.grad(energy, pos, create_graph=False)[0]


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_export_creates_pt2(tmp_path: Path, small_mlp: nn.Sequential) -> None:
    x = torch.randn(4, 10)
    out = Exporter(small_mlp).export((x,), tmp_path / "m.pt2")
    assert out == tmp_path / "m.pt2"
    assert out.is_file()


def test_export_makes_parent_dirs(tmp_path: Path, small_mlp: nn.Sequential) -> None:
    x = torch.randn(4, 10)
    out = Exporter(small_mlp).export((x,), tmp_path / "nested" / "deep" / "m.pt2")
    assert out.is_file()


# ---------------------------------------------------------------------------
# Loadability + correctness
# ---------------------------------------------------------------------------


def test_load_and_run_matches_eager(tmp_path: Path, small_mlp: nn.Sequential) -> None:
    x = torch.randn(4, 10)
    with torch.no_grad():
        expected = small_mlp(x)
    path = Exporter(small_mlp).export((x,), tmp_path / "m.pt2")
    runner = Exporter.load(path)
    actual = torch.as_tensor(runner(x))
    assert torch.allclose(expected, actual, atol=1e-5)


def test_dynamic_shapes_one_package_varying_batch(tmp_path: Path, small_mlp: nn.Sequential) -> None:
    """One ``.pt2`` with a dynamic batch dim serves multiple sizes (no padding)."""
    x = torch.randn(4, 10)
    b = torch.export.Dim("b", min=2, max=4096)
    path = Exporter(small_mlp).export((x,), tmp_path / "m.pt2", dynamic_shapes=({0: b},))
    runner = Exporter.load(path)
    for n in (2, 8):
        xn = torch.randn(n, 10)
        with torch.no_grad():
            expected = small_mlp(xn)
        assert torch.allclose(expected, torch.as_tensor(runner(xn)), atol=1e-5)


def test_force_model_exports_via_make_fx(tmp_path: Path) -> None:
    """Headline: a module that differentiates inside forward exports and runs,
    forces matching eager — and generalizes to an un-traced size.

    CPU is refused (torch 2.12 miscompiles the baked backward scatter), so this
    runs the real export only on CUDA; on CPU it asserts the explicit raise.
    """
    pos = torch.randn(6, 3)
    n = torch.export.Dim("n", min=2, max=4096)

    if not torch.cuda.is_available():
        model = _ForceModel().eval()
        with pytest.raises(RuntimeError, match="disabled on CPU"):
            Exporter(model).export_pretraced(
                (pos,), tmp_path / "force.pt2", dynamic_shapes=({0: n},)
            )
        return

    model = _ForceModel().cuda().eval()
    pos = pos.cuda()
    ref = model(pos).detach()
    path = Exporter(model).export_pretraced(
        (pos,), tmp_path / "force.pt2", dynamic_shapes=({0: n},)
    )
    runner = Exporter.load(path)

    out = torch.as_tensor(runner(pos))
    assert out.shape == (6, 3)
    assert torch.allclose(out, ref, atol=1e-5)

    pos2 = torch.randn(10, 3, device="cuda")  # different N than traced
    assert torch.allclose(model(pos2).detach(), torch.as_tensor(runner(pos2)), atol=1e-5)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_direct_export_on_force_model_raises_no_fallback() -> None:
    """`export()` does NOT silently fall back to make_fx — a force model raises,
    directing the caller to `export_pretraced()`."""
    model = _ForceModel().eval()
    pos = torch.randn(6, 3)
    with pytest.raises(Exception):  # noqa: B017 — torch.export's own error surfaces
        Exporter(model).export((pos,), "unused.pt2")


def test_non_module_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        Exporter("not_a_module")


def test_non_tuple_inputs_raises_typeerror(tmp_path: Path, small_mlp: nn.Sequential) -> None:
    with pytest.raises(TypeError):
        Exporter(small_mlp).export(torch.randn(4, 10), tmp_path / "m.pt2")
