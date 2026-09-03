"""Fixed-capacity GTO k-space grid — shape, weights, energy, 0 breaks, guards.

Living documentation for ``KSpaceGrid``: the periodic GTO branch runs on an
integer half-box whose shape is sized **once** and never depends on the cell,
so ``torch.compile`` sees one graph with zero breaks, and an over-capacity cell
is refused (eager ``RuntimeError``) or poisoned (in-graph all-NaN weights) but
**never silently truncated**.

What is pinned here:

* the box shape for a 10 Å cubic cell at ``σ = 1.5``, ``l_max = 1``
  (``k_cutoff = 1.8467166200173746`` Å⁻¹) and ``capacity_factor = 1.2``;
* the ``{0.0, 2.0}`` weights and the 46 live half-sphere k-vectors they select;
* four fp64 Hartree energies (periodic / isolated × self-interaction off / on);
* bitwise equality of the ``(3, 3)`` and ``(1, 3, 3)`` cell inputs;
* ``explain`` → (1 graph, 0 breaks) on the periodic branch of both GTO modules;
* the 2.0×-cell overflow: NaN weights from ``vectors``, ``RuntimeError`` naming
  both ``KSpaceGrid`` and ``capacity_factor`` from ``check_capacity``, and no
  NaN at 1.5× (``floor(1.5 · 2.939) = 4 = N_c`` is still inside the box).

Goldens were captured on the pre-change tree (molnex @ ``feat/oplsaa-parameterizer``,
2026-09-03) with the eager ``half_sphere_kvectors`` path that this grid replaces
— see ``_GOLDEN`` provenance below. Public API only; no mace / e3nn /
graph_longrange / ASE import and no subprocess.

    PYTHONPATH=src python regressions/gto-kspace-fixed-grid.py
"""

from __future__ import annotations

import torch

from molix import config
from molix.compile import Compiler
from molpot.potentials.gto import GTOFieldFeatures, GTOHartreeEnergy, KSpaceGrid, kspace_cutoff

config.set_precision("fp64")

# --- goldens -----------------------------------------------------------------
# Provenance: molnex @ feat/oplsaa-parameterizer, 2026-09-03, CPU fp64,
#   PYTHONPATH=src python regressions/gto-kspace-fixed-grid.py
# captured on the PRE-change tree, i.e. from the eager ``half_sphere_kvectors``
# reciprocal path that ``KSpaceGrid`` replaces. Nothing third-party was run:
# the numbers are this repo's own pre-refactor output, hard-coded on purpose.
_SIGMA = 1.5
_L_MAX = 1
_K_CUTOFF = 1.8467166200173746  # = kspace_cutoff(1.5, 1) Å⁻¹
_CAPACITY_FACTOR = 1.2
_CAPACITY = (4, 4, 4)
_N_GRID = 364  # ((2·4+1)³ − 1) / 2 half-box rows
_N_LIVE = 46  # rows with weight 2.0, i.e. 0 < |k| < k_cutoff
_E_PERIODIC_NO_SELF = -0.6945993771741041  # eV
_E_PERIODIC_SELF = 0.06349227817473702  # eV
_E_ISOLATED_NO_SELF = -0.6916466177417959  # eV
_E_ISOLATED_SELF = 0.06644503760704523  # eV
_ATOL = 1.0e-10  # eV

_CELL = torch.eye(3, dtype=torch.float64) * 10.0
_POS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.4, 0.3, -0.2],
        [-0.9, 1.6, 0.7],
        [2.1, -1.1, 1.3],
    ],
    dtype=torch.float64,
)
# ``ir_mul`` order (q, μ_y, μ_z, μ_x) in e / e·Å.
_FEATS = torch.tensor(
    [
        [0.4, 0.20, -0.10, 0.05],
        [-0.3, -0.15, 0.22, -0.08],
        [-0.1, 0.05, 0.12, 0.18],
        [0.0, -0.10, -0.24, -0.15],
    ],
    dtype=torch.float64,
)
_BATCH = torch.zeros(4, dtype=torch.long)


def main() -> None:
    assert abs(kspace_cutoff(_SIGMA, _L_MAX) - _K_CUTOFF) < 1e-15

    # (a) the first bind sizes the box from the lattice norms, once.
    grid = KSpaceGrid(k_cutoff=_K_CUTOFF, capacity_factor=_CAPACITY_FACTOR)
    grid.bind(_CELL)
    assert grid.capacity == _CAPACITY, grid.capacity
    assert tuple(grid.n.shape) == (_N_GRID, 3), tuple(grid.n.shape)

    # (b) weights are 0/2 multiplicative masks and the denominator is guarded.
    kvec, k_sq_guarded, weights = grid.vectors(_CELL)
    assert set(weights.unique().tolist()) <= {0.0, 2.0}, weights.unique().tolist()
    n_live = int((weights == 2.0).sum())
    assert n_live == _N_LIVE, n_live
    assert bool((k_sq_guarded > 0).all())
    assert kvec.shape == (_N_GRID, 3), kvec.shape

    # (c) fp64 energies match the pre-change eager path.
    hartree = GTOHartreeEnergy(sigma=_SIGMA, l_max=_L_MAX, include_self_interaction=False)
    hartree_self = GTOHartreeEnergy(sigma=_SIGMA, l_max=_L_MAX, include_self_interaction=True)
    e_periodic = hartree(_FEATS, _POS, _BATCH, cell=_CELL, num_graphs=1)
    e_periodic_self = hartree_self(_FEATS, _POS, _BATCH, cell=_CELL, num_graphs=1)
    e_isolated = hartree(_FEATS, _POS, _BATCH, num_graphs=1)
    e_isolated_self = hartree_self(_FEATS, _POS, _BATCH, num_graphs=1)
    d_periodic = abs(float(e_periodic.reshape(-1)[0]) - _E_PERIODIC_NO_SELF)
    d_periodic_self = abs(float(e_periodic_self.reshape(-1)[0]) - _E_PERIODIC_SELF)
    d_isolated = abs(float(e_isolated.reshape(-1)[0]) - _E_ISOLATED_NO_SELF)
    d_isolated_self = abs(float(e_isolated_self.reshape(-1)[0]) - _E_ISOLATED_SELF)
    assert d_periodic < _ATOL, f"periodic dE={d_periodic} eV"
    assert d_periodic_self < _ATOL, f"periodic (self) dE={d_periodic_self} eV"
    assert d_isolated < _ATOL, f"isolated dE={d_isolated} eV"
    assert d_isolated_self < _ATOL, f"isolated (self) dE={d_isolated_self} eV"

    # (d) a (1, 3, 3) batch cell is bitwise identical to the (3, 3) cell.
    features = GTOFieldFeatures(
        density_sigma=_SIGMA,
        density_lmax=_L_MAX,
        feature_lmax=_L_MAX,
        feature_widths=[1.5, 3.0],
    )
    batched_cell = _CELL.unsqueeze(0)
    assert torch.equal(e_periodic, hartree(_FEATS, _POS, _BATCH, cell=batched_cell, num_graphs=1))
    f_flat = features(_FEATS, _POS, _BATCH, cell=_CELL, num_graphs=1)
    f_batched = features(_FEATS, _POS, _BATCH, cell=batched_cell, num_graphs=1)
    assert torch.equal(f_flat, f_batched)

    # (e) the periodic branch of both modules is one graph with zero breaks.
    for module in (hartree, features):
        module.bind_cell(_CELL)
        explanation = Compiler.explain(module, _FEATS, _POS, _BATCH, cell=_CELL, num_graphs=1)
        name = type(module).__name__
        assert explanation.graph_count == 1, f"{name}: {explanation.graph_count} graphs"
        assert explanation.graph_break_count == 0, f"{name}: {explanation.graph_break_count} breaks"

    # (f) over-capacity poisons in-graph and raises in eager — never truncates.
    # floor(2 · 2.939) = 5 > 4 overflows; 1.5× (floor(4.41) = 4 = N_c) does not.
    _, _, weights_2x = grid.vectors(_CELL * 2.0)
    assert bool(torch.isnan(weights_2x).all())
    _, _, weights_1p5x = grid.vectors(_CELL * 1.5)
    assert not bool(torch.isnan(weights_1p5x).any())
    try:
        grid.check_capacity(_CELL * 2.0)
    except RuntimeError as exc:
        message = str(exc)
        assert "KSpaceGrid" in message, message
        assert "capacity_factor" in message, message
    else:
        raise AssertionError("check_capacity accepted a 2.0x over-capacity cell")

    print(
        f"gto-kspace-fixed-grid: ok  M={_N_GRID}  live={n_live}  "
        f"dE={d_periodic:.3e} eV  breaks=0  overflow=NaN+RuntimeError"
    )


if __name__ == "__main__":
    main()
