"""PMECalculator: the fixed-cell mesh cache never hides the cell derivative."""

import torch

from molpot.potentials.elec import CoulombPotential, PMECalculator
from tests.test_molpot.test_elec.conftest import periodic_neighbor_list

DTYPE = torch.float64


def _cscl():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=DTYPE)
    charges = torch.tensor([[-1.0], [1.0]], dtype=DTYPE)
    return positions, charges, torch.eye(3, dtype=DTYPE)


def _strain_gradient(calc: PMECalculator) -> torch.Tensor:
    """``dE/dstrain`` at zero strain (the virial), through cell and positions."""
    positions, charges, cell = _cscl()
    pairs, shifts, _ = periodic_neighbor_list(positions, cell, 0.8, full_list=False, periodic=True)
    strain = torch.zeros(3, 3, dtype=DTYPE, requires_grad=True)
    strained_pos = positions + positions @ strain.T
    strained_cell = cell + cell @ strain.T
    vectors = (
        strained_pos[pairs[:, 1]] - strained_pos[pairs[:, 0]] + shifts.to(DTYPE) @ strained_cell
    )
    potentials = calc(
        charges=charges,
        cell=strained_cell,
        positions=strained_pos,
        neighbor_indices=pairs,
        neighbor_distances=vectors.norm(dim=-1),
    )
    return torch.autograd.grad((potentials * charges).sum(), strain)[0]


def _calculator() -> PMECalculator:
    calc = PMECalculator(CoulombPotential(smearing=0.16), mesh_spacing=0.05)
    return calc.to(DTYPE)


def test_warm_cache_does_not_change_the_strain_gradient():
    """A prior call at the same cell values must not serve a mesh without strain history."""
    positions, charges, cell = _cscl()
    pairs, _, distances = periodic_neighbor_list(
        positions, cell, 0.8, full_list=False, periodic=True
    )
    warm = _calculator()
    warm(
        charges=charges,
        cell=cell,
        positions=positions,
        neighbor_indices=pairs,
        neighbor_distances=distances,
    )
    torch.testing.assert_close(_strain_gradient(warm), _strain_gradient(_calculator()))


def test_constant_cell_reuses_the_mesh(monkeypatch):
    """Fixed-cell calls without autograd keep the cache (the MD fast path)."""
    positions, charges, cell = _cscl()
    pairs, _, distances = periodic_neighbor_list(
        positions, cell, 0.8, full_list=False, periodic=True
    )
    calc = _calculator()
    updates: list[torch.Tensor] = []
    original = calc.kspace_filter.update

    def counting(cell, ns_mesh=None):
        updates.append(cell)
        return original(cell, ns_mesh)

    monkeypatch.setattr(calc.kspace_filter, "update", counting)
    kwargs = dict(neighbor_indices=pairs, neighbor_distances=distances)
    calc(charges=charges, cell=cell, positions=positions, **kwargs)
    calc(charges=charges, cell=cell.clone(), positions=positions, **kwargs)
    assert len(updates) == 1
    calc(charges=charges, cell=cell.clone().requires_grad_(True), positions=positions, **kwargs)
    assert len(updates) == 2
