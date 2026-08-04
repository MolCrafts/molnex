import torch

from molpot.potentials.elec import (
    CoulombPotential,
    EwaldCalculator,
)
from molpot.potentials.elec.tuning.tuner import TuningTimings
from tests.regression.conftest import define_crystal, neighbor_list

DTYPE = torch.float32
DEFAULT_CUTOFF = 4.4


def _supercell(
    pos: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    reps: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tile a unit cell ``reps`` times along each lattice vector (numpy-free)."""
    nx, ny, nz = reps
    images = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = ix * cell[0] + iy * cell[1] + iz * cell[2]
                images.append(pos + shift)
    pos_sc = torch.cat(images, dim=0)
    charges_sc = charges.repeat(nx * ny * nz, 1)
    scale = torch.tensor([nx, ny, nz], dtype=cell.dtype, device=cell.device)
    cell_sc = cell * scale.unsqueeze(1)
    return pos_sc, charges_sc, cell_sc


def test_timer():
    n_repeat_1 = 10
    n_repeat_2 = 100
    pos, charges, cell, _, _ = define_crystal()

    # Enlarge the crystal without ASE: 4×4×4 supercell of the primitive cell.
    pos = pos.to(DTYPE)
    charges = charges.to(DTYPE).reshape(-1, 1)
    cell = cell.to(DTYPE)
    pos, charges, cell = _supercell(pos, charges, cell, (4, 4, 4))

    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )

    calculator = EwaldCalculator(
        potential=CoulombPotential(smearing=1.0),
        lr_wavelength=0.25,
    )

    timing_1 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        n_repeat=n_repeat_1,
    )

    timing_2 = TuningTimings(
        charges=charges,
        cell=cell,
        positions=pos,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        n_repeat=n_repeat_2,
    )

    time_1 = timing_1.forward(calculator)
    time_2 = timing_2.forward(calculator)

    assert time_1 > 0
    assert time_1 * n_repeat_1 < time_2 * n_repeat_2
