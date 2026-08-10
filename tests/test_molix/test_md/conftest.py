"""Shared fixtures for the MD test package: a tiny PiNet system and a lattice.

Imported by package path (``from tests.test_molix.test_md.conftest import …``)
per the repo test-layout rule — no free-floating helper modules.
"""

import torch

_DEVICE = torch.device("cpu")


def make_cubic_lattice(n_side: int = 3, spacing: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """A simple cubic lattice and its cell — a periodic system with real edges.

    The one owner of the MD suite's periodic fixture (``test_forcefield.py``,
    ``test_neighbors.py`` and ``test_driver.py`` each held or wanted a
    byte-identical copy). ``n_side=4, spacing=3.0`` is the 64-atom, 12 Å cube
    the policy and driver suites run argon in: minimum perpendicular half-width
    6.0 Å, exact coordination shells at 3.0 / 4.2426 / 5.196 Å.

    Args:
        n_side: Atoms per axis; the system holds ``n_side ** 3`` atoms.
        spacing: Lattice constant in Angstrom.

    Returns:
        ``(positions (n_side ** 3, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    grid = torch.arange(n_side, dtype=torch.float64) * spacing
    pos = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * spacing)
    return pos, cell


def make_tiny_potential():
    """A 4-species PiNet potential small enough for eager unit tests."""
    from molzoo.pinet import PiNetPotential

    torch.manual_seed(0)
    return (
        PiNetPotential(
            atom_types=[1, 6, 7, 8],
            r_max=4.0,
            n_basis=3,
            pp_nodes=[8, 8],
            pi_nodes=[8, 8],
            ii_nodes=[8, 8],
            depth=2,
            rank=3,
            hidden_dim=16,
            # Force derivation is fixed at construction since b85d12f (the
            # monomorphic pipelines torch.compile needs); an MD force field
            # cannot ask for it per call.
            compute_forces=True,
        )
        .to(_DEVICE)
        .eval()
    )


def make_pinet_template():
    """A 4-atom collated batch matching :func:`make_tiny_potential`."""
    from tests.conftest import make_graph_batch

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.3, 1.2, 0.2], [1.4, 1.1, -0.1]],
        dtype=torch.float32,
        device=_DEVICE,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long, device=_DEVICE)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
        device=_DEVICE,
    )
    batch = torch.zeros(4, dtype=torch.long, device=_DEVICE)
    return make_graph_batch(pos, z, edge_index, batch)
