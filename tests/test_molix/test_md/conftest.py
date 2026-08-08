"""Shared fixtures for the MD test package: a tiny PiNet system.

Imported by package path (``from tests.test_molix.test_md.conftest import …``)
per the repo test-layout rule — no free-floating helper modules.
"""

import torch

_DEVICE = torch.device("cpu")


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
