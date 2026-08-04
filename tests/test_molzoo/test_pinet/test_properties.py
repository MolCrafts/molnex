"""Smoke tests for PiNetDipole / PiNetPolarizability façades.

Detailed head physics lives under tests/test_molpot/test_heads/; these only
assert the zoo façade still wires the encoder → head path after the package split.
"""

from __future__ import annotations

import torch

from molzoo.pinet import PiNet, PiNetDipole, PiNetPolarizability
from tests.conftest import make_graph_batch


def _graph():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.2, 1.0, 0.1],
            [1.0, 1.1, -0.1],
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
    )
    batch = torch.zeros(4, dtype=torch.long)
    return make_graph_batch(
        pos,
        z,
        edge_index,
        batch,
        graphs={"total_charge": torch.tensor([0.0], dtype=torch.float32)},
    )


def _encoder() -> PiNet:
    torch.manual_seed(0)
    return PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=3,
    )


def test_dipole_ac_runs():
    model = PiNetDipole(encoder=_encoder(), variant="ac")
    out = model(_graph())
    assert "dipole" in out or "mu" in out or any("charge" in k for k in out)


def test_polarizability_localchi_runs():
    model = PiNetPolarizability(encoder=_encoder(), variant="localchi", atom_types=[1, 6, 7, 8])
    out = model(_graph())
    assert len(out) > 0
