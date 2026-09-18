"""Fixtures for the Rondo recipe: one gas-phase butane, collated the real way."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molix import config
from molix.data import TotalCharge
from molix.data.collate import collate_molecules
from tests.conftest import (
    BUTANE_BONDS,
    BUTANE_POSITIONS,
    butane_angles,
    butane_propers,
    frame,
)


def butane_frame():
    """One butane at rest, **no lattice** — Rondo's template is gas-phase.

    The pair fixture in ``tests/conftest.py`` is a periodic box because its
    Coulomb term is Ewald. Rondo declares ``CoulombOpenPotential``, whose
    ``1/r`` sum is bounded by a declared radius rather than a lattice, so its
    fixture is a single isolated molecule.
    """
    angles = butane_angles()
    propers = butane_propers()
    return frame(
        Z=[6] * 4 + [1] * 10,
        pos=np.asarray(BUTANE_POSITIONS, dtype=np.float64),
        bonds=([b[0] for b in BUTANE_BONDS], [b[1] for b in BUTANE_BONDS]),
        terms={
            "angles": {
                "atomi": [a[0] for a in angles],
                "atomj": [a[1] for a in angles],
                "atomk": [a[2] for a in angles],
            },
            "propers": {
                "atomi": [p[0] for p in propers],
                "atomj": [p[1] for p in propers],
                "atomk": [p[2] for p in propers],
                "atoml": [p[3] for p in propers],
            },
        },
        total_charge=0.0,
        energy=-1.0,
    )


@pytest.fixture
def butane_batch() -> TensorDict:
    """The batch a Rondo model reads: collated butane with its own pair list.

    The list radius is the **field's** (``RondoForceField.neighbor_list()``),
    never a number this fixture picks — the model must price exactly the pairs
    the batch carries (ledger finding **F21**).
    """
    from molzoo.rondo import RondoForceField

    batch = collate_molecules([butane_frame()])
    RondoForceField().neighbor_list().execute(batch)
    TotalCharge().execute(batch)
    # The real path casts every float leaf to the project dtype at collate;
    # this fixture calls ``collate_molecules`` directly, so it casts here.
    batch = batch.apply(lambda t: t.to(config.ftype) if t.is_floating_point() else t)
    batch["atoms", "forces"] = torch.zeros_like(batch["atoms", "pos"])
    return batch
