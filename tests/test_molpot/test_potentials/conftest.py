"""Two four-atom chains in one batch, and the molpy oracle every potential is checked against.

The oracle is molrs's own kernels, built per explicit term with
``mp.ff.compile.ExplicitTerms(category, style, atoms, **params).compile()`` from
the same instances and parameters and pushed into one ``mp.ff.potential.Potentials``:
no force-field typing sits between the two sides.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
import torch
from tensordict import TensorDict

#: Two non-planar chains, every angle and dihedral generic.
POS = np.array(
    [
        [1.2, -0.4, 0.3],
        [0.0, 0.0, 0.0],
        [-0.2, 1.5, 0.1],
        [0.9, 2.1, -0.8],
        [6.1, 0.2, -0.3],
        [5.0, 0.0, 0.4],
        [4.6, 1.4, 0.0],
        [5.3, 2.4, 0.9],
    ],
    dtype=np.float64,
)
BONDS = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
ANGLES = [(0, 1, 2), (1, 2, 3), (4, 5, 6), (5, 6, 7)]
QUADS = [(0, 1, 2, 3), (4, 5, 6, 7)]


def chain_batch() -> TensorDict:
    """The two chains as a collated batch (``float64`` positions)."""

    def block(rows: list[tuple[int, ...]]) -> TensorDict:
        cols = list(zip(*rows, strict=True))
        names = ("atomi", "atomj", "atomk", "atoml")
        return TensorDict(
            {names[c]: torch.tensor(cols[c]) for c in range(len(cols))},
            batch_size=[len(rows)],
        )

    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor(POS),
                    "batch": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                },
                batch_size=[8],
            ),
            "graphs": TensorDict({}, batch_size=[2]),
            "bonds": block(BONDS),
            "angles": block(ANGLES),
            "propers": block(QUADS),
            "impropers": block(QUADS),
        },
        batch_size=[],
    )


def oracle(*terms: tuple) -> tuple[float, np.ndarray]:
    """``(energy, forces)`` of explicit molrs terms pushed into one ``Potentials``.

    Each term is ``(category, style, rows, params)`` with ``params`` the keyword
    arguments of :class:`molpy.ff.compile.ExplicitTerms` (per-term values as
    lists, style settings as scalars, ``charges=`` per atom).
    """
    pots = mp.ff.potential.Potentials()
    for category, style, rows, params in terms:
        pots.push(mp.ff.compile.ExplicitTerms(category, style, rows, **params).compile())
    energy, forces = pots.calc_energy_forces(POS.reshape(-1))
    return float(energy), np.asarray(forces)


def energy_and_forces(potential, batch: TensorDict, **params) -> tuple[float, np.ndarray]:
    """Total energy and ``-dE/dpos`` of a molpot potential on ``batch``."""
    pos = batch["atoms", "pos"].clone().requires_grad_(True)
    probe = batch.clone()
    probe["atoms", "pos"] = pos
    energy = potential(probe, **params).sum()
    (grad,) = torch.autograd.grad(energy, pos)
    return float(energy), (-grad).numpy()
