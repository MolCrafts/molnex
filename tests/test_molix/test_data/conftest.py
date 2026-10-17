"""Shared fixtures of the data tests: one short chain, several frames per row."""

from __future__ import annotations

from tests.conftest import frame

BOX = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]


def record(origin: float, *, frames: int = 1) -> tuple:
    """A C-H-H-O chain shifted to *origin*; *frames* conformers share one dataset row."""
    out = []
    for conformer in range(frames):
        pos = [[origin + atom * 1.0, conformer * 0.1, 0.0] for atom in range(4)]
        out.append(
            frame(
                Z=[6, 1, 1, 8],
                pos=pos,
                bonds=([0, 1, 2], [1, 2, 3]),
                terms={
                    "angles": {"atomi": [0], "atomj": [1], "atomk": [2]},
                    "propers": {"atomi": [0], "atomj": [1], "atomk": [2], "atoml": [3]},
                },
                energy=1.5 + origin,
                box=BOX,
                task=0,
            )
        )
    return tuple(out)
