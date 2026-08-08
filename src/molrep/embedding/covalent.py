"""Z-indexed covalent-radius lookup table.

A single free function because the table has no natural owning type: it is
published numerical data materialised as a dense ``Z``-indexed tensor so radial
transforms and pair-repulsion terms can index it with raw atomic numbers. Two
consumers today — :class:`molrep.embedding.radial.AgnesiTransform` and
:class:`molpot.potentials.repulsion.ZBLRepulsion`.

The values are inlined rather than read from ``molpy.Element`` deliberately.
This is a constant of nature, not a molpy domain type, and it sits on the import
path of a core ``molrep`` block: sourcing it from molpy would make every
equivariant encoder depend on the compiled molrs extension for 119 floats — and
molpy stores radii in single precision, which is a fidelity loss against the
float64 reference table the MACE foundation checkpoints were fitted with.

Reference:
    Cordero et al. "Covalent radii revisited" Dalton Trans. 2008, 2832-2838.
    https://doi.org/10.1039/B801115J
"""

from __future__ import annotations

import torch

from molix import config

#: Radius used where the source tabulates none — index 0 (not an element) and
#: everything past curium. Matches the reference table byte for byte, which is
#: what a MACE checkpoint's own ``covalent_radii`` buffer carries.
_MISSING_RADIUS = 0.2

# Cordero (2008) covalent radii in Angstrom for Z = 1..96, the full span the
# paper tabulates. Heavier elements fall back to _MISSING_RADIUS.
_CORDERO_RADII: tuple[float, ...] = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
)

#: Largest atomic number a table may be built for.
MAX_Z = 118

_COVALENT_RADII: tuple[float, ...] = (
    (_MISSING_RADIUS,)
    + _CORDERO_RADII
    + (_MISSING_RADIUS,) * (MAX_Z - len(_CORDERO_RADII))
)


def covalent_radii(max_z: int = MAX_Z) -> torch.Tensor:
    """Build a dense ``Z``-indexed covalent-radius table.

    Args:
        max_z: Largest atomic number to include; the table has ``max_z + 1``
            rows so ``table[Z]`` is valid for ``Z`` in ``[0, max_z]``.

    Returns:
        Covalent radii in Angstrom ``(max_z + 1,)``; index 0 holds a
        dummy-atom placeholder.

    Raises:
        ValueError: If ``max_z`` is below 1 or beyond the tabulated range.
    """
    if not 1 <= max_z <= MAX_Z:
        raise ValueError(f"max_z must be in [1, {MAX_Z}], got {max_z}")
    return torch.tensor(_COVALENT_RADII[: max_z + 1], dtype=config.ftype)
