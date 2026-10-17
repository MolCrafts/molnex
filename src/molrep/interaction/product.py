"""Irreps-string helpers shared by the message-passing blocks and the readouts.

- ``irreps_from_l_max``: ``"Cx0e+Cx1o+…"`` with one uniform multiplicity.
- ``sh_irreps_from_l_max``: the spherical-harmonic irreps ``"1x0e+1x1o+…"``.

The tensor products themselves live with their owners:
:class:`molzoo.mace.layers._CueqChannelwiseBlockMixin` (MACE),
:class:`molzoo.allegro.layers.Contracter` (Allegro).
"""

from __future__ import annotations


def irreps_from_l_max(l_max: int, hidden_dim: int) -> str:
    """Generate irreps string with uniform multiplicities for optimal cuEquivariance performance.

    Args:
        l_max: Maximum angular momentum.
        hidden_dim: Uniform multiplicity across all l values.

    Returns:
        Irreps string (e.g., "32x0e+32x1o+32x2e").

    Note:
        Uniform multiplicities enable cuEquivariance's fast uniform_1d method (~5-10x speedup).
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"{hidden_dim}x{l}{parity}")
    return "+".join(irreps_list)


def sh_irreps_from_l_max(l_max: int) -> str:
    """Generate spherical harmonics irreps string from l_max.

    Args:
        l_max: Maximum angular momentum.

    Returns:
        Irreps string (e.g., "1x0e+1x1o+1x2e").
    """
    irreps_list = []
    for l in range(l_max + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"1x{l}{parity}")
    return "+".join(irreps_list)
