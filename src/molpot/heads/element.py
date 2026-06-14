"""Per-element baseline tables for atomic charges and polarizabilities.

Two opt-in additive baselines that supply per-element prior values for
``q`` (formal-charge convention) and ``α`` (CRC handbook atomic
polarizabilities). The numerical tables are public physical / chemical
data — atomic polarizabilities from the CRC Handbook of Chemistry and
Physics (e.g. 100th ed., Table on Atomic and Molecular Polarizabilities)
and formal-charge values from the standard Pauling/IUPAC convention —
not copyrightable expression. The tables here are values verified
against the upstream LES library at ``github.com/ChengUCB/les``
(``les.module.fixedcharges`` and ``les.module.atomicalpha``); attribution
is preserved via this docstring rather than via package imports so the
test path is dependency-free.

Both modules behave like :class:`molpot.heads.rescale.PerSpeciesScaleShift`
— Z-indexed embedding-style lookup with a single global scaling factor.
The baseline output is meant to be **added** to a learnable head's
prediction (the head learns the *residual*), not to replace it; toggling
the baseline off is equivalent to setting the scaling factor to zero.

Example::

    table = ElementChargeTable(scaling_factor=0.5)
    q_total = head(features) + table(atomic_numbers)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Atomic polarizability table (Bohr³)
# ---------------------------------------------------------------------------
# Values mirror les.module.atomicalpha.alpha_dict (Z=1..86). CRC Handbook
# atomic polarizability values, in Bohr³ (atomic units of volume).
# Conversion to e·Å·V⁻¹·Å⁻¹ uses the LES default factor 0.1481847/14.3996,
# which converts Bohr³ → e·Å²·V⁻¹ in the eV-Å unit system.
ALPHA_DICT_BOHR3: dict[int, float] = {
    1: 4.50,
    2: 1.38,
    3: 164.11,
    4: 37.74,
    5: 20.50,
    6: 11.30,
    7: 7.40,
    8: 5.30,
    9: 3.74,
    10: 2.67,
    11: 162.70,
    12: 71.30,
    13: 60.00,
    14: 37.30,
    15: 25.00,
    16: 19.40,
    17: 14.60,
    18: 11.10,
    19: 290.00,
    20: 169.00,
    21: 120.00,
    22: 98.00,
    23: 84.00,
    24: 78.00,
    25: 63.00,
    26: 56.00,
    27: 50.00,
    28: 49.00,
    29: 47.00,
    30: 38.70,
    31: 50.00,
    32: 40.00,
    33: 30.00,
    34: 28.90,
    35: 21.90,
    36: 16.80,
    37: 319.00,
    38: 197.00,
    39: 162.00,
    40: 121.00,
    41: 106.00,
    42: 86.40,
    43: 80.00,
    44: 65.00,
    45: 58.00,
    46: 26.10,
    47: 55.00,
    48: 49.70,
    49: 70.00,
    50: 52.00,
    51: 43.00,
    52: 37.60,
    53: 35.00,
    54: 27.30,
    55: 401.00,
    56: 273.00,
    57: 213.00,
    58: 204.00,
    59: 196.00,
    60: 190.00,
    61: 185.00,
    62: 180.00,
    63: 175.00,
    64: 160.00,
    65: 159.00,
    66: 157.00,
    67: 156.00,
    68: 153.00,
    69: 151.00,
    70: 142.00,
    71: 148.00,
    72: 109.00,
    73: 88.00,
    74: 74.00,
    75: 65.00,
    76: 57.00,
    77: 51.00,
    78: 39.70,
    79: 36.00,
    80: 33.90,
    81: 50.00,
    82: 47.00,
    83: 48.00,
    84: 45.00,
    85: 38.00,
    86: 33.00,
}

# Default conversion factor: Bohr³ → e·Å²·V⁻¹ in the eV-Å-e unit system,
# matching LES upstream's ``normalization_factor = 0.1481847 / 14.3996``.
DEFAULT_ALPHA_NORM_FACTOR: float = 0.1481847 / 14.3996


# ---------------------------------------------------------------------------
# Typical formal-charge table
# ---------------------------------------------------------------------------
# Values mirror les.module.fixedcharges.typical_charge (Z=1..118). Standard
# Pauling/IUPAC formal charges (oxidation states of common ions). Used as a
# weak prior over training; LES upstream defaults to scaling_factor=0.5 so
# the baseline is half the formal charge and the network learns the rest.
TYPICAL_CHARGE_DICT: dict[int, float] = {
    1: +1.0,
    2: 0.0,
    3: +1.0,
    4: +2.0,
    5: +3.0,
    6: +4.0,
    7: -3.0,
    8: -2.0,
    9: -1.0,
    10: 0.0,
    11: +1.0,
    12: +2.0,
    13: +3.0,
    14: +4.0,
    15: +5.0,
    16: -2.0,
    17: -1.0,
    18: 0.0,
    19: +1.0,
    20: +2.0,
    21: +3.0,
    22: +4.0,
    23: +5.0,
    24: +3.0,
    25: +2.0,
    26: +2.0,
    27: +2.0,
    28: +2.0,
    29: +1.0,
    30: +2.0,
    31: +3.0,
    32: +4.0,
    33: +5.0,
    34: -2.0,
    35: -1.0,
    36: 0.0,
    37: +1.0,
    38: +2.0,
    39: +3.0,
    40: +4.0,
    41: +5.0,
    42: +6.0,
    43: +7.0,
    44: +3.0,
    45: +3.0,
    46: +2.0,
    47: +1.0,
    48: +2.0,
    49: +3.0,
    50: +2.0,
    51: +3.0,
    52: -2.0,
    53: -1.0,
    54: 0.0,
    55: +1.0,
    56: +2.0,
    57: +3.0,
    58: +3.0,
    59: +3.0,
    60: +3.0,
    61: +3.0,
    62: +3.0,
    63: +2.0,
    64: +3.0,
    65: +3.0,
    66: +3.0,
    67: +3.0,
    68: +3.0,
    69: +3.0,
    70: +2.0,
    71: +3.0,
    72: +4.0,
    73: +5.0,
    74: +6.0,
    75: +7.0,
    76: +4.0,
    77: +3.0,
    78: +2.0,
    79: +1.0,
    80: +2.0,
    81: +1.0,
    82: +2.0,
    83: +3.0,
    84: +2.0,
    85: -1.0,
    86: 0.0,
    87: +1.0,
    88: +2.0,
    89: +3.0,
    90: +4.0,
    91: +5.0,
    92: +6.0,
    93: +5.0,
    94: +4.0,
    95: +3.0,
    96: +3.0,
    97: +3.0,
    98: +3.0,
    99: +3.0,
    100: +3.0,
    101: +3.0,
    102: +2.0,
    103: +3.0,
    104: +4.0,
    105: +5.0,
    106: +6.0,
    107: +7.0,
    108: +4.0,
    109: +3.0,
    110: +2.0,
    111: +1.0,
    112: +2.0,
    113: +3.0,
    114: +2.0,
    115: +3.0,
    116: +2.0,
    117: -1.0,
    118: 0.0,
}

DEFAULT_CHARGE_NORM_FACTOR: float = 0.5


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def _build_lookup(
    table: dict[int, float], *, default: float = 0.0, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Build a Z-indexed lookup tensor from a dict, padded with ``default``."""
    max_z = max(table.keys())
    out = torch.full((max_z + 1,), default, dtype=dtype)
    for z, v in table.items():
        out[z] = v
    return out


class ElementChargeTable(nn.Module):
    """Per-element formal-charge baseline (additive).

    Computes a fixed prior charge ``q_baseline = scaling_factor · table[Z]``
    for each atom. Used as an additive baseline in front of a learnable
    charge head (the head learns the *residual* environment-dependent
    charge). Toggling this baseline off is equivalent to multiplying the
    output by 0; there is no learnable parameter.

    Args:
        scaling_factor: Global multiplier on the table values. Default
            ``0.5`` (LES upstream default — halves the formal charge as
            a soft prior).
        table: Optional override of the per-Z formal-charge dict.
            Default :data:`TYPICAL_CHARGE_DICT`.
    """

    def __init__(
        self,
        *,
        scaling_factor: float = DEFAULT_CHARGE_NORM_FACTOR,
        table: dict[int, float] | None = None,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.table_dict = TYPICAL_CHARGE_DICT if table is None else dict(table)
        lookup = _build_lookup(self.table_dict)
        self.register_buffer("lookup", lookup)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Return per-atom baseline charge in ``e``.

        Args:
            atomic_numbers: ``(N,)`` int tensor of atomic numbers.

        Returns:
            ``(N,)`` baseline charge.
        """
        return self.lookup[atomic_numbers] * self.scaling_factor


class ElementAlphaTable(nn.Module):
    """Per-element atomic-polarizability baseline (additive).

    Computes a fixed prior polarizability
    ``α_baseline = normalization_factor · table[Z]`` per atom. Used as
    an additive baseline in front of :class:`PolarizabilityHead`.

    Args:
        normalization_factor: Multiplier converting the dict's
            Bohr³ values into the project's unit system. Default
            ``0.1481847 / 14.3996`` (Bohr³ → e·Å²·V⁻¹), matching LES
            upstream.
        table: Optional override of the per-Z polarizability dict
            (in Bohr³). Default :data:`ALPHA_DICT_BOHR3`.
    """

    def __init__(
        self,
        *,
        normalization_factor: float = DEFAULT_ALPHA_NORM_FACTOR,
        table: dict[int, float] | None = None,
    ):
        super().__init__()
        self.normalization_factor = normalization_factor
        self.table_dict = ALPHA_DICT_BOHR3 if table is None else dict(table)
        lookup = _build_lookup(self.table_dict)
        self.register_buffer("lookup", lookup)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Return per-atom baseline polarizability.

        Args:
            atomic_numbers: ``(N,)`` int tensor of atomic numbers.

        Returns:
            ``(N,)`` baseline polarizability.
        """
        return self.lookup[atomic_numbers] * self.normalization_factor
