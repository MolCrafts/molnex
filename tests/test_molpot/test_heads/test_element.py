"""Bitwise parity tests for :class:`ElementChargeTable` and :class:`ElementAlphaTable`.

Asserts the per-element values match a hardcoded reference dict drawn
from the CRC Handbook of Chemistry and Physics (atomic polarizabilities)
and the standard Pauling/IUPAC formal-charge convention. No upstream
LES import — the parity is against this reference dict alone, so the
test path is dependency-free.

The reference values here are spot-checks from the CRC Handbook; the
full ``ALPHA_DICT_BOHR3`` and ``TYPICAL_CHARGE_DICT`` modules in
``molpot.heads.element_baselines`` cover Z=1..118.
"""

from __future__ import annotations

import torch

from molpot.heads import ElementAlphaTable, ElementChargeTable
from molpot.heads.element import (
    ALPHA_DICT_BOHR3,
    DEFAULT_ALPHA_NORM_FACTOR,
    DEFAULT_CHARGE_NORM_FACTOR,
    TYPICAL_CHARGE_DICT,
)

# ---------------------------------------------------------------------------
# Spot-check reference dicts (CRC Handbook + IUPAC formal charges).
#
# These are independent of the source-side ``ALPHA_DICT_BOHR3`` / ``TYPICAL_CHARGE_DICT``
# imports above — they are the trusted physical / chemical reference. The
# test asserts that the source values (a) round-trip into the table buffer
# correctly and (b) match these CRC-cited reference values.
# ---------------------------------------------------------------------------

CRC_ALPHA_REFERENCE: dict[int, float] = {
    1: 4.50,  # H
    6: 11.30,  # C
    7: 7.40,  # N
    8: 5.30,  # O
    16: 19.40,  # S
    17: 14.60,  # Cl
    79: 36.00,  # Au
}

IUPAC_FORMAL_CHARGE_REFERENCE: dict[int, float] = {
    1: +1.0,  # H
    6: +4.0,  # C
    7: -3.0,  # N
    8: -2.0,  # O
    11: +1.0,  # Na
    17: -1.0,  # Cl
    79: +1.0,  # Au
}


# ---------------------------------------------------------------------------
# ElementChargeTable
# ---------------------------------------------------------------------------


class TestElementChargeTable:
    def test_default_scaling_factor(self) -> None:
        """LES upstream default: scaling=0.5 — halves the formal charge."""
        assert DEFAULT_CHARGE_NORM_FACTOR == 0.5

    def test_full_table_roundtrip(self) -> None:
        """Every Z in the source dict appears identically in the lookup buffer."""
        table = ElementChargeTable(scaling_factor=1.0)
        for z, v in TYPICAL_CHARGE_DICT.items():
            actual = table.lookup[z].item()
            assert actual == v, f"Z={z}: lookup={actual}, dict={v}"

    def test_spot_check_against_crc_reference(self) -> None:
        """Independent CRC/IUPAC reference values match the source dict."""
        for z, ref in IUPAC_FORMAL_CHARGE_REFERENCE.items():
            assert TYPICAL_CHARGE_DICT[z] == ref, (
                f"Z={z}: source dict {TYPICAL_CHARGE_DICT[z]}, ref {ref}"
            )

    def test_forward_applies_scaling(self) -> None:
        table = ElementChargeTable(scaling_factor=0.5)
        Z = torch.tensor([1, 6, 7, 8, 17])
        out = table(Z)
        expected = torch.tensor([TYPICAL_CHARGE_DICT[z.item()] * 0.5 for z in Z], dtype=out.dtype)
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)

    def test_zero_scaling_disables_baseline(self) -> None:
        table = ElementChargeTable(scaling_factor=0.0)
        Z = torch.tensor([1, 6, 79])
        out = table(Z)
        assert (out == 0.0).all()


# ---------------------------------------------------------------------------
# ElementAlphaTable
# ---------------------------------------------------------------------------


class TestElementAlphaTable:
    def test_default_normalization_factor(self) -> None:
        """LES upstream default: 0.1481847 / 14.3996 (Bohr³ → e·Å²·V⁻¹)."""
        assert abs(DEFAULT_ALPHA_NORM_FACTOR - (0.1481847 / 14.3996)) < 1e-15

    def test_full_table_roundtrip(self) -> None:
        table = ElementAlphaTable(normalization_factor=1.0)
        for z, v in ALPHA_DICT_BOHR3.items():
            actual = table.lookup[z].item()
            assert actual == v, f"Z={z}: lookup={actual}, dict={v}"

    def test_spot_check_against_crc_reference(self) -> None:
        for z, ref in CRC_ALPHA_REFERENCE.items():
            assert ALPHA_DICT_BOHR3[z] == ref, f"Z={z}: source {ALPHA_DICT_BOHR3[z]}, CRC {ref}"

    def test_forward_applies_normalization(self) -> None:
        table = ElementAlphaTable()
        Z = torch.tensor([1, 6, 8, 79])
        out = table(Z)
        expected = torch.tensor(
            [ALPHA_DICT_BOHR3[z.item()] * DEFAULT_ALPHA_NORM_FACTOR for z in Z],
            dtype=out.dtype,
        )
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=0.0)

    def test_zero_factor_disables_baseline(self) -> None:
        table = ElementAlphaTable(normalization_factor=0.0)
        Z = torch.tensor([1, 6, 79])
        out = table(Z)
        assert (out == 0.0).all()
