"""Tests for ``molix.io.extxyz`` — in-tree extxyz metadata parser.

Acceptance trace
----------------
* ac-006 → ``test_parse_water_fixture`` / ``test_no_ase_imports``

The former ``TestChargedDimersRemoved`` class (ac-007 of the extxyz spec)
pinned the absence of one dataset adapter; the whole acquisition package is
gone now (spec molix-data-dissolution-04-etl-exit) and
``tests/test_repository_boundaries.py`` enforces that outright, so the
narrower class was retired rather than rewritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molix.io.extxyz import ExtxyzFrame, parse_extxyz_frames


class TestParseFixture:
    def test_parse_water_fixture(self, water_les_root: Path) -> None:
        """ac-006: parser returns one ExtxyzFrame per fixture frame with correct shapes."""
        path = water_les_root / "train-H2O_RPBE-D3.xyz"
        frames = parse_extxyz_frames(path)
        assert len(frames) == 40
        for i, frame in enumerate(frames):
            assert isinstance(frame, ExtxyzFrame)
            assert frame.n_atoms == 6  # 2 H₂O = 6 atoms
            assert frame.cell.shape == (3, 3)
            assert frame.cell.dtype in (np.float64, np.float32)
            assert frame.pbc == (True, True, True)
            assert isinstance(frame.energy, float)
            assert frame.energy == pytest.approx(-60.0 - 0.1 * i, abs=1e-4)
            assert len(frame.species) == 6
            assert frame.species == ["O", "H", "H", "O", "H", "H"]
            assert frame.pos.shape == (6, 3)
            assert frame.forces is not None
            assert frame.forces.shape == (6, 3)

    def test_cell_row_major(self, water_les_root: Path) -> None:
        """ac-006: Lattice="..." parses into row-major (3,3) Å matrix.

        Fixture writes ``Lattice="12.0 0 0 0 12.0 0 0 0 12.0"``; the matrix is
        ``edge·I₃``. Read it back element-wise to nail the row-major ordering.
        """
        path = water_les_root / "test-H2O_RPBE-D3.xyz"
        frames = parse_extxyz_frames(path)
        expected = 12.0 * np.eye(3)
        for frame in frames:
            np.testing.assert_allclose(frame.cell, expected, atol=1e-6)

    def test_lattice_with_spaces_inside_quotes(self, tmp_path: Path) -> None:
        """ac-006: tokenizer handles quoted values containing whitespace.

        ``comment.split()`` would shatter ``Lattice="a b c d ..."`` into nine
        broken tokens. The parser must use a quote-aware splitter.
        """
        f = tmp_path / "single.xyz"
        f.write_text(
            "1\n"
            'Lattice="10.0 0.0 0.0 0.0 11.0 0.0 0.0 0.0 12.0" '
            'Properties=species:S:1:pos:R:3 energy=-1.5 pbc="T T T"\n'
            "H  1.0 2.0 3.0\n"
        )
        frames = parse_extxyz_frames(f)
        assert len(frames) == 1
        expected = np.diag([10.0, 11.0, 12.0])
        np.testing.assert_allclose(frames[0].cell, expected, atol=1e-9)
        assert frames[0].energy == pytest.approx(-1.5)

    def test_forces_optional_when_missing_from_properties(self, tmp_path: Path) -> None:
        """ac-006: Properties without ``forces:R:3`` ⇒ ``frame.forces is None``."""
        f = tmp_path / "noforce.xyz"
        f.write_text(
            "1\n"
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
            'Properties=species:S:1:pos:R:3 energy=-2.0 pbc="T T T"\n'
            "He  5.0 5.0 5.0\n"
        )
        frames = parse_extxyz_frames(f)
        assert len(frames) == 1
        assert frames[0].forces is None
        np.testing.assert_allclose(frames[0].pos, [[5.0, 5.0, 5.0]])

    def test_missing_energy_raises(self, tmp_path: Path) -> None:
        """ac-006: missing ``energy=...`` in comment raises ``ValueError``."""
        f = tmp_path / "noenergy.xyz"
        f.write_text(
            "1\n"
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
            'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            "H  0.0 0.0 0.0\n"
        )
        with pytest.raises(ValueError, match="energy"):
            parse_extxyz_frames(f)

    def test_pbc_defaults_to_true_when_missing(self, tmp_path: Path) -> None:
        """ac-006: missing ``pbc=`` defaults to (True, True, True) (logs a warning)."""
        f = tmp_path / "nopbc.xyz"
        f.write_text(
            "1\n"
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
            "Properties=species:S:1:pos:R:3 energy=-1.0\n"
            "H  0.0 0.0 0.0\n"
        )
        frames = parse_extxyz_frames(f)
        assert frames[0].pbc == (True, True, True)
