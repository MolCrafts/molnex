"""Tests for ``molix.io.extxyz`` — in-tree extxyz metadata parser.

Acceptance trace
----------------
* ``test_parse_water_fixture`` / ``test_no_ase_imports``

The former ``TestChargedDimersRemoved`` class
pinned the absence of one dataset adapter; the whole acquisition package is
gone now (spec molix-data-dissolution-04-etl-exit) and
``tests/test_repository_boundaries.py`` enforces that outright, so the
narrower class was retired rather than rewritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molix.io.extxyz import ExtxyzFrame, parse_extxyz_frames, write_extxyz_frames


class TestParseFixture:
    def test_parse_water_fixture(self, water_les_root: Path) -> None:
        """Parser returns one ExtxyzFrame per fixture frame with correct shapes."""
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
        """Lattice="..." parses into row-major (3,3) Å matrix.

        Fixture writes ``Lattice="12.0 0 0 0 12.0 0 0 0 12.0"``; the matrix is
        ``edge·I₃``. Read it back element-wise to nail the row-major ordering.
        """
        path = water_les_root / "test-H2O_RPBE-D3.xyz"
        frames = parse_extxyz_frames(path)
        expected = 12.0 * np.eye(3)
        for frame in frames:
            np.testing.assert_allclose(frame.cell, expected, atol=1e-6)

    def test_lattice_with_spaces_inside_quotes(self, tmp_path: Path) -> None:
        """Tokenizer handles quoted values containing whitespace.

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
        """Properties without ``forces:R:3`` ⇒ ``frame.forces is None``."""
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
        """Missing ``energy=...`` in comment raises ``ValueError``."""
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
        """Missing ``pbc=`` defaults to (True, True, True) (logs a warning)."""
        f = tmp_path / "nopbc.xyz"
        f.write_text(
            "1\n"
            'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
            "Properties=species:S:1:pos:R:3 energy=-1.0\n"
            "H  0.0 0.0 0.0\n"
        )
        frames = parse_extxyz_frames(f)
        assert frames[0].pbc == (True, True, True)


class TestOpenBoundary:
    def test_a_frame_without_lattice_is_open(self, tmp_path: Path) -> None:
        """No ``Lattice``: no cell, and ``pbc`` defaults to ``F F F`` (extxyz rule)."""
        f = tmp_path / "open.xyz"
        f.write_text("1\nProperties=species:S:1:pos:R:3 energy=345.15 row=7\nC 0.0 0.0 0.0\n")
        (frame,) = parse_extxyz_frames(f)
        assert frame.cell is None
        assert frame.pbc == (False, False, False)
        assert frame.energy == pytest.approx(345.15)

    def test_a_periodic_direction_without_lattice_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.xyz"
        f.write_text('1\nProperties=species:S:1:pos:R:3 energy=0.0 pbc="T F F"\nC 0 0 0\n')
        with pytest.raises(ValueError, match="Lattice"):
            parse_extxyz_frames(f)

    def test_other_comment_tokens_land_in_info(self, tmp_path: Path) -> None:
        f = tmp_path / "info.xyz"
        f.write_text('1\nProperties=species:S:1:pos:R:3 energy=1.0 row=17 pbc="F F F"\nH 0 0 0\n')
        (frame,) = parse_extxyz_frames(f)
        assert frame.info == {"row": "17"}

    def test_the_writer_output_reads_back(self, tmp_path: Path) -> None:
        f = tmp_path / "traj.xyz"
        pos = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.1
        write_extxyz_frames(
            f,
            species=["O", "H"],
            positions=pos,
            energies=np.array([-1.5, -1.25]),
            tags=["temperature=300.00", "temperature=301.50"],
        )
        frames = parse_extxyz_frames(f)
        assert [fr.energy for fr in frames] == [-1.5, -1.25]
        assert [fr.info["temperature"] for fr in frames] == ["300.00", "301.50"]
        assert all(fr.cell is None and fr.species == ["O", "H"] for fr in frames)
        np.testing.assert_allclose(np.stack([fr.pos for fr in frames]), pos)
