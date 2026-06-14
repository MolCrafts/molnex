"""Minimal in-tree extended-XYZ (extxyz) parser — molpy-only stack.

This module exists because :class:`molpy.io.trajectory.xyz.XYZTrajectoryReader`
only reads the canonical XYZ format (``n_atoms`` + comment + ``element x y z``
rows) and discards the comment-line metadata that extxyz files carry —
``Lattice="..."``, ``Properties=...``, ``energy=...``, ``pbc="..."`` — as well
as any per-atom columns beyond ``x y z``. The Sonata bulk-water RPBE-D3 data
ships in extended-XYZ format with per-frame ``cell``, ``energy``, and per-atom
``forces``, so an extxyz-aware parser is required.

This parser is intentionally narrow:

* depends on ``numpy`` + Python stdlib only (NO ``ase``);
* recognises only the column tags this project consumes
  (``species:S:1``, ``pos:R:3``, ``forces:R:3``) and skips others;
* returns a flat list of :class:`ExtxyzFrame` dataclasses; callers
  (e.g. :class:`molix.datasets.water_les.WaterLESSource`) bridge to
  :class:`torch.Tensor` and the flat-sample dict contract.

References:
    Extended-XYZ format spec (ASE wiki / Schimka et al. 2017 supp.):
    https://github.com/libAtoms/extxyz
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ExtxyzFrame", "parse_extxyz_frames"]


_log = logging.getLogger(__name__)


@dataclass
class ExtxyzFrame:
    """One parsed extended-XYZ frame.

    Attributes:
        n_atoms: Number of atoms in this frame.
        cell: ``(3, 3)`` row-major periodic cell in Å. Built from the
            comment line's ``Lattice="r00 r01 r02 r10 r11 r12 r20 r21 r22"``
            tag and reshaped to ``(3, 3)``.
        pbc: Tuple of three booleans for the x / y / z periodic
            directions. Defaults to ``(True, True, True)`` when absent from
            the comment line (with a debug-level log emitted).
        energy: Per-frame total energy in eV, read from the comment line's
            ``energy=...`` token. Missing tag raises ``ValueError``.
        species: Atomic symbols, length ``n_atoms``.
        pos: ``(n_atoms, 3)`` positions in Å.
        forces: ``(n_atoms, 3)`` forces in eV·Å⁻¹, or ``None`` when the
            ``Properties=`` declaration does not list a ``forces:R:3``
            column.
    """

    n_atoms: int
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    energy: float
    species: list[str]
    pos: np.ndarray
    forces: np.ndarray | None


def parse_extxyz_frames(path: str | Path) -> list[ExtxyzFrame]:
    """Parse every frame in an extended-XYZ file.

    Args:
        path: Path to the extxyz file. The file may contain multiple frames
            concatenated; each frame is ``n_atoms`` / comment / ``n_atoms``
            atom rows.

    Returns:
        One :class:`ExtxyzFrame` per frame, in file order.

    Raises:
        ValueError: A frame's comment line lacks an ``energy=`` token, or
            an atom row has fewer columns than the ``Properties=`` layout
            declares.
        FileNotFoundError: ``path`` does not exist.
    """
    p = Path(path)
    text = p.read_text()
    lines = text.splitlines()

    frames: list[ExtxyzFrame] = []
    i = 0
    while i < len(lines):
        # Skip blank lines between frames.
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atom_lines = lines[i + 2 : i + 2 + n_atoms]
        if len(atom_lines) != n_atoms:
            raise ValueError(
                f"{p}: frame at line {i + 1} declares {n_atoms} atoms but "
                f"only {len(atom_lines)} atom rows follow"
            )

        tokens = _tokenize_comment(comment)
        cell = _parse_lattice(tokens, source=p)
        pbc = _parse_pbc(tokens)
        energy = _parse_energy(tokens, source=p, frame_idx=len(frames))
        layout = _parse_properties(tokens, source=p)

        species: list[str] = []
        pos = np.empty((n_atoms, 3), dtype=np.float64)
        forces: np.ndarray | None = (
            np.empty((n_atoms, 3), dtype=np.float64) if layout["has_forces"] else None
        )
        for j, row in enumerate(atom_lines):
            parts = row.split()
            if len(parts) < layout["min_cols"]:
                raise ValueError(
                    f"{p}: frame {len(frames)} atom {j} has {len(parts)} "
                    f"columns; need at least {layout['min_cols']} per Properties"
                )
            species.append(parts[layout["species_col"]])
            pos[j] = (
                float(parts[layout["pos_col"]]),
                float(parts[layout["pos_col"] + 1]),
                float(parts[layout["pos_col"] + 2]),
            )
            if forces is not None:
                fc = layout["forces_col"]
                forces[j] = (float(parts[fc]), float(parts[fc + 1]), float(parts[fc + 2]))

        frames.append(
            ExtxyzFrame(
                n_atoms=n_atoms,
                cell=cell,
                pbc=pbc,
                energy=energy,
                species=species,
                pos=pos,
                forces=forces,
            )
        )
        i += 2 + n_atoms

    return frames


# ---------------------------------------------------------------------------
# Comment-line helpers
# ---------------------------------------------------------------------------


def _tokenize_comment(comment: str) -> dict[str, str]:
    """Split an extxyz comment line into ``{key: value}`` pairs.

    Values quoted with double quotes may contain whitespace (e.g.
    ``Lattice="12.0 0.0 0.0 ..."``); bareword values run until the next
    whitespace. A naive ``comment.split()`` would shatter quoted lattice
    strings into nine broken tokens, so this tokenizer is quote-aware.

    Args:
        comment: Raw comment line (line 2 of each extxyz frame).

    Returns:
        Mapping from key to raw value (without surrounding quotes).
    """
    tokens: dict[str, str] = {}
    n = len(comment)
    i = 0
    while i < n:
        while i < n and comment[i].isspace():
            i += 1
        if i >= n:
            break
        # Read key up to '='.
        eq = comment.find("=", i)
        if eq == -1:
            break
        key = comment[i:eq].strip()
        i = eq + 1
        if i >= n:
            tokens[key] = ""
            break
        if comment[i] == '"':
            close = comment.find('"', i + 1)
            if close == -1:
                tokens[key] = comment[i + 1 :]
                break
            tokens[key] = comment[i + 1 : close]
            i = close + 1
        else:
            j = i
            while j < n and not comment[j].isspace():
                j += 1
            tokens[key] = comment[i:j]
            i = j
    return tokens


def _parse_lattice(tokens: dict[str, str], *, source: Path) -> np.ndarray:
    """Parse ``Lattice="r00 r01 ..."`` into a ``(3, 3)`` row-major matrix."""
    raw = tokens.get("Lattice")
    if raw is None:
        raise ValueError(f'{source}: comment line missing Lattice="..." tag')
    vals = [float(x) for x in raw.split()]
    if len(vals) != 9:
        raise ValueError(f"{source}: Lattice tag has {len(vals)} numbers; need exactly 9")
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def _parse_pbc(tokens: dict[str, str]) -> tuple[bool, bool, bool]:
    """Parse ``pbc="T T T"``; default ``(True, True, True)`` when absent."""
    raw = tokens.get("pbc")
    if raw is None:
        _log.debug("extxyz comment missing pbc tag; defaulting to (T, T, T)")
        return (True, True, True)
    parts = raw.split()
    if len(parts) != 3:
        raise ValueError(f"pbc tag has {len(parts)} entries; need 3")
    return tuple(p.upper() == "T" for p in parts)  # type: ignore[return-value]


def _parse_energy(tokens: dict[str, str], *, source: Path, frame_idx: int) -> float:
    """Parse ``energy=<float>``; raise if missing."""
    raw = tokens.get("energy")
    if raw is None:
        raise ValueError(f"{source}: frame {frame_idx} comment line missing energy=... tag")
    return float(raw)


def _parse_properties(tokens: dict[str, str], *, source: Path) -> dict[str, int | bool]:
    """Parse ``Properties=species:S:1:pos:R:3:forces:R:3`` into column offsets.

    Returns a dict with::

        species_col:  column index of the symbol column
        pos_col:      column index where the 3 position columns begin
        forces_col:   column index where the 3 force columns begin (only if has_forces)
        has_forces:   whether forces are declared
        min_cols:     minimum number of columns each atom row must have
    """
    raw = tokens.get("Properties", "species:S:1:pos:R:3")
    parts = raw.split(":")
    if len(parts) % 3 != 0:
        raise ValueError(f"{source}: Properties tag {raw!r} is not a triple of (name:type:width)")

    species_col: int | None = None
    pos_col: int | None = None
    forces_col: int | None = None
    cursor = 0
    for k in range(0, len(parts), 3):
        name, _type, width_str = parts[k], parts[k + 1], parts[k + 2]
        width = int(width_str)
        if name == "species":
            species_col = cursor
        elif name == "pos":
            pos_col = cursor
        elif name == "forces":
            forces_col = cursor
        cursor += width

    if species_col is None or pos_col is None:
        raise ValueError(
            f"{source}: Properties tag must declare both species and pos columns; got {raw!r}"
        )

    layout: dict[str, int | bool] = {
        "species_col": species_col,
        "pos_col": pos_col,
        "has_forces": forces_col is not None,
        "min_cols": cursor,
    }
    if forces_col is not None:
        layout["forces_col"] = forces_col
    return layout
