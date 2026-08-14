"""Shared fixtures for the ``molix`` unit tests.

Two builders live here, both for the same reason: their consumers sit in
different sub-packages, so the shared code lives at the nearest common
ancestor rather than being copied (and drifting) in each leaf conftest.

- :func:`equal_count_samples` — the degenerate ``n_edges == n_atoms``
  geometry used by ``test_store/test_cache.py``,
  ``test_store/test_dataset.py`` and ``test_batch/test_collate_packed.py``
  to pin the packed-cache bucket-identity contract.
- :func:`water_les_root` — the synthetic extended-XYZ water trajectory,
  read by ``test_io/test_extxyz.py``. It moved up here when the codec moved
  to ``molix.io`` and its second consumer, the dataset loader, left the
  repository with the acquisition layer (now MolHub's).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch


def equal_count_samples(n: int = 3) -> list[dict]:
    """Build *n* samples with ``n_edges == n_atoms == 2`` in every sample.

    A 2-atom molecule with a single bidirectional pair (``NeighborList``'s
    default ``symmetry=True`` → ``E = 2 x n_pairs``) is the smallest
    physically real case where the per-sample edge count collides with the
    atom count. Nothing in the leading dim then distinguishes a per-atom
    key from a per-edge key, so the packed schema must decide bucket
    membership by key role, not by numerology.

    Every value is hand-built (no RNG), so the fixture is bit-identical
    across runs and independent of global torch seed state.

    Args:
        n: Number of samples.

    Returns:
        Flat sample dicts with per-atom ``Z`` ``(2,)`` / ``pos`` ``(2, 3)``,
        per-edge ``edge_index`` ``(2, 2)`` / ``edge_diff`` ``(2, 3)`` /
        ``edge_dist`` ``(2,)``, and graph-level ``targets.U0`` ``(1,)``
        carrying the sample's identity ``float(i)``.
    """
    samples: list[dict] = []
    for i in range(n):
        # Atom 1 sits at +1 A along x from atom 0; the pair is stored in both
        # directions, so edge_diff = pos[target] - pos[source] flips sign.
        samples.append(
            {
                "Z": torch.tensor([1, 6], dtype=torch.long),
                "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "edge_diff": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                "edge_dist": torch.tensor([1.0, 1.0]),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Extended-XYZ water fixture (shared by test_io/ and test_datasets/)
# ---------------------------------------------------------------------------
#
# Files are written in the exact extended-XYZ format consumed by
# ``parse_extxyz_frames``: a comment line carrying ``Lattice="..."``,
# ``Properties=...``, ``energy=...`` (eV) and ``pbc="T T T"``, then atom rows
# ``<symbol> x y z fx fy fz``.


def _format_lattice(edge: float) -> str:
    """Return a 9-float row-major cubic lattice string for extxyz."""
    return f'"{edge:.6f} 0.0 0.0 0.0 {edge:.6f} 0.0 0.0 0.0 {edge:.6f}"'


def _make_water_frame(
    *,
    energy: float,
    edge: float,
    o_positions: Iterable[tuple[float, float, float]],
    seed: int,
) -> str:
    """Build a single extxyz frame of n_water H₂O molecules.

    Each O is placed at the supplied position; the two H atoms of that
    water hang off the O at canonical OH bond length / HOH angle, with a
    deterministic seed-based jitter so the dataset isn't identical
    between frames. Forces are seed-derived deterministic floats.
    """
    rng = random.Random(seed)
    o_list = list(o_positions)
    n_atoms = 3 * len(o_list)
    bond = 0.96  # Å
    half_angle = math.radians(104.5 / 2.0)
    rows: list[str] = []
    fx_total = 0.0  # forces will be neutralized at the end so they sum near zero
    for ox, oy, oz in o_list:
        # O
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"O  {ox:.6f} {oy:.6f} {oz:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        # H₁: along +x rotated by half_angle in xy
        h1x = ox + bond * math.cos(half_angle)
        h1y = oy + bond * math.sin(half_angle)
        h1z = oz
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"H  {h1x:.6f} {h1y:.6f} {h1z:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        # H₂
        h2x = ox + bond * math.cos(half_angle)
        h2y = oy - bond * math.sin(half_angle)
        h2z = oz
        fx, fy, fz = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        rows.append(f"H  {h2x:.6f} {h2y:.6f} {h2z:.6f}  {fx:.6f} {fy:.6f} {fz:.6f}")
        fx_total += 1
    comment = (
        f"Lattice={_format_lattice(edge)} "
        f"Properties=species:S:1:pos:R:3:forces:R:3 "
        f"energy={energy:.6f} "
        f'pbc="T T T"'
    )
    return "\n".join([str(n_atoms), comment, *rows]) + "\n"


#: Per-atom binding-energy reference used to build paper-realistic fixture
#: energies. RPBE-D3 bulk water sits near ``-10 eV/atom`` (cf. Cheng 2025,
#: npj Comput. Mater. 11:80 — 64 H₂O / 192 atoms total energies cluster
#: around ``-2000 eV``). A 2-water 6-atom micro-fixture therefore targets
#: ``-60 eV`` ± a small per-frame jitter, keeping the fixture in the same
#: order of magnitude as the upstream extxyz files so unit checks are
#: meaningful.
_WATER_E0_PER_ATOM_EV: float = -10.0


def _write_water_extxyz(path: Path, *, n_frames: int, n_water: int, edge: float) -> None:
    """Write ``n_frames`` water-box frames to *path*.

    Each frame's O atoms sit on a deterministic 1-D string spaced by
    ``edge / (n_water + 1)`` so the box is well within periodic limits
    and atoms never fall outside the cubic cell. Per-frame energies follow
    ``E(i) = (3 · n_water) · _WATER_E0_PER_ATOM_EV - 0.1·i`` so the values
    land in the eV-scale range RPBE-D3 actually produces (≈ ``-10 eV/atom``);
    forces are ``[-1, 1] eV·Å⁻¹`` (well below the 50 eV·Å⁻¹ sustained-force
    threshold typical of well-converged liquid-water frames).
    """
    spacing = edge / (n_water + 1)
    o_positions = [((i + 1) * spacing, edge / 2.0, edge / 2.0) for i in range(n_water)]
    n_atoms = 3 * n_water
    e_base = n_atoms * _WATER_E0_PER_ATOM_EV
    frames = [
        _make_water_frame(
            energy=e_base - 0.1 * frame_i,
            edge=edge,
            o_positions=o_positions,
            seed=frame_i,
        )
        for frame_i in range(n_frames)
    ]
    path.write_text("".join(frames))


@pytest.fixture
def water_les_root(tmp_path: Path) -> Path:
    """Directory with synthetic ``train-H2O_RPBE-D3.xyz`` (40 frames) and
    ``test-H2O_RPBE-D3.xyz`` (2 frames).

    40 train frames is the smallest count that produces a clean
    0.95 / 0.05 deterministic slice (38 train / 2 val) so the
    deterministic-split test asserts disjoint, ordered, non-empty
    slices on both sides.
    """
    root = tmp_path / "water_les"
    root.mkdir()
    train_path = root / "train-H2O_RPBE-D3.xyz"
    test_path = root / "test-H2O_RPBE-D3.xyz"
    _write_water_extxyz(train_path, n_frames=40, n_water=2, edge=12.0)
    _write_water_extxyz(test_path, n_frames=2, n_water=2, edge=12.0)
    return root
