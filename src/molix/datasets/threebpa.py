"""3BPA DataSource: temperature-transferability benchmark for MLPs.

Reference:
    Kovacs et al., "Linear Atomic Cluster Expansion Force Fields for
    Organic Molecules: Beyond RMSE" J. Chem. Theory Comput. 2021.
    https://doi.org/10.1021/acs.jctc.1c00647

3BPA = 3-(benzyloxy)pyridin-2-amine (C13H14N2O). The benchmark ships
four extended-XYZ files:

  * ``train_300K.xyz``   — 500 structures sampled at 300 K (training pool).
  * ``test_300K.xyz``    — held-out 300 K structures (in-distribution).
  * ``test_600K.xyz``    — held-out 600 K  (temperature extrapolation).
  * ``test_1200K.xyz``   — held-out 1200 K (harder extrapolation).

The extended-XYZ comment line carries ``energy=<float>`` (eV) and each
atom row is ``<symbol> x y z fx fy fz`` (eV/Å).

Usage::

    from molix.datasets.threebpa import ThreeBPASource, THREEBPA_TARGET_SCHEMA

    src_train = ThreeBPASource(data_dir / "train_300K.xyz", tag="train_300K")
    src_600   = ThreeBPASource(data_dir / "test_600K.xyz",  tag="test_600K")
"""

from __future__ import annotations

from pathlib import Path

import torch

from molix.data.collate import TargetSchema
from molix.data.source import Sample


THREEBPA_TARGET_SCHEMA: TargetSchema = TargetSchema(
    graph_level=frozenset({"energy"}),
    atom_level=frozenset({"forces"}),
)


def _parse_extxyz(path: Path) -> list[Sample]:
    """Parse an extended-XYZ file shipped with the 3BPA benchmark."""
    from molpy.core.element import Element

    samples: list[Sample] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())
        comment = lines[i + 1]
        energy: float | None = None
        for tok in comment.split():
            if tok.startswith("energy="):
                energy = float(tok.split("=", 1)[1])
                break
        if energy is None:
            raise ValueError(
                f"Missing 'energy=...' tag in {path} at structure {len(samples)}"
            )
        Zs, pos, forces = [], [], []
        for row in lines[i + 2 : i + 2 + natoms]:
            parts = row.split()
            Zs.append(Element.get_atomic_number(parts[0]))
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
            forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        samples.append(
            {
                "Z": torch.tensor(Zs, dtype=torch.long),
                "pos": torch.tensor(pos, dtype=torch.float32),
                "targets": {
                    "energy": torch.tensor([energy], dtype=torch.float32),
                    "forces": torch.tensor(forces, dtype=torch.float32),
                },
            }
        )
        i += 2 + natoms
    return samples


class ThreeBPASource:
    """DataSource for one 3BPA extended-XYZ split.

    Args:
        path: Path to the ``.xyz`` file.
        tag: Short identifier of the split (e.g. ``"train_300K"``), used
             to make :attr:`source_id` distinct between temperature shards.
    """

    def __init__(self, path: str | Path, *, tag: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"3BPA file not found: {self.path}. Download the benchmark "
                "from the Kovacs et al. (2021) supplementary material."
            )
        self.tag = tag
        self._samples = _parse_extxyz(self.path)
        self._size = self.path.stat().st_size

    @property
    def source_id(self) -> str:
        return f"3bpa:{self.tag}:size={self._size}:n={len(self._samples)}"

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


__all__ = ["ThreeBPASource", "THREEBPA_TARGET_SCHEMA"]
