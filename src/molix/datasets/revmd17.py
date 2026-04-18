"""revMD17 DataSource: revised MD17 molecular dynamics trajectories.

Reference:
    Christensen & von Lilienfeld, "On the role of gradients for machine
    learning of molecular energies and forces" MLST 2020.
    https://doi.org/10.1088/2632-2153/abba6f

The revised MD17 dataset recomputes energies and forces at a tighter
PBE/def2-TZVP convergence threshold than the original MD17 trajectories,
removing the noise that lets some models memorise artefacts. It is the
benchmark used in the Allegro paper (Musaelian et al. 2023).

Usage::

    from molix.data import Pipeline, NeighborList, MmapDataset
    from molix.data.cache import cache
    from molix.datasets import RevMD17Source

    source = RevMD17Source(data_dir, molecule="aspirin")
    pipe = Pipeline("revmd17-aspirin").add(NeighborList(cutoff=7.0)).build()
    cache(pipe, source, sink=sink_path)
    ds = MmapDataset(sink_path)
    # RevMD17Source.TARGET_SCHEMA exposes graph {"energy"} + atom {"forces"}.
"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import numpy as np
import torch

from molix.data.collate import TargetSchema
from molix.data.source import Sample

ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]


# Canonical 10 molecules of revMD17 and their filenames on the mirror.
_MOLECULES: dict[str, str] = {
    "aspirin": "rmd17_aspirin.npz",
    "azobenzene": "rmd17_azobenzene.npz",
    "benzene": "rmd17_benzene.npz",
    "ethanol": "rmd17_ethanol.npz",
    "malonaldehyde": "rmd17_malonaldehyde.npz",
    "naphthalene": "rmd17_naphthalene.npz",
    "paracetamol": "rmd17_paracetamol.npz",
    "salicylic": "rmd17_salicylic.npz",
    "toluene": "rmd17_toluene.npz",
    "uracil": "rmd17_uracil.npz",
}


class RevMD17Source:
    """DataSource for the revised MD17 trajectories.

    Each sample contains ``Z``, ``pos``, and targets ``energy`` / ``forces``.
    Energies are in kcal/mol and forces in kcal/(mol·Å) as distributed.

    Args:
        root: Directory for the downloaded NPZ file.
        molecule: One of the 10 revMD17 molecule names (e.g. ``"aspirin"``).
        download: Download the file if it does not exist.
    """

    # Figshare mirror of the revised dataset (Christensen et al.).
    BASE_URL = "https://figshare.com/ndownloader/files/23950376"

    TARGET_SCHEMA: TargetSchema = TargetSchema(
        graph_level=frozenset({"energy"}),
        atom_level=frozenset({"forces"}),
    )

    def __init__(
        self,
        root: str | Path,
        molecule: str = "aspirin",
        download: bool = True,
    ) -> None:
        if molecule not in _MOLECULES:
            raise ValueError(
                f"Unknown revMD17 molecule '{molecule}'. "
                f"Available: {sorted(_MOLECULES)}"
            )
        self.root = Path(root)
        self.molecule = molecule
        self.filename = _MOLECULES[molecule]
        self.filepath = self.root / self.filename
        self.root.mkdir(parents=True, exist_ok=True)

        if download and not self.filepath.exists():
            raise FileNotFoundError(
                f"revMD17 file not found at {self.filepath}. "
                f"Download the archive from {self.BASE_URL} and extract "
                f"{self.filename} into {self.root}. (The Figshare archive "
                "bundles all 10 molecules in one tarball.)"
            )
        if not self.filepath.exists():
            raise FileNotFoundError(f"revMD17 file missing: {self.filepath}")

        data = np.load(self.filepath)
        for key in ("nuclear_charges", "coords", "energies", "forces"):
            if key not in data:
                raise KeyError(f"revMD17 file missing required key '{key}'")
        self._z = torch.from_numpy(data["nuclear_charges"]).long()
        self._R = torch.from_numpy(data["coords"]).float()
        self._E = torch.from_numpy(data["energies"].reshape(-1)).float()
        self._F = torch.from_numpy(data["forces"]).float()

    @property
    def source_id(self) -> str:
        size = self.filepath.stat().st_size
        return f"revmd17:{self.molecule}:size={size}:n={len(self)}"

    def __len__(self) -> int:
        return int(self._R.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        return {
            "Z": self._z,
            "pos": self._R[idx],
            "targets": {
                "energy": self._E[idx : idx + 1],
                "forces": self._F[idx],
            },
        }


__all__ = ["RevMD17Source"]
