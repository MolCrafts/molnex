"""MD17 DataSource: molecular dynamics trajectories."""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molix.data.source import Sample

ssl._create_default_https_context = ssl._create_unverified_context


class MD17Source:
    """DataSource for MD17 molecular dynamics trajectories.

    Each sample contains ``Z``, ``pos``, and targets ``energy`` / ``forces``.

    Args:
        root: Directory for the downloaded NPZ file.
        molecule: Molecule name (e.g. ``"aspirin"``).
        download: Download the file if it does not exist.
    """

    BASE_URL = "http://www.quantum-machine.org/gdml/data/npz/md17"

    def __init__(
        self,
        root: str | Path,
        molecule: str = "aspirin",
        download: bool = True,
    ) -> None:
        self.root = Path(root)
        self.molecule = molecule
        self.filename = f"md17_{molecule}.npz"
        self.filepath = self.root / self.filename

        self.root.mkdir(parents=True, exist_ok=True)

        if download and not self.filepath.exists():
            self._download()

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"MD17 file not found at {self.filepath}. "
                "Set download=True or place it manually."
            )

        data = np.load(self.filepath)
        for key in ("z", "R", "E", "F"):
            if key not in data:
                raise KeyError(f"MD17 file missing required key '{key}'")

        self._z = torch.from_numpy(data["z"]).long()
        self._R = torch.from_numpy(data["R"]).float()
        self._E = torch.from_numpy(data["E"].reshape(-1)).float()
        self._F = torch.from_numpy(data["F"]).float()

    def _download(self) -> None:
        url = f"{self.BASE_URL}/{self.filename}"
        urllib.request.urlretrieve(url, str(self.filepath))

    # -- DataSource protocol ------------------------------------------------

    @property
    def source_id(self) -> str:
        size = self.filepath.stat().st_size
        return f"md17:{self.molecule}:size={size}:n={len(self)}"

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
