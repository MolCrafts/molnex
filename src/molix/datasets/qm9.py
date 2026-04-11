"""QM9 DataSource: molecular property prediction."""

from __future__ import annotations

import io
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from molpy.io.data.xyz import XYZReader
from tqdm import tqdm

from molix.data.source import Sample


class QM9Source:
    """DataSource for the QM9 dataset (molecular properties).

    Each sample contains ``Z``, ``pos``, and scalar targets
    (``U0``, ``H``, ``G``, ``mu``, etc.).

    Args:
        root: Directory for downloaded/extracted files.
        source: URL or local path to the tarball.  Defaults to Figshare.
        extract: If *True*, extract the tarball to disk for faster IO.
        total: Subsample to at most this many molecules.
    """

    DEFAULT_URL = "https://ndownloader.figshare.com/files/3195389"
    EXCLUDE_URL = "https://figshare.com/ndownloader/files/3195404"

    PROPERTY_NAMES = [
        "tag",
        "index",
        "A",
        "B",
        "C",
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
    ]

    def __init__(
        self,
        root: str | Path = "./data/qm9",
        source: str | Path | None = None,
        extract: bool = False,
        total: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.tarball_path = self.root / "qm9.tar.bz2"
        self.extracted_path = self.root / "qm9_extracted"
        self.exclude_path = self.root / "qm9_exclude.txt"
        self.extract = extract
        self.total = total

        self.root.mkdir(parents=True, exist_ok=True)

        if source is None:
            source = self.DEFAULT_URL

        source_is_url = isinstance(source, str) and source.startswith("http")
        source_path = Path(source) if not source_is_url else None

        if source_path is not None and source_path.exists():
            if source_path.resolve() != self.tarball_path.resolve():
                shutil.copy2(source_path, self.tarball_path)
        elif source_is_url and not self.tarball_path.exists():
            self._download_tarball(str(source))

        if not self.exclude_path.exists():
            self._download_exclusion_list()

        if not self.tarball_path.exists():
            raise RuntimeError(f"QM9 tarball not found at {self.tarball_path}")

        self.excluded_indices = self._load_exclusion_list()
        self._tar_info_cache: dict[str, tarfile.TarInfo] = {}

        if self.extract:
            if not self.extracted_path.exists() or not any(self.extracted_path.iterdir()):
                self._extract_tarball()
            self.xyz_members = self._index_files()
        else:
            self.xyz_members = self._index_archive()

        if self.total is not None and self.total < len(self.xyz_members):
            random.seed(42)
            self.xyz_members = sorted(random.sample(self.xyz_members, self.total))

        # Lazy loaded samples
        self._samples: list[Sample] | None = None

    # -- DataSource protocol ------------------------------------------------

    @property
    def source_id(self) -> str:
        return f"qm9:n={len(self)}:excl={len(self.excluded_indices)}"

    def __len__(self) -> int:
        if self._samples is not None:
            return len(self._samples)
        return len(self.xyz_members)

    def __getitem__(self, idx: int) -> Sample:
        if self._samples is None:
            self._load_all()
        assert self._samples is not None
        return self._samples[idx]

    # -- Download / IO helpers ----------------------------------------------

    def _download_tarball(self, url: str) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            self.tarball_path.write_bytes(response.read())

    def _download_exclusion_list(self) -> None:
        req = urllib.request.Request(self.EXCLUDE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            self.exclude_path.write_bytes(response.read())

    def _extract_tarball(self) -> None:
        self.extracted_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self.tarball_path, "r:bz2") as tar:
            tar.extractall(path=self.extracted_path)

    def _index_files(self) -> list[str]:
        members = [f.name for f in self.extracted_path.glob("*.xyz")]
        xyz_files: list[str] = []
        for name in members:
            try:
                mol_idx = int(name[-10:-4])
            except ValueError:
                continue
            if mol_idx not in self.excluded_indices:
                xyz_files.append(name)
        xyz_files.sort()
        return xyz_files

    def _index_archive(self) -> list[str]:
        xyz_files: list[str] = []
        with tarfile.open(self.tarball_path, "r:bz2") as tar:
            for member in tqdm(tar, desc="Indexing QM9 archive"):
                if not member.name.endswith(".xyz"):
                    continue
                try:
                    mol_idx = int(member.name[-10:-4])
                except ValueError:
                    continue
                if mol_idx in self.excluded_indices:
                    continue
                xyz_files.append(member.name)
                self._tar_info_cache[member.name] = member
        xyz_files.sort()
        return xyz_files

    def _load_exclusion_list(self) -> set[int]:
        excluded: set[int] = set()
        if self.exclude_path.exists():
            lines = self.exclude_path.read_text().splitlines()
            for line in lines[9:-1]:
                parts = line.split()
                if parts:
                    excluded.add(int(parts[0]))
        return excluded

    def _load_member_text(self, member_name: str, tar: tarfile.TarFile | None) -> str:
        if self.extract:
            return (self.extracted_path / member_name).read_text(encoding="utf-8")

        if tar is None:
            raise RuntimeError("tar file handle is required when extract=False")

        tar_info = self._tar_info_cache[member_name]
        f = tar.extractfile(tar_info)
        if f is None:
            raise RuntimeError(f"Failed to read {member_name}")
        return f.read().decode("utf-8")

    def _parse_xyz(self, content: str) -> Sample:
        content = content.replace("*^", "E")
        reader = XYZReader(io.StringIO(content), header=self.PROPERTY_NAMES)
        frame = reader.read()

        z = torch.from_numpy(frame["atoms"]["number"]).long()
        pos = torch.from_numpy(
            np.stack(
                (frame["atoms"]["x"], frame["atoms"]["y"], frame["atoms"]["z"]),
                axis=1,
            )
        ).float()

        targets: dict[str, torch.Tensor] = {}
        for key in self.PROPERTY_NAMES:
            if key in ("tag", "index"):
                continue
            targets[key] = torch.tensor([float(frame.metadata[key])], dtype=torch.float32)

        return {"Z": z, "pos": pos, "targets": targets}

    def _load_all(self) -> None:
        samples: list[Sample] = []
        tar = None
        if not self.extract:
            tar = tarfile.open(self.tarball_path, "r:bz2")
        try:
            for name in tqdm(self.xyz_members, desc="Loading QM9"):
                content = self._load_member_text(name, tar)
                samples.append(self._parse_xyz(content))
        finally:
            if tar is not None:
                tar.close()
        self._samples = samples
