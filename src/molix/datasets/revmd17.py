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
    from molix.datasets import RevMD17Source

    source = RevMD17Source(data_dir, molecule="aspirin")
    pipe = Pipeline("revmd17-aspirin").add(NeighborList(cutoff=7.0)).build()
    packed = pipe.cache(source, base_dir=run_dir / "cache")
    ds = MmapDataset(packed.sink)
    # RevMD17Source.TARGET_SCHEMA exposes graph {"energy"} + atom {"forces"}.
"""

from __future__ import annotations

import contextlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

from molix.data.collate import TargetSchema
from molix.data.source import Sample

_TARBALL_NAME = "rmd17.tar.bz2"
_FIGSHARE_ARTICLE_ID = 12672038
_FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038"
# Per-file download URLs are resolved through the figshare REST API: the public
# ``ndownloader`` HTML endpoint sits behind an anti-bot WAF that answers scripted
# GETs with an empty ``202``, whereas the API returns direct CDN links.
_FIGSHARE_FILES_API = f"https://api.figshare.com/v2/articles/{_FIGSHARE_ARTICLE_ID}/files"


@contextlib.contextmanager
def _download_lock(lock_path: Path):
    """Best-effort inter-process lock so concurrent jobs download once.

    Uses ``fcntl.flock`` on a sidecar file; degrades to a no-op where flock is
    unavailable (non-POSIX, some network filesystems). Correctness never relies
    on the lock — the caller re-checks the target exists and renames atomically —
    it only avoids redundant simultaneous downloads.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:  # pragma: no cover — flock unsupported on this fs
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)


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

    TARGET_SCHEMA: TargetSchema = TargetSchema(
        graph_level=frozenset({"energy"}),
        atom_level=frozenset({"forces"}),
    )

    @classmethod
    def _materialise(cls, root: Path, filename: str) -> None:
        """Download the per-molecule NPZ straight from the figshare record.

        Self-contained — molix.datasets depends only on the stdlib for I/O, not
        on any sibling package. Resolves download URLs via the figshare REST API
        (the HTML ``ndownloader`` endpoint is WAF-gated and 202s scripted GETs;
        the API's CDN links are not). Eight of the ten molecules are published as
        standalone NPZ files and fetched directly; the remaining two (salicylic,
        uracil) live only inside the ``rmd17.tar.bz2`` archive, so for those the
        archive is downloaded (cached for reuse) and the member extracted. Every
        write goes through a ``*.part`` temp file renamed atomically into place,
        and a ``*.lock`` sidecar serialises concurrent SLURM jobs so they
        download once instead of clobbering each other.

        Reference:
            figshare article 12672038 — Christensen & von Lilienfeld, revMD17.
        """
        target = root / filename
        with _download_lock(root / f"{filename}.lock"):
            if target.exists():  # another job won the race while we waited
                return
            req = urllib.request.Request(_FIGSHARE_FILES_API, headers={"User-Agent": "molix"})
            with urllib.request.urlopen(req) as resp:  # noqa: S310 — pinned figshare API
                files = json.load(resp)
            by_name = {f["name"]: f["download_url"] for f in files}
            if filename in by_name:
                tmp = root / f"{filename}.part"
                urllib.request.urlretrieve(by_name[filename], tmp)  # noqa: S310 — figshare CDN
                tmp.replace(target)
                return
            # Not published standalone — fall back to the archive + extract.
            archive_url = by_name.get(_TARBALL_NAME)
            if archive_url is None:
                raise FileNotFoundError(
                    f"revMD17: neither {filename!r} nor {_TARBALL_NAME!r} listed in "
                    f"figshare article {_FIGSHARE_ARTICLE_ID}; browse {_FIGSHARE_ARTICLE}"
                )
            tarball = root / _TARBALL_NAME
            if not tarball.exists():
                tmp = root / f"{_TARBALL_NAME}.part"
                urllib.request.urlretrieve(archive_url, tmp)  # noqa: S310 — figshare CDN
                tmp.replace(tarball)
            with tarfile.open(tarball, "r:bz2") as tar:
                member = tar.extractfile(f"rmd17/npz_data/{filename}")
                if member is None:
                    raise FileNotFoundError(f"revMD17: {filename!r} absent from {_TARBALL_NAME}")
                tmp = root / f"{filename}.part"
                tmp.write_bytes(member.read())
                tmp.replace(target)

    def __init__(
        self,
        root: str | Path,
        molecule: str = "aspirin",
        *,
        total: int | None = None,
        download: bool = True,
    ) -> None:
        if molecule not in _MOLECULES:
            raise ValueError(
                f"Unknown revMD17 molecule '{molecule}'. Available: {sorted(_MOLECULES)}"
            )
        self.root = Path(root)
        self.molecule = molecule
        self.filename = _MOLECULES[molecule]
        self.filepath = self.root / self.filename
        self.root.mkdir(parents=True, exist_ok=True)

        if download and not self.filepath.exists():
            self._materialise(self.root, self.filename)
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
        # Subset for smokes / quick tests (deterministic first-N frames).
        if total is not None and total < self._R.shape[0]:
            self._R = self._R[:total]
            self._E = self._E[:total]
            self._F = self._F[:total]
        self.total = total

    @property
    def source_id(self) -> str:
        """Cache-key identity ``revmd17:<molecule>:size=<bytes>:n=<n>``.

        Appends ``:total=<n>`` when a subset was requested, so subsetted
        and full sources get distinct caches.
        """
        size = self.filepath.stat().st_size
        sid = f"revmd17:{self.molecule}:size={size}:n={len(self)}"
        if self.total is not None:
            sid += f":total={self.total}"
        return sid

    def __len__(self) -> int:
        return int(self._R.shape[0])

    def __getitem__(self, idx: int) -> Sample:
        """Return the ``idx``-th trajectory frame as a flat sample dict.

        The atomic numbers ``Z`` are shared across frames (a single
        molecule). Returns ``Z``, ``pos`` ``(N, 3)`` for this frame, and a
        ``targets`` sub-dict with ``energy`` ``(1,)`` and ``forces``
        ``(N, 3)`` (kcal/mol and kcal/(mol·Å) as distributed).
        """
        return {
            "Z": self._z,
            "pos": self._R[idx],
            "targets": {
                "energy": self._E[idx : idx + 1],
                "forces": self._F[idx],
            },
        }


__all__ = ["RevMD17Source"]
