"""QM9 data source: 130k small organic molecules with quantum chemical properties.

Reference:
    Ramakrishnan et al. "Quantum chemistry structures and properties of 134 kilo molecules"
    Scientific Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22

Usage pattern (new API)::

    from molix.data import pipeline, AtomicDress, NeighborList, MmapDataset
    from molix.datasets.qm9 import QM9Source, download_qm9, QM9_TARGET_SCHEMA

    # 1. Workflow-side: materialize once into a shared cache directory.
    download_qm9(data_dir)                       # optional; QM9Source does it lazily
    source = QM9Source(data_dir)
    pipe = pipeline("qm9").add(...).build()
    pipe.materialize(source, sink=cache_path)

    # 2. Training-side: zero pipeline work, just read.
    ds = MmapDataset.from_cache(cache_path)
    dm = DataModule(*ds.split(ratio=0.8),
                    target_schema=QM9_TARGET_SCHEMA, ...)
"""

from __future__ import annotations

import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import torch
from tqdm import tqdm

from molix.data.collate import TargetSchema
from molix.data.source import Sample


# All scalar properties exposed by raw QM9 records (excluding "tag" and "index").
_QM9_GRAPH_TARGETS: frozenset[str] = frozenset(
    {"A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
     "U0", "U", "H", "G", "Cv"}
)


QM9_TARGET_SCHEMA: TargetSchema = TargetSchema(
    graph_level=_QM9_GRAPH_TARGETS,
    atom_level=frozenset(),
)


# ---------------------------------------------------------------------------
# Raw loader helpers
# ---------------------------------------------------------------------------

_PROPERTY_NAMES = [
    "tag", "index",
    "A", "B", "C", "mu", "alpha",
    "homo", "lumo", "gap", "r2", "zpve",
    "U0", "U", "H", "G", "Cv",
]

_DEFAULT_URL = "https://ndownloader.figshare.com/files/3195389"
_EXCLUDE_URL = "https://figshare.com/ndownloader/files/3195404"


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        dest.write_bytes(r.read())


def _load_exclusion_list(path: Path) -> set[int]:
    excluded: set[int] = set()
    for line in path.read_text().splitlines()[9:-1]:
        parts = line.split()
        if parts:
            excluded.add(int(parts[0]))
    return excluded


def _parse_xyz(content: str) -> dict:
    from molpy.core.element import Element

    content = content.replace("*^", "E")
    lines = content.splitlines()
    natoms = int(lines[0].strip())
    prop_values = lines[1].split()
    metadata = dict(zip(_PROPERTY_NAMES, prop_values))

    symbols, xs, ys, zs = [], [], [], []
    for line in lines[2: 2 + natoms]:
        parts = line.split()
        symbols.append(parts[0])
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))
        zs.append(float(parts[3]))

    z = torch.tensor(
        [Element.get_atomic_number(s) for s in symbols], dtype=torch.long
    )
    pos = torch.tensor(list(zip(xs, ys, zs)), dtype=torch.float32)

    targets: dict[str, torch.Tensor] = {}
    for key in _PROPERTY_NAMES:
        if key in ("tag", "index"):
            continue
        targets[key] = torch.tensor([float(metadata[key])], dtype=torch.float32)

    return {"Z": z, "pos": pos, "targets": targets}


def _filter_targets(sample: dict, kept: frozenset[str]) -> dict:
    """Drop targets not in *kept* to shrink cache when user wants a subset."""
    targets = {k: v for k, v in sample.get("targets", {}).items() if k in kept}
    return {**sample, "targets": targets}


def _ensure_downloaded(root: Path) -> None:
    """Download the QM9 tarball and exclusion list if not already present.

    Idempotent — safe to call unconditionally at the start of a workflow's
    ``prepare_data`` stage.
    """
    root.mkdir(parents=True, exist_ok=True)
    tarball = root / "qm9.tar.bz2"
    exclude_file = root / "qm9_exclude.txt"
    if not tarball.exists():
        print("Downloading QM9 tarball...", flush=True)
        _download(_DEFAULT_URL, tarball)
    if not exclude_file.exists():
        _download(_EXCLUDE_URL, exclude_file)


def download_qm9(root: str | Path) -> Path:
    """Public entry point: ensure QM9 raw files are present in *root*.

    Equivalent to constructing :class:`QM9Source` purely for its download
    side-effect, but skips the expensive raw parse. Use this in a
    workflow's ``prepare_data`` stage when you want downloading decoupled
    from source construction.
    """
    root = Path(root)
    _ensure_downloaded(root)
    return root


def _load_raw(root: Path, total: int | None) -> list[dict]:
    """Return all raw QM9 samples as ``list[dict]`` (auto-downloads if needed)."""
    _ensure_downloaded(root)
    tarball = root / "qm9.tar.bz2"
    exclude_file = root / "qm9_exclude.txt"

    excluded = _load_exclusion_list(exclude_file)

    members: list[str] = []
    tar_info_cache: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(tarball, "r:bz2") as tar:
        for member in tqdm(tar, desc="Indexing QM9", file=sys.stdout):
            if not member.name.endswith(".xyz"):
                continue
            try:
                mol_idx = int(member.name[-10:-4])
            except ValueError:
                continue
            if mol_idx in excluded:
                continue
            members.append(member.name)
            tar_info_cache[member.name] = member
    members.sort()

    if total is not None and total < len(members):
        random.seed(42)
        members = sorted(random.sample(members, total))

    samples: list[dict] = []
    with tarfile.open(tarball, "r:bz2") as tar:
        for name in tqdm(members, desc="Loading QM9", file=sys.stdout):
            f = tar.extractfile(tar_info_cache[name])
            assert f is not None
            samples.append(_parse_xyz(f.read().decode("utf-8")))

    return samples


# ---------------------------------------------------------------------------
# QM9Source — raw DataSource (no pipeline, no cache)
# ---------------------------------------------------------------------------


class QM9Source:
    """DataSource exposing raw QM9 samples for pipeline consumption.

    A :class:`QM9Source` is the "dataset" layer in the molix separation: it
    only knows where raw data lives and how to index it. Preprocessing,
    caching, and DataLoader behavior are layered on top via a pipeline
    (``pipe.materialize(source, sink=...)``) and :class:`MmapDataset.from_cache`.

    Args:
        root: Directory for the raw QM9 tarball (downloaded on first use).
        total: Subsample to at most this many molecules (reproducible seed).
        targets: If given, keep only these scalar properties in each sample.
            ``None`` keeps all of :data:`_QM9_GRAPH_TARGETS`.
        download: Download raw files if missing. Set to ``False`` in
            offline / test environments.

    Attributes:
        source_id: Stable identifier used by
            :func:`~molix.data.compute_cache_identity` and recorded in the
            cache ``meta.json``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        total: int | None = None,
        targets: list[str] | tuple[str, ...] | None = None,
        download: bool = True,
    ) -> None:
        self.root = Path(root)

        if targets is not None:
            kept = frozenset(targets)
            unknown = kept - _QM9_GRAPH_TARGETS
            if unknown:
                raise ValueError(
                    f"Unknown QM9 targets {sorted(unknown)}. "
                    f"Available: {sorted(_QM9_GRAPH_TARGETS)}"
                )
            self._targets: tuple[str, ...] | None = tuple(sorted(kept))
            kept_set = kept
        else:
            self._targets = None
            kept_set = _QM9_GRAPH_TARGETS

        if download:
            _ensure_downloaded(self.root)
        elif not (self.root / "qm9.tar.bz2").exists():
            raise FileNotFoundError(
                f"QM9 tarball not found at {self.root / 'qm9.tar.bz2'}. "
                "Set download=True or call download_qm9() first."
            )

        samples = _load_raw(self.root, total)
        if self._targets is not None:
            samples = [_filter_targets(s, kept_set) for s in samples]
        self._samples = samples
        self._total = total

    # -- DataSource protocol ------------------------------------------------

    @property
    def source_id(self) -> str:
        parts = [f"qm9:v1:n={len(self._samples)}"]
        if self._targets is not None:
            parts.append(f"targets={'+'.join(self._targets)}")
        return ":".join(parts)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]
