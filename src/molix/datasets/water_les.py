"""WaterLESSource: bulk liquid water (RPBE-D3) extxyz loader, molpy-only stack.

Reference:
    Cheng B., *Latent Ewald summation for machine-learning potentials*
    npj Comput. Mater. **11**:80 (2025), doi:10.1038/s41524-025-01577-7.

The upstream Cheng-UCB ``les_fit`` repository ships two extended-XYZ files:

* ``train-H2O_RPBE-D3.xyz`` — pooled training + validation snapshots.
  Internally split 0.95 / 0.05 by a deterministic **tail-slice**
  (the last ``ceil(0.05 * n_total)`` frames go to val, no shuffle),
  matching the upstream ``lr_r45_nlayer3_lmax2.yaml`` recipe.
* ``test-H2O_RPBE-D3.xyz`` — independent held-out test split.

Each frame is a 64-H₂O periodic cubic box (~12 Å edge), 192 atoms.
Energies are in eV, forces in eV·Å⁻¹, positions and cell in Å.

This source returns a **flat** sample dict per the periodic-data contract::

    {
        "Z":      torch.LongTensor((N,)),         # atomic numbers
        "pos":    torch.FloatTensor((N, 3)),       # Å
        "cell":   torch.FloatTensor((3, 3)),       # Å, row-major
        "energy": float,                           # eV (Python scalar)
        "forces": torch.FloatTensor((N, 3)),       # eV·Å⁻¹
    }

The flat top-level ``energy``/``forces`` shape (rather than ``RevMD17``'s
nested ``targets`` sub-dict) is the periodic-water delta from non-periodic
revMD17 samples; downstream consumers know which contract applies via
:attr:`WaterLESSource.TARGET_SCHEMA`.

Usage::

    from molix.datasets import WaterLESSource

    train = WaterLESSource("~/datasets/water_les", split="train")
    val   = WaterLESSource("~/datasets/water_les", split="val")
    test  = WaterLESSource("~/datasets/water_les", split="test")

See ``src/molix/datasets/_data_acquisition.md`` for manual-fetch
instructions and SHA-256 placeholders.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path
from typing import Literal

import torch
from molpy.core.element import Element

from molix.data.collate import TargetSchema
from molix.data.source import Sample
from molix.datasets._extxyz import ExtxyzFrame, parse_extxyz_frames

__all__ = ["WaterLESSource"]


Split = Literal["train", "val", "test"]


class WaterLESSource:
    """DataSource for bulk-water RPBE-D3 from the LES paper benchmark suite.

    Args:
        root: Directory containing ``train-H2O_RPBE-D3.xyz`` and / or
            ``test-H2O_RPBE-D3.xyz``. Used as the download destination
            when ``download=True``.
        split: One of ``"train"`` / ``"val"`` / ``"test"``. ``"train"`` and
            ``"val"`` slice the same upstream ``train-…`` file via the
            deterministic tail-slice with :attr:`TRAIN_VAL_RATIO`.
        download: If ``True``, fetch any missing file from
            :attr:`BASE_URL` via :mod:`urllib.request` (no ASE).
        verify_checksum: If ``True``, compute SHA-256 of every consumed
            file and compare against :attr:`_CHECKSUMS`. Default ``False``
            so the placeholder digests in ``_data_acquisition.md`` do not
            force every contributor to disable verification by hand.

    Raises:
        ValueError: Unknown ``split``, or ``verify_checksum=True`` and the
            file hash does not match the pinned digest (message contains
            both expected and actual 64-hex strings).
        FileNotFoundError: ``download=False`` and a required file is
            missing.
    """

    #: Canonical upstream basename for the train/val pool.
    TRAIN_FILE: str = "train-H2O_RPBE-D3.xyz"
    #: Canonical upstream basename for the held-out test split.
    TEST_FILE: str = "test-H2O_RPBE-D3.xyz"
    #: Train / val deterministic tail-slice ratio.
    TRAIN_VAL_RATIO: tuple[float, float] = (0.95, 0.05)
    #: Raw-file base URL on the ChengUCB/les_fit GitHub mirror.
    BASE_URL: str = "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark"
    #: Pinned SHA-256 digests; placeholder ``"0" * 64`` until a
    #: contributor lands real values via the recipe in
    #: ``_data_acquisition.md``. Tests inject real digests via
    #: ``monkeypatch.setattr(WaterLESSource, "_CHECKSUMS", ...)``.
    _CHECKSUMS: dict[str, str] = {
        "train-H2O_RPBE-D3.xyz": "0" * 64,
        "test-H2O_RPBE-D3.xyz": "0" * 64,
    }

    TARGET_SCHEMA: TargetSchema = TargetSchema(
        graph_level=frozenset({"energy"}),
        atom_level=frozenset({"forces"}),
    )

    _VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})

    def __init__(
        self,
        root: str | Path,
        *,
        split: Split,
        download: bool = False,
        verify_checksum: bool = False,
    ) -> None:
        if split not in self._VALID_SPLITS:
            raise ValueError(
                f"WaterLESSource: unknown split {split!r}; "
                f"expected one of {sorted(self._VALID_SPLITS)}"
            )
        self.root = Path(root).expanduser()
        self.split: Split = split
        self._download = download
        self._verify_checksum = verify_checksum

        source_file = self.TEST_FILE if split == "test" else self.TRAIN_FILE
        self._source_path = self.root / source_file
        self._ensure_present(self._source_path, source_file)
        if verify_checksum:
            self._verify(self._source_path, source_file)

        frames = parse_extxyz_frames(self._source_path)
        # Per the source-layer contract: forces are required by TARGET_SCHEMA
        # (atom_level frozenset({"forces"})). Sources missing them are
        # malformed for this consumer; refuse early with a clear message.
        for j, f in enumerate(frames):
            if f.forces is None:
                raise ValueError(
                    f"WaterLESSource: frame {j} in {self._source_path} declares "
                    f"no forces column in Properties; cannot satisfy TARGET_SCHEMA"
                )
        self._all_frames = frames

        if split == "test":
            self._slice = slice(0, len(frames))
        else:
            n_total = len(frames)
            n_train = int(math.ceil(n_total * self.TRAIN_VAL_RATIO[0]))
            if split == "train":
                self._slice = slice(0, n_train)
            else:  # "val"
                self._slice = slice(n_train, n_total)

        self._frame_indices = list(range(self._slice.start, self._slice.stop))
        self._file_size = self._source_path.stat().st_size

    # ------------------------------------------------------------------ DataSource

    @property
    def source_id(self) -> str:
        """Stable, deterministic cache-key string.

        Format: ``water_les:split=<split>:size=<bytes>:n=<n_samples>``.
        Two splits sharing a backing file (``train`` and ``val``) still
        differ by the ``split=`` tag; ``test`` further differs by file
        size and sample count.
        """
        return f"water_les:split={self.split}:size={self._file_size}:n={len(self)}"

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __getitem__(self, idx: int) -> Sample:
        """Return the ``idx``-th frame as a flat-``dict`` sample.

        Builds ``Z`` (atomic numbers), positions, cell, and the energy /
        forces targets from the underlying extxyz frame.
        """
        frame: ExtxyzFrame = self._all_frames[self._frame_indices[idx]]
        Z = torch.tensor(
            [Element.get_atomic_number(s) for s in frame.species],
            dtype=torch.long,
        )
        # Forces presence is enforced in __init__; static type-narrow here.
        assert frame.forces is not None
        return {
            "Z": Z,
            "pos": torch.from_numpy(frame.pos).to(dtype=torch.float32),
            "cell": torch.from_numpy(frame.cell).to(dtype=torch.float32),
            "energy": float(frame.energy),
            "forces": torch.from_numpy(frame.forces).to(dtype=torch.float32),
        }

    # ------------------------------------------------------------------ helpers

    def _ensure_present(self, path: Path, filename: str) -> None:
        if path.exists():
            return
        if not self._download:
            raise FileNotFoundError(
                f"WaterLESSource: {path} not found. Either pre-fetch from "
                f"{self.BASE_URL}/{filename} (see _data_acquisition.md) or "
                f"pass download=True."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{filename}"
        urllib.request.urlretrieve(url, path)  # noqa: S310 — pinned upstream URL

    def _verify(self, path: Path, filename: str) -> None:
        expected = self._CHECKSUMS.get(filename)
        if expected is None:
            raise ValueError(
                f"WaterLESSource: no pinned checksum for {filename!r}; "
                f"known keys: {sorted(self._CHECKSUMS)}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"WaterLESSource: checksum mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
