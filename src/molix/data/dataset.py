"""Dataset base class and built-in implementations.

Choose the right subclass based on dataset size and DataLoader concurrency:

+------------------+-----------------+---------------------------+
| Class            | Dataset size    | num_workers               |
+==================+=================+===========================+
| CachedDataset    | small (<10k)    | 0 (or very low)           |
+------------------+-----------------+---------------------------+
| MmapDataset      | large (10k+)    | any — no FD limit         |
+------------------+-----------------+---------------------------+

All subclasses implement the same interface and are interchangeable in
:class:`DataModule` via the ``dataset_cls`` parameter.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import mmap as _mmap
import os as _os
import shutil as _shutil
import uuid as _uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Cache format constants
# ---------------------------------------------------------------------------

# Bump when the on-disk cache layout or meta schema changes in a
# non-backwards-compatible way. Kept in sync with identity._IDENTITY_SCHEMA_VERSION.
CACHE_SCHEMA_VERSION = 1

_SAMPLES_STEM = "samples"
_TASK_STATES_DIR = "task_states"
_META_FILE = "meta.json"
_SAMPLES_FORMAT = "molix-mmap-v1"
_IDX_FORMAT = "json-v1"


class CacheValidationError(ValueError):
    """Raised when a cache directory fails the :meth:`MmapDataset.from_cache` validation."""

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseDataset(Dataset[Any], ABC):
    """Abstract base for all molix dataset implementations.

    Subclass this — not ``torch.utils.data.Dataset`` directly — so that
    :class:`~molix.data.DataModule` can accept any implementation
    interchangeably.

    Typical flow: construct via :meth:`MmapDataset.from_cache` (read an
    asset written by :meth:`PipelineSpec.materialize`) or via
    :meth:`CachedDataset.__init__` for small in-memory collections.
    """

    @classmethod
    @abstractmethod
    def from_samples(
        cls,
        samples: list[dict],
        *,
        cache_dir: str | Path | None = None,
        name: str = "data",
    ) -> "BaseDataset":
        """Construct from pre-computed samples produced by
        :meth:`~molix.data.PipelineSpec.prepare`."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict: ...  # type: ignore[override]

    def split(
        self,
        ratio: float | None = None,
        *,
        sizes: tuple[int, ...] | None = None,
        seed: int = 42,
    ) -> tuple["SubsetDataset", ...]:
        """Split into N shuffled subsets without copying data.

        Exactly one of ``ratio`` (2-way split) or ``sizes`` (N-way split)
        must be provided.

        Args:
            ratio: Fraction of samples for the first subset. Produces a
                ``(train, val)`` 2-tuple.
            sizes: Per-subset sizes; must sum to ``len(self)``. Produces
                an N-tuple of :class:`SubsetDataset` views in the given
                order.
            seed: RNG seed for reproducible shuffling.

        Returns:
            Tuple of :class:`SubsetDataset` views over this dataset.
        """
        import torch as _torch

        if (ratio is None) == (sizes is None):
            raise ValueError("Provide exactly one of `ratio` or `sizes`.")

        n = len(self)
        gen = _torch.Generator().manual_seed(seed)
        perm = _torch.randperm(n, generator=gen).tolist()

        if ratio is not None:
            cut = int(n * ratio)
            return (SubsetDataset(self, perm[:cut]), SubsetDataset(self, perm[cut:]))

        assert sizes is not None
        if sum(sizes) != n:
            raise ValueError(f"sizes must sum to len(self)={n}, got sum={sum(sizes)}")
        parts: list[SubsetDataset] = []
        offset = 0
        for sz in sizes:
            parts.append(SubsetDataset(self, perm[offset:offset + sz]))
            offset += sz
        return tuple(parts)


# ---------------------------------------------------------------------------
# CachedDataset
# ---------------------------------------------------------------------------


class CachedDataset(BaseDataset):
    """In-memory dataset for small collections (up to ~10 k samples).

    All samples are held in a Python ``list``.  ``__getitem__`` is an
    O(1) list lookup.

    .. warning::
        Using ``num_workers > 0`` in the DataLoader forces PyTorch to share
        each tensor via OS shared memory (one file descriptor per tensor).
        On large datasets this exhausts ``ulimit -n`` and raises
        ``RuntimeError: Too many open files``.  Keep ``num_workers=0`` with
        this class, or switch to :class:`MmapDataset`.
    """

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    @classmethod
    def from_samples(
        cls,
        samples: list[dict],
        *,
        cache_dir: str | Path | None = None,  # noqa: ARG003
        name: str = "data",  # noqa: ARG003
    ) -> "CachedDataset":
        return cls(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return self._samples[idx]


# ---------------------------------------------------------------------------
# MmapDataset — standard-library mmap, no pickle of tensor data
# ---------------------------------------------------------------------------


def _serialize(
    obj: Any,
    buf: bytearray,
    desc: dict,
) -> None:
    """Recursively write tensors in *obj* into *buf*; record metadata in *desc*.

    *obj* is a ``dict`` (possibly nested) whose leaves are ``torch.Tensor``.
    Non-tensor, non-dict values are stored as-is under a ``"__scalar__"`` key
    so they round-trip cleanly through JSON.
    """
    for key, val in obj.items():
        if isinstance(val, torch.Tensor):
            t = val.contiguous().cpu()
            offset = len(buf)
            buf += t.numpy().tobytes()
            desc[key] = {
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "offset": offset,
            }
        elif isinstance(val, dict):
            desc[key] = {}
            _serialize(val, buf, desc[key])
        else:
            # scalar / string metadata — store inline in the index
            desc[key] = {"__scalar__": val}


# ---------------------------------------------------------------------------
# Cache I/O helpers (module-level, used by MmapDataset.write_cache / from_cache)
# ---------------------------------------------------------------------------

_TASK_STATE_FORMAT = "tensordict-memmap-v1"


def _is_ready(path: Path) -> bool:
    meta_path = path / _META_FILE
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("status") == "ready"
    except (json.JSONDecodeError, OSError):
        return False


def _validate_cache(path: Path) -> None:
    """Validate a cache directory end-to-end.

    ``_READY`` alone is a commit sentinel, not a full validation — we also
    check the meta schema, required files, and that every task-state
    directory listed in the manifest actually exists.
    """
    if not path.is_dir():
        raise CacheValidationError(f"cache path is not a directory: {path}")

    meta_path = path / _META_FILE
    if not meta_path.exists():
        raise CacheValidationError(f"missing meta.json: {path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CacheValidationError(f"meta.json is not valid JSON at {path}") from e

    schema = meta.get("schema_version")
    if schema != CACHE_SCHEMA_VERSION:
        raise CacheValidationError(
            f"unsupported cache schema_version={schema!r} "
            f"(expected {CACHE_SCHEMA_VERSION}) at {path}"
        )
    status = meta.get("status")
    if status != "ready":
        raise CacheValidationError(
            f"cache status={status!r}, not 'ready' at {path}"
        )

    for rel in (f"{_SAMPLES_STEM}.bin", f"{_SAMPLES_STEM}.idx"):
        p = path / rel
        if not p.exists() or p.stat().st_size == 0:
            raise CacheValidationError(f"missing or empty sample file: {p}")

    for name, info in (meta.get("task_states") or {}).items():
        fmt = info.get("format")
        if fmt != _TASK_STATE_FORMAT:
            raise CacheValidationError(
                f"task_state {name!r} has unsupported format={fmt!r}"
            )
        tdir = path / info["dir"]
        if not tdir.is_dir():
            raise CacheValidationError(
                f"missing task_state directory for {name!r}: {tdir}"
            )


def _write_task_state(dest: Path, state: Any) -> None:
    """Serialise *state* (TensorDict or ``dict[str, Tensor]``) to *dest* via memmap."""
    from tensordict import TensorDict  # local import — tensordict is heavy

    if not isinstance(state, TensorDict):
        state = TensorDict.from_dict(dict(state))
    dest.mkdir(parents=True, exist_ok=True)
    state.memmap(dest)


def _load_task_states(
    root: Path, manifest: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    if not manifest:
        return {}
    from tensordict import TensorDict

    states: dict[str, Any] = {}
    for name, info in manifest.items():
        states[name] = TensorDict.load_memmap(root / info["dir"])
    return states


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync so rename-based commits are durable."""
    try:
        fd = _os.open(str(path), _os.O_RDONLY)
    except OSError:
        return
    try:
        _os.fsync(fd)
    except OSError:
        pass
    finally:
        _os.close(fd)


def _molix_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("molix")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------


def _reconstruct(desc: dict, mm: _mmap.mmap) -> dict:
    """Reconstruct a sample dict from the mmap *mm* using *desc* as index."""
    result: dict = {}
    for key, val in desc.items():
        if "__scalar__" in val:
            result[key] = val["__scalar__"]
        elif "dtype" in val:
            dtype = _STR_TO_DTYPE[val["dtype"]]
            shape: list[int] = val["shape"]
            numel = math.prod(shape) if shape else 1
            offset: int = val["offset"]
            # torch.frombuffer returns a 1-D tensor sharing the mmap pages —
            # no copy, no file descriptor, no shared-memory handshake.
            t = torch.frombuffer(mm, dtype=dtype, count=numel, offset=offset)
            result[key] = t.reshape(shape) if shape else t.reshape([])
        else:
            # nested dict
            result[key] = _reconstruct(val, mm)
    return result


class MmapDataset(BaseDataset):
    """Memory-mapped dataset using standard-library ``mmap``.

    **On-disk layout** (written once by :meth:`from_samples`, reused on
    subsequent runs):

    * ``<name>.bin`` — raw tensor bytes, tightly packed in sample order.
    * ``<name>.idx`` — JSON index: ``list[dict]`` mapping each sample to
      ``{key: {dtype, shape, offset}}`` entries.

    ``__getitem__`` calls ``torch.frombuffer(mmap_object, ...)`` at the
    recorded byte offset.  Tensor data is never copied into a Python object
    and never passes through pickle.  Workers access the same OS page-cache
    pages without opening additional file descriptors, so ``num_workers`` is
    effectively unlimited.

    The dataset is fork- and spawn-safe: ``__getstate__`` / ``__setstate__``
    close the mmap handle before pickling and reopen it in each worker.

    Args:
        stem: Path prefix for the backing files, **without** extension
            (e.g. ``Path("cache/train")`` → ``cache/train.bin`` +
            ``cache/train.idx``).
    """

    def __init__(self, stem: str | Path) -> None:
        self._bin_path = Path(stem).with_suffix(".bin")
        self._idx_path = Path(stem).with_suffix(".idx")
        with open(self._idx_path, encoding="utf-8") as f:
            self._index: list[dict] = json.load(f)
        self._open_mmap()

    # -- mmap lifecycle -------------------------------------------------------

    def _open_mmap(self) -> None:
        self._f = open(self._bin_path, "rb")  # noqa: SIM115 — kept open intentionally
        # ACCESS_COPY: copy-on-write — file is never modified, but PyTorch can
        # treat the pages as writable (needed to silence torch.frombuffer warning
        # about non-writable buffers; any apparent writes go to private pages).
        self._mm = _mmap.mmap(self._f.fileno(), 0, access=_mmap.ACCESS_COPY)

    def _close_mmap(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self._close_mmap()

    # -- pickle support for DataLoader spawn workers --------------------------

    def __getstate__(self) -> dict:
        # Exclude the open file handle and mmap — workers reopen them.
        # Preserve all other attributes (including any subclass additions like
        # QM9Dataset._ready) so they survive the pickle round-trip.
        state = self.__dict__.copy()
        state.pop("_mm", None)
        state.pop("_f", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_mmap()

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        samples: list[dict],
        *,
        cache_dir: str | Path | None = None,
        name: str = "data",
    ) -> "MmapDataset":
        """Serialize *samples* to ``<cache_dir>/<name>.bin`` / ``.idx`` and
        return an :class:`MmapDataset` backed by those files.

        Existing files are reused without re-serializing (same semantics as
        :meth:`~molix.data.PipelineSpec.prepare`'s disk cache).

        Args:
            samples: Pre-computed sample dicts from
                :meth:`~molix.data.PipelineSpec.prepare`.
            cache_dir: Directory for the backing files.  **Required.**
            name: Split name used to derive file names
                (``"train"`` → ``train.bin`` / ``train.idx``).
        """
        if cache_dir is None:
            raise ValueError(
                "MmapDataset requires cache_dir to be set on DataModule. "
                "Pass cache_dir='./cache' (or any writable path) when "
                "constructing DataModule."
            )
        stem = Path(cache_dir) / name
        bin_path = stem.with_suffix(".bin")
        idx_path = stem.with_suffix(".idx")
        stem.parent.mkdir(parents=True, exist_ok=True)

        if not bin_path.exists() or not idx_path.exists():
            buf: bytearray = bytearray()
            index: list[dict] = []
            for sample in samples:
                desc: dict = {}
                _serialize(sample, buf, desc)
                index.append(desc)
            bin_path.write_bytes(buf)
            idx_path.write_text(json.dumps(index), encoding="utf-8")

        return cls(stem)

    # -- Dataset interface ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return _reconstruct(self._index[idx], self._mm)

    # -- Standard cache format: write_cache / from_cache ---------------------

    @classmethod
    def write_cache(
        cls,
        sink: str | Path,
        samples: Iterable[dict],
        *,
        pipeline_id: str,
        source_id: str,
        fit_source_id: str | None = None,
        pipeline_spec: Mapping[str, Any] | None = None,
        task_states: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write *samples* + metadata + task_states to *sink* in the standard molix cache format.

        Atomic: data is written to a sibling ``<sink>.partial.<uuid>`` directory,
        fsynced, then ``os.rename``'d into place. On a ready *sink*, this is a
        no-op unless ``overwrite=True``.

        This is a pure *writer*: it does not know about workspaces, cache
        identity, or directory naming — those are the caller's concern. Any
        producer of sample dicts (pipelines, tests, ad-hoc scripts) may call it.

        Args:
            sink: Target directory (resolved — the caller decides where it lives).
            samples: Iterable of processed sample dicts whose leaves are tensors
                or JSON-safe scalars. Consumed once; lists are fine.
            pipeline_id: Content id of the transform that produced *samples*.
            source_id: Id of the raw source samples were drawn from.
            fit_source_id: Id of the fit-source (defaults to *source_id*).
            pipeline_spec: Optional serialised pipeline description for debug.
            task_states: Mapping ``{name: TensorDict | dict[str, Tensor]}``.
                Plain dicts are wrapped via :meth:`TensorDict.from_dict`.
            overwrite: If True, replace a ready sink; else return early.
        """
        sink = Path(sink)

        if _is_ready(sink) and not overwrite:
            return

        sink.parent.mkdir(parents=True, exist_ok=True)
        tmp = sink.parent / f"{sink.name}.partial.{_uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)

        try:
            buf: bytearray = bytearray()
            index: list[dict] = []
            n = 0
            for sample in samples:
                desc: dict = {}
                _serialize(sample, buf, desc)
                index.append(desc)
                n += 1
            (tmp / f"{_SAMPLES_STEM}.bin").write_bytes(buf)
            (tmp / f"{_SAMPLES_STEM}.idx").write_text(
                json.dumps(index), encoding="utf-8"
            )

            task_states_manifest: dict[str, dict[str, str]] = {}
            if task_states:
                states_root = tmp / _TASK_STATES_DIR
                states_root.mkdir()
                for name, state in task_states.items():
                    state_dir = states_root / name
                    _write_task_state(state_dir, state)
                    task_states_manifest[name] = {
                        "dir": f"{_TASK_STATES_DIR}/{name}",
                        "format": _TASK_STATE_FORMAT,
                    }

            meta = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "molix_version": _molix_version(),
                "status": "ready",
                "pipeline_id": pipeline_id,
                "source_id": source_id,
                "fit_source_id": (
                    fit_source_id if fit_source_id is not None else source_id
                ),
                "n_samples": n,
                "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "pipeline_spec": dict(pipeline_spec)
                if pipeline_spec is not None
                else None,
                "task_states": task_states_manifest,
                "storage": {
                    "samples_format": _SAMPLES_FORMAT,
                    "idx_format": _IDX_FORMAT,
                },
            }
            (tmp / _META_FILE).write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            _fsync_dir(tmp)

            if sink.exists():
                # overwrite path: replace a pre-existing ready directory.
                _shutil.rmtree(sink)
            _os.rename(tmp, sink)
        except BaseException:
            _shutil.rmtree(tmp, ignore_errors=True)
            raise

    @classmethod
    def from_cache(cls, path: str | Path) -> "MmapDataset":
        """Load an :class:`MmapDataset` from a cache directory in the standard format.

        Validates (in order): ``meta.json`` parses, ``schema_version``
        matches, ``status == "ready"``, required sample files exist and are
        non-empty, every task-state directory listed in ``meta.json`` exists.

        Args:
            path: Cache directory written by :meth:`write_cache` (or any
                equivalent producer of the standard format).

        Returns:
            A fully constructed dataset whose :attr:`meta` and
            :attr:`task_states` attributes reflect the on-disk manifest.
        """
        path = Path(path)
        _validate_cache(path)

        meta = json.loads((path / _META_FILE).read_text(encoding="utf-8"))

        inst = cls(path / _SAMPLES_STEM)
        inst._meta = meta
        inst._task_states = _load_task_states(
            path, meta.get("task_states", {}) or {}
        )
        return inst

    # -- Cache metadata accessors (populated by from_cache) ------------------

    @property
    def meta(self) -> Mapping[str, Any]:
        """Cache ``meta.json`` contents; empty dict if not loaded via :meth:`from_cache`."""
        return getattr(self, "_meta", {})

    @property
    def task_states(self) -> Mapping[str, Any]:
        """Mapping ``{task_name: TensorDict}`` loaded from the cache."""
        return getattr(self, "_task_states", {})

    def get_task_state(self, name: str) -> Any:
        """Return the state for ``name`` (raises :class:`KeyError` if absent)."""
        return self.task_states[name]


# ---------------------------------------------------------------------------
# SubsetDataset — index-based view, no data copy
# ---------------------------------------------------------------------------


class SubsetDataset(BaseDataset):
    """Read-only index view into another :class:`BaseDataset`.

    Created by :meth:`BaseDataset.split`; not meant to be constructed
    directly.  Workers access the underlying dataset (and its mmap, if any)
    through the remapped index — no data is copied.

    ``from_samples`` is intentionally unsupported: a subset is always derived
    from an existing dataset, never built from raw samples.
    """

    def __init__(self, dataset: BaseDataset, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attribute access (e.g. dataset-declared target_schema)
        # to the wrapped dataset so DataModule auto-discovery works on subsets.
        # Guard against private/dunder names — in particular, during unpickling
        # __setstate__ has not yet populated __dict__, so accessing self._dataset
        # would loop forever. Raising AttributeError lets pickle proceed normally.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    @classmethod
    def from_samples(cls, samples: list[dict], **kwargs: Any) -> "SubsetDataset":  # type: ignore[override]
        raise TypeError(
            "SubsetDataset cannot be constructed from raw samples. "
            "Call BaseDataset.split() instead."
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return self._dataset[self._indices[idx]]
