"""Short-lived pipeline output cache.

**Scope**: one training run. A :class:`PackedCache` holds a single on-disk
file that stores the output of a :class:`~molix.data.pipeline.PipelineSpec`
applied to a :class:`~molix.data.source.DataSource`. It exists so that
expensive preprocessing (neighbor lists, atomic dress, etc.) is computed
*once* at the start of a run and reused for every training step thereafter.
It is explicitly **not** a persistence format:

* no schema version, no ``meta.json``, no ``_READY`` sentinel;
* no validation beyond "file exists and opens";
* no cross-version / cross-run compatibility guarantees;
* the workflow owns placement and invalidation — typical placement is under
  ``run_ctx.run_dir / "cache"`` so the file is naturally scoped to the run.

If long-term persistence matters, use :mod:`molix.datasets` (curated
datasets) instead.

Single-file layout — one ``<sink>.pt`` per cache, written via
``torch.save`` and read via ``torch.load(mmap=True)``. A failed write never
leaves a partial file behind (atomic ``os.rename``).

Rank / DDP coordination is owned by
:meth:`molix.data.pipeline.PipelineSpec.cache` — user code never touches the
rank env var directly.

Why single-file packed-bucket and not ``TensorDict.memmap_()``
--------------------------------------------------------------

``LazyStackedTensorDict.memmap_(prefix)`` writes **one subdirectory per
sample** (``prefix/<i>/<key>.memmap`` + companion ``.shape.memmap`` for
variable shapes). For QM9-scale datasets (130k molecules × ~8 leaves per
sample ≈ 1M files + ~260k directories) this blows past typical inode
budgets on HPC shared filesystems and makes ``rsync`` / ``tar`` / ``rm``
painfully slow.

:class:`PackedCache` stores ``O(schema_keys)`` concatenated tensors in a
**single** file, with ``atom_ptr`` / ``edge_ptr`` cumsum pointers locating
each sample's slice. Zero-copy mmap read is preserved
(``torch.load(mmap=True)`` on the packed tensors); one file replaces the
ragged directory tree. The layout is purpose-built for fixed-schema
molecular datasets; ``TensorDict.memmap_`` is purpose-built for
heterogeneous RLHF-style rollout buffers. They solve different problems.

**Do not migrate the cache format to ``TensorDict.memmap_``.** If you
need TD-native memmap semantics for a specific downstream use case, add a
thin view/adapter on top of :class:`PackedCache` rather than changing the
on-disk format.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch

__all__ = ["PackedCache"]


# Reserved keys in the packed payload — never collide with user sample keys.
_RESERVED_TOP_KEYS = frozenset(
    {
        "format_version",
        "n_samples",
        "atom_ptr",
        "edge_ptr",
        "atoms",
        "edges",
        "graphs",
        "scalars",
        "schema",
        "task_states",
    }
)


class PackedCache:
    """One on-disk cache file in packed layout.

    A :class:`PackedCache` is a thin OOP facade over a single ``.pt`` file:
    it owns readiness checks, atomic save, mmap-backed load, DDP-friendly
    polling, and per-sample unpacking. The cache is identified by its
    *sink* path alone — identity hashing is handled by
    :meth:`molix.data.pipeline.Node.cache_key` and normally accessed
    through :meth:`molix.data.pipeline.PipelineSpec.cache_key`.

    On-disk layout is a **packed** representation: each sample key is
    concatenated across all samples into a single large tensor (per-atom,
    per-edge, or per-graph depending on its shape), with ``atom_ptr`` /
    ``edge_ptr`` cumsum indices locating each sample's slice. That keeps
    the per-sample-object overhead off the critical path on load (one
    :func:`torch.load` deserialises O(schema_keys) tensors instead of
    O(n_samples × n_keys)).
    """

    # Packed cache format version. Bump on incompatible layout changes so
    # loaders can reject stale sinks instead of silently misreading.
    FORMAT_VERSION: ClassVar[int] = 2

    __slots__ = ("_sink",)

    def __init__(self, sink: str | Path) -> None:
        self._sink = Path(sink)

    # -- identity ----------------------------------------------------------

    @property
    def sink(self) -> Path:
        """The file path backing this cache."""
        return self._sink

    def __fspath__(self) -> str:
        """Return the backing sink path as a string (``os.PathLike`` protocol).

        Lets a :class:`PackedCache` be passed directly to ``open``,
        ``torch.load``, and other path-accepting APIs.
        """
        return os.fspath(self._sink)

    def __repr__(self) -> str:
        return f"PackedCache(sink={self._sink!s})"

    # -- readiness & DDP polling ------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if the sink is a readable cache file.

        Readable = exists, is a regular file, and has non-zero size.
        Unpickle safety is deferred to :meth:`load` — a half-written file
        would fail there, which is fine: the workflow would treat it as
        not-ready and rebuild.
        """
        try:
            return self._sink.is_file() and self._sink.stat().st_size > 0
        except OSError:
            return False

    def wait_until_ready(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Block until :meth:`is_ready` returns ``True``, or raise.

        Used by non-primary DDP ranks while rank 0 materialises the cache.

        Args:
            timeout: Maximum seconds to wait before raising
                :class:`TimeoutError`.
            poll_interval: Seconds between readiness probes.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_ready():
                return
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Timed out waiting {timeout:.0f}s for cache at {self._sink}. "
            "PipelineSpec.cache() must be driven from rank 0 (e.g. a "
            "prepare_data stage) before workers start."
        )

    # -- IO ----------------------------------------------------------------

    def save(
        self,
        samples: list[dict],
        *,
        task_states: Mapping[str, Mapping[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Serialize *samples* (+ optional *task_states*) atomically.

        Writes ``<sink>.partial.<uuid>``, fsyncs, then ``os.rename`` onto
        the sink — single-file rename is POSIX-atomic, so observers never
        see a partial file.

        Args:
            samples: Processed sample dicts. Leaves must be
                ``torch.Tensor`` or JSON-safe scalars
                (``int``/``float``/``str``/``bool``) so that downstream
                :meth:`load` can use ``weights_only=True``. Every sample
                must share the same schema (same keys, same per-atom /
                per-edge / per-graph classification per key); the first
                sample is taken as the reference schema.
            task_states: Optional fitted state for
                :class:`~molix.data.task.DatasetTask` instances, typically
                produced by
                :meth:`~molix.data.pipeline.PipelineSpec.collect_task_states`.
            overwrite: If the sink already exists, replace it. Otherwise
                keep the existing file (no-op).
        """
        if self._sink.exists() and not overwrite:
            return

        self._sink.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._sink.parent / f"{self._sink.name}.partial.{uuid.uuid4().hex[:8]}"

        payload = _pack_samples(list(samples))
        if task_states:
            payload["task_states"] = {k: dict(v) for k, v in task_states.items()}

        try:
            torch.save(payload, tmp)
            _fsync_file(tmp)
            os.replace(tmp, self._sink)  # atomic on POSIX (incl. same-mount NFS)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def load(self, *, mmap: bool = True) -> dict[str, Any]:
        """Load this cache (packed format).

        Args:
            mmap: Memory-map the tensor storages (default). Set to
                ``False`` for a full in-memory copy.

        Returns:
            Packed payload dict with keys::

                {
                    "format_version": int,
                    "n_samples": int,
                    "schema": {key: spec, ...},
                    "atom_ptr": LongTensor(n_samples+1),   # absent if no per-atom keys
                    "edge_ptr": LongTensor(n_samples+1),   # absent if no per-edge keys
                    "atoms":   {key: concat tensor, ...},
                    "edges":   {key: concat tensor, ...},
                    "graphs":  {key: stacked tensor, ...},
                    "scalars": {key: [v0, v1, ...], ...},  # non-tensor values
                    "task_states": {...},                  # optional
                }

            Also exposes ``payload["samples"]`` as a lazy list-like view
            (each access reconstructs a sample dict on the fly). Mostly
            for tests and compatibility — hot paths should use
            :meth:`unpack_sample` directly to skip constructing an
            intermediate list.
        """
        payload = torch.load(self._sink, mmap=mmap, weights_only=True)

        version = payload.get("format_version")
        if version != self.FORMAT_VERSION:
            raise ValueError(
                f"Cache file {self._sink} has format_version={version!r}, "
                f"expected {self.FORMAT_VERSION}. The packed cache format "
                "changed — delete the stale cache and rerun the pipeline."
            )
        payload["samples"] = _LazySampleView(payload)
        return payload

    @staticmethod
    def unpack_sample(payload: Mapping[str, Any], idx: int) -> dict:
        """Reconstruct the ``idx``-th sample dict from a packed payload.

        Slices into the shared concat tensors — no storage copy when the
        payload was loaded via ``torch.load(mmap=True)``.
        """
        return _unpack_one(payload, idx)


# ---------------------------------------------------------------------------
# Packed format: pack / unpack helpers (module-private implementation)
# ---------------------------------------------------------------------------


# Schema spec per key is a tuple:
#   ("atom",   dtype, extra_shape)   → concat on dim 0 across samples
#   ("edge",   dtype, extra_shape)   → concat on dim 0 across samples
#   ("graph",  dtype, shape)         → stack on new leading dim across samples
#   ("scalar", py_type)              → Python int/float/bool/str list
#
# ``extra_shape`` for atom/edge is the shape *after* dim 0 (e.g. (3,) for pos).
# ``shape`` for graph is the full per-sample tensor shape (e.g. (1,) or ()).
# Keys are ``str`` with "." separators for nested dict paths.


def _pack_samples(samples: list[dict]) -> dict[str, Any]:
    """Convert a list of sample dicts into the packed payload format.

    Schema classification scans *all* samples so a key's shape[0] is:

    * **atom** — tracks ``len(Z)`` for every sample (concat along dim 0);
    * **edge** — tracks ``len(edge_index)`` for every sample (concat along dim 0);
    * **graph** — invariant shape across samples (stack on a new leading dim);
    * **scalar** — non-tensor value (``int``/``float``/``bool``/``str``), kept as a list.

    A full scan is necessary because the leading-dim ambiguity (e.g. a 1-atom
    molecule colliding with a scalar of shape ``(1,)``) cannot be resolved
    from a single reference sample. Schema conflicts — inconsistent dtypes or
    a shape that neither tracks atoms/edges nor stays invariant — raise
    :class:`ValueError` so callers see the incompatibility at build time.
    """
    n = len(samples)
    if n == 0:
        return {
            "format_version": PackedCache.FORMAT_VERSION,
            "n_samples": 0,
            "schema": {},
            "atoms": {},
            "edges": {},
            "graphs": {},
            "scalars": {},
        }

    flats = [_flatten(s) for s in samples]
    keys = set(flats[0].keys())
    for i, f in enumerate(flats[1:], start=1):
        if set(f.keys()) != keys:
            missing = keys - set(f.keys())
            extra = set(f.keys()) - keys
            raise ValueError(
                f"Sample {i} has non-conforming keys "
                f"(missing={sorted(missing)}, extra={sorted(extra)}). "
                "All samples in a cache must share the same schema."
            )

    n_atoms = [_ref_len(f, "Z") for f in flats]
    n_edges = [_ref_len(f, "edge_index") for f in flats]
    has_atom_ref = any(x > 0 for x in n_atoms)
    has_edge_ref = any(x > 0 for x in n_edges)

    schema = _infer_schema_across(flats, n_atoms, n_edges, has_atom_ref, has_edge_ref)

    atom_keys = [k for k, spec in schema.items() if spec[0] == "atom"]
    edge_keys = [k for k, spec in schema.items() if spec[0] == "edge"]
    graph_keys = [k for k, spec in schema.items() if spec[0] == "graph"]
    scalar_keys = [k for k, spec in schema.items() if spec[0] == "scalar"]

    atom_ptr = [0]
    for na in n_atoms:
        atom_ptr.append(atom_ptr[-1] + na)
    edge_ptr = [0]
    for ne in n_edges:
        edge_ptr.append(edge_ptr[-1] + ne)

    payload: dict[str, Any] = {
        "format_version": PackedCache.FORMAT_VERSION,
        "n_samples": n,
        "schema": schema,
        "atoms": {k: torch.cat([f[k] for f in flats], dim=0) for k in atom_keys},
        "edges": {k: torch.cat([f[k] for f in flats], dim=0) for k in edge_keys},
        "graphs": {k: torch.stack([f[k] for f in flats], dim=0) for k in graph_keys},
        "scalars": {k: [f[k] for f in flats] for k in scalar_keys},
    }
    if atom_keys:
        payload["atom_ptr"] = torch.tensor(atom_ptr, dtype=torch.long)
    if edge_keys:
        payload["edge_ptr"] = torch.tensor(edge_ptr, dtype=torch.long)
    return payload


def _unpack_one(payload: Mapping[str, Any], idx: int) -> dict:
    n = payload["n_samples"]
    if idx < 0:
        idx += n
    if not 0 <= idx < n:
        raise IndexError(f"sample index {idx} out of range for n_samples={n}")

    flat: dict[str, Any] = {}

    atom_ptr = payload.get("atom_ptr")
    if atom_ptr is not None:
        a0, a1 = int(atom_ptr[idx]), int(atom_ptr[idx + 1])
        for k, t in payload["atoms"].items():
            flat[k] = t[a0:a1]

    edge_ptr = payload.get("edge_ptr")
    if edge_ptr is not None:
        e0, e1 = int(edge_ptr[idx]), int(edge_ptr[idx + 1])
        for k, t in payload["edges"].items():
            flat[k] = t[e0:e1]

    for k, t in payload["graphs"].items():
        flat[k] = t[idx]

    for k, vals in payload["scalars"].items():
        flat[k] = vals[idx]

    return _unflatten(flat)


class _LazySampleView:
    """List-like view that lazily reconstructs sample dicts on indexing.

    Exists mainly for test compatibility; downstream readers like
    :class:`molix.data.dataset.MmapDataset` call
    :meth:`PackedCache.unpack_sample` directly to avoid the
    method-dispatch indirection in tight loops.
    """

    __slots__ = ("_payload",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = payload

    def __len__(self) -> int:
        return int(self._payload["n_samples"])

    def __getitem__(self, idx: int) -> dict:
        if isinstance(idx, slice):
            return [_unpack_one(self._payload, i) for i in range(*idx.indices(len(self)))]
        return _unpack_one(self._payload, int(idx))

    def __iter__(self):
        for i in range(len(self)):
            yield _unpack_one(self._payload, i)


# -- schema inference + dict flattening -------------------------------------


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys. Tensors and scalars are leaves."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if key in _RESERVED_TOP_KEYS and not prefix:
            raise ValueError(f"Sample key {k!r} collides with a reserved packed-cache key.")
        if isinstance(v, Mapping):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_flatten` — rebuild nested dict from dotted keys."""
    out: dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return out


def _ref_len(flat: Mapping[str, Any], key: str) -> int:
    t = flat.get(key)
    if t is None:
        return 0
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{key!r} must be a Tensor to infer length, got {type(t).__name__}")
    return int(t.shape[0]) if t.ndim > 0 else 0


def _infer_schema_across(
    flats: list[dict[str, Any]],
    n_atoms: list[int],
    n_edges: list[int],
    has_atom_ref: bool,
    has_edge_ref: bool,
) -> dict[str, tuple]:
    """Classify each key by scanning its shape/type across every sample."""
    schema: dict[str, tuple] = {}
    keys = sorted(flats[0].keys())

    for k in keys:
        v0 = flats[0][k]

        if isinstance(v0, (int, float, bool, str)):
            typ = type(v0).__name__
            for i, f in enumerate(flats):
                if not isinstance(f[k], (int, float, bool, str)):
                    raise TypeError(
                        f"Key {k!r}: sample 0 is {typ} but sample {i} is {type(f[k]).__name__}"
                    )
            schema[k] = ("scalar", typ)
            continue

        if not isinstance(v0, torch.Tensor):
            raise TypeError(
                f"Unsupported leaf at {k!r}: {type(v0).__name__}. "
                "Only torch.Tensor and int/float/bool/str scalars are allowed."
            )

        shape0s: list[int | None] = []
        shape_rests: list[tuple[int, ...]] = []
        dtypes: set = set()
        for i, f in enumerate(flats):
            t = f[k]
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"Key {k!r}: sample 0 is Tensor but sample {i} is {type(t).__name__}"
                )
            dtypes.add(t.dtype)
            if t.ndim == 0:
                shape0s.append(None)
                shape_rests.append(())
            else:
                shape0s.append(int(t.shape[0]))
                shape_rests.append(tuple(t.shape[1:]))
        if len(dtypes) > 1:
            raise ValueError(f"Key {k!r}: dtype varies across samples: {dtypes}")
        dtype = next(iter(dtypes))

        tracks_atoms = has_atom_ref and all(s0 == na for s0, na in zip(shape0s, n_atoms))
        tracks_edges = has_edge_ref and all(s0 == ne for s0, ne in zip(shape0s, n_edges))
        # Prefer atom classification when both track (can happen if n_atoms == n_edges
        # holds across every sample, e.g. in degenerate cases).
        if tracks_atoms:
            rest_set = set(shape_rests)
            if len(rest_set) != 1:
                raise ValueError(
                    f"Key {k!r}: leading dim tracks n_atoms but trailing shape varies: {rest_set}"
                )
            schema[k] = ("atom", dtype, shape_rests[0])
        elif tracks_edges:
            rest_set = set(shape_rests)
            if len(rest_set) != 1:
                raise ValueError(
                    f"Key {k!r}: leading dim tracks n_edges but trailing shape varies: {rest_set}"
                )
            schema[k] = ("edge", dtype, shape_rests[0])
        else:
            full_shapes = {tuple(f[k].shape) for f in flats}
            if len(full_shapes) != 1:
                raise ValueError(
                    f"Key {k!r}: shape varies across samples ({full_shapes}) "
                    "but does not track n_atoms or n_edges — cannot pack."
                )
            schema[k] = ("graph", dtype, next(iter(full_shapes)))

    return schema


def _fsync_file(path: Path) -> None:
    """Best-effort fsync so the subsequent rename is durable."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
