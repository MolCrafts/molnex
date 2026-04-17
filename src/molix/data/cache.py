"""Short-lived pipeline output cache.

**Scope**: one training run. The cache exists so that expensive preprocessing
(neighbor lists, atomic dress, etc.) is computed *once* at the start of a run
and reused for every training step thereafter. It is explicitly **not** a
persistence format:

* no schema version, no ``meta.json``, no ``_READY`` sentinel;
* no validation beyond "file exists and opens";
* no cross-version / cross-run compatibility guarantees;
* the workflow owns placement and invalidation — typical placement is under
  ``run_ctx.run_dir / "cache"`` so the file is naturally scoped to the run.

If long-term persistence matters, use :mod:`molix.datasets` (curated
datasets) instead.

Single-file layout — one ``<sink>.pt`` per cache, written via ``torch.save``
and read via ``torch.load(mmap=True)``. A failed write never leaves a partial
file behind (atomic ``os.rename``). Callers own placement, naming, and
invalidation policy.

Typical workflow::

    from molix.data.cache import cache, cache_key, is_ready
    from molix.data.ddp import rank, wait_for_ready

    key = cache_key(
        pipeline_id=pipe.pipeline_id,
        source_id=source.source_id,
        fit_source_id=train.source_id,
        extra={"n_train": str(n_train), "seed": str(seed)},
    )
    sink = run_dir / "cache" / f"{pipe.name}-{key}.pt"
    sink.parent.mkdir(parents=True, exist_ok=True)

    if rank() == 0 and not is_ready(sink):
        cache(pipe, source, sink=sink, fit_source=train)
    else:
        wait_for_ready(sink)
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from molix.data.execute import collect_task_states, run

if TYPE_CHECKING:
    from molix.data.pipeline import PipelineSpec
    from molix.data.source import DataSource


__all__ = ["cache_key", "is_ready", "save", "load", "cache"]


_KEY_HEX_LEN = 12


def cache_key(
    *,
    pipeline_id: str,
    source_id: str,
    fit_source_id: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    """Return a 12-hex SHA256 for the ``(pipeline, source, fit_source, extra)`` tuple.

    The *workflow* composes these strings however it wants — in particular,
    ``extra`` is a free-form dict for pinning split sizes, seeds, dtype, etc.
    Changing any string invalidates the cache.
    """
    fs_id = fit_source_id if fit_source_id is not None else source_id
    parts = [
        f"pipeline_id={pipeline_id}",
        f"source_id={source_id}",
        f"fit_source_id={fs_id}",
    ]
    if extra:
        for k in sorted(extra):
            parts.append(f"{k}={extra[k]}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:_KEY_HEX_LEN]


def is_ready(sink: str | Path) -> bool:
    """Return ``True`` if *sink* is a readable cache file.

    Readable = exists, is a regular file, and has non-zero size. Actual
    unpickle safety is deferred to load time — a torture of half-written
    files would fail there, which is fine: the workflow would treat it as
    not-ready and rebuild.
    """
    p = Path(sink)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def save(
    sink: str | Path,
    samples: list[dict],
    *,
    task_states: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """Serialize *samples* (+ optional *task_states*) atomically to *sink*.

    Uses ``torch.save``; readable later with ``torch.load(mmap=True)``.
    Writes ``<sink>.partial.<uuid>``, fsyncs, then ``os.rename`` onto *sink*
    — single-file rename is POSIX-atomic, so observers never see a partial
    file.

    Args:
        sink: Target file path (``.pt`` extension recommended).
        samples: Processed sample dicts. Leaves must be ``torch.Tensor`` or
            JSON-safe scalars (``int``/``float``/``str``/``bool``) so that
            downstream :func:`load` can use ``weights_only=True``.
        task_states: Optional fitted state for :class:`DatasetTask`
            instances, typically produced by
            :func:`~molix.data.execute.collect_task_states`.
        overwrite: If *sink* already exists, replace it. Otherwise keep the
            existing file (no-op).
    """
    sink = Path(sink)
    if sink.exists() and not overwrite:
        return

    sink.parent.mkdir(parents=True, exist_ok=True)
    tmp = sink.parent / f"{sink.name}.partial.{uuid.uuid4().hex[:8]}"

    payload: dict[str, Any] = {"samples": list(samples)}
    if task_states:
        payload["task_states"] = {k: dict(v) for k, v in task_states.items()}

    try:
        torch.save(payload, tmp)
        _fsync_file(tmp)
        os.replace(tmp, sink)        # atomic on POSIX (incl. same-mount NFS)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load(sink: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load a cache file.

    Args:
        sink: Path previously written by :func:`save`.
        mmap: Memory-map the tensor storages (default). Set to ``False`` if
            you want a full in-memory copy.

    Returns:
        ``{"samples": [...], "task_states": {...}}``. ``task_states`` is
        absent or empty when the pipeline had no :class:`DatasetTask`.
    """
    return torch.load(sink, mmap=mmap, weights_only=True)


def cache(
    pipeline: "PipelineSpec",
    source: "DataSource",
    *,
    sink: str | Path,
    fit_source: "DataSource | None" = None,
    overwrite: bool = False,
) -> None:
    """Run *pipeline* against *source* and cache the result atomically.

    Equivalent to::

        samples = list(run(pipeline, source, fit_source=fit_source))
        save(sink, samples, task_states=collect_task_states(pipeline))

    Leaves DDP, readiness checks, and path placement to the caller; see the
    module docstring for the recommended workflow skeleton.
    """
    sink = Path(sink)
    if sink.exists() and not overwrite:
        return
    samples = list(run(pipeline, source, fit_source=fit_source))
    save(sink, samples, task_states=collect_task_states(pipeline), overwrite=overwrite)


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
