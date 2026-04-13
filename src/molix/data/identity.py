"""Cache identity helper for workflow-side cache directory naming.

This module provides a *utility* that workflows (user scripts) may choose to
call when they need a stable hash that identifies a (pipeline, source,
fit_source) combination. It is **not** called from inside
:class:`~molix.data.PipelineSpec` — naming and placement of cache directories
is a scheduling concern that belongs to the workflow layer, not to the
transform layer.

Typical use::

    from molix.data import compute_cache_identity

    from molix.data import compute_cache_identity, is_cache_ready

    ident = compute_cache_identity(pipe, source)
    sink = workspace / "qm9" / "cache" / f"{pipe.name}__{ident}"
    if not is_cache_ready(sink):
        pipe.materialize(source, sink=sink)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molix.data.pipeline import PipelineSpec
    from molix.data.source import DataSource

# Bump when the cache directory schema changes in an identity-affecting way
# (e.g. new required meta fields, different serialization layout). Kept in
# sync with ``CACHE_SCHEMA_VERSION`` in ``molix.data.dataset``.
_IDENTITY_SCHEMA_VERSION = 1

_HASH_LEN = 12


def compute_cache_identity(
    pipeline: "PipelineSpec",
    source: "DataSource",
    *,
    fit_source: "DataSource | None" = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    """Return a stable short hash identifying a (pipeline, source, fit_source) asset.

    The formula is intentionally centralised here so future additions
    (``molix_version``, ``implementation_version``, env dimensions) change one
    function rather than every call site. Callers must treat the return value
    as opaque.

    Args:
        pipeline: Compiled pipeline spec.
        source: Raw data source to be materialised.
        fit_source: Source used for :meth:`DatasetTask.fit`. Defaults to
            *source*; pass a distinct source (e.g. train-only split) when
            fitting statistics should not see the full dataset.
        extra: Additional string key/value pairs to fold into the hash.
            Keys are sorted for stability. Use for workflow-local
            dimensions that should invalidate cache (e.g. ``{"impl": "v2"}``).

    Returns:
        A short hex string suitable for use as part of a directory name.
    """
    fs_id = fit_source.source_id if fit_source is not None else source.source_id
    parts = [
        f"schema={_IDENTITY_SCHEMA_VERSION}",
        f"pipeline_id={pipeline.pipeline_id}",
        f"source_id={source.source_id}",
        f"fit_source_id={fs_id}",
    ]
    if extra:
        for k in sorted(extra):
            parts.append(f"{k}={extra[k]}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:_HASH_LEN]
