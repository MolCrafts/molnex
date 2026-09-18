"""Repository dependency-boundary checks."""

from __future__ import annotations

import importlib.util

#: Verbs of data *acquisition*. MolHub downloads, parses, preprocesses and
#: caches; molnex only reads what it is handed. A public symbol here named for
#: one of these is the ETL layer coming back.
_ACQUISITION = ("download", "fetch", "ingest", "preprocess", "etl", "acquire", "scrape")


def test_molix_owns_no_data_acquisition() -> None:
    """Data acquisition belongs to MolHub — no ETL layer, under any name.

    A package named ``molix.data`` used to *be* that ETL layer and was deleted
    outright, so this guard used to refuse the name itself. The name is now the
    data **layer** — datasets, collate, padder, sampler, loader, merged from
    ``molix.store`` / ``molix.dataloader`` / ``molix.batch`` (spec
    ``data-layer-01-consolidation``) — which is exactly what molnex should own.
    So the guard asserts the invariant it always meant instead of the spelling:
    nothing in molnex acquires data.
    """
    assert importlib.util.find_spec("molix.datasets") is None, (
        "molix.datasets is the deleted ETL layer's other name; it must not resolve "
        "(a bare directory left behind as a namespace package also fails)"
    )

    import molix.data

    offenders = sorted(
        name for name in molix.data.__all__ if any(verb in name.lower() for verb in _ACQUISITION)
    )
    assert offenders == [], (
        f"data acquisition belongs to MolHub, but molix.data exports {offenders}"
    )
