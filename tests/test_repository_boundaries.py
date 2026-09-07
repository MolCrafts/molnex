"""Repository dependency-boundary checks."""

from __future__ import annotations

import importlib.util


def test_molix_data_packages_are_gone() -> None:
    """The ETL layer was deleted outright — no package, no shim, no alias."""
    resolvable = [
        name
        for name in ("molix.data", "molix.datasets")
        if importlib.util.find_spec(name) is not None
    ]

    assert resolvable == [], (
        "data acquisition belongs to MolHub; these packages must not resolve "
        "(stage: experimental — deleted without a deprecation path, so a bare "
        f"directory left behind as a namespace package also fails): {resolvable}"
    )
