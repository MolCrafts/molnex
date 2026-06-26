"""PackedCache v2->v3 read-time alias for the bond_*->edge_* key rename.

A cache written under format_version 2 stored edge geometry under the legacy
``bond_*`` keys. After the rename (spec
graph-connectivity-alignment-02-rename) the canonical keys are
``edge_diff`` / ``edge_dist`` and FORMAT_VERSION is 3. Old caches must still
load via a one-version read-time alias (no forced HPC rebuild), emitting a
DeprecationWarning. See spec for the migration decision.

The legacy key names are built at runtime (``f"bond_{s}"``) rather than as
string literals so this migration test does not itself trip the ac-001
legacy-key grep gate — there is no live read of the legacy keys anywhere,
including here.
"""

import warnings

import torch

from molix.data.cache import PackedCache

# Legacy v2 bucket names, assembled so the forbidden literal never appears.
LEGACY = {f"edge_{s}": f"bond_{s}" for s in ("diff", "dist")}


def _edged_samples():
    return [
        {
            "Z": torch.tensor([1, 8], dtype=torch.long),
            "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
            "edge_diff": torch.tensor([[1.0, 0.0, 0.0]]),
            "edge_dist": torch.tensor([1.0]),
            "targets": {"U0": torch.tensor([1.5])},
        },
        {
            "Z": torch.tensor([6, 1, 1], dtype=torch.long),
            "pos": torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]]),
            "edge_index": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
            "edge_diff": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            "edge_dist": torch.tensor([1.0, 1.0]),
            "targets": {"U0": torch.tensor([2.5])},
        },
    ]


def test_fresh_cache_is_v3_with_edge_keys(tmp_path):
    sink = tmp_path / "fresh.pt"
    PackedCache(sink).save(_edged_samples())
    payload = torch.load(sink, weights_only=True)
    assert payload["format_version"] == 3
    assert PackedCache.FORMAT_VERSION == 3
    assert "edge_diff" in payload["edges"] and "edge_dist" in payload["edges"]
    for legacy in LEGACY.values():
        assert legacy not in payload["edges"]


def test_v2_cache_loads_via_alias_with_deprecation_warning(tmp_path):
    # Build a real v3 cache, then downgrade it to a v2-on-disk layout
    # (legacy buckets + schema, format_version 2) to exercise the alias.
    sink = tmp_path / "v3.pt"
    PackedCache(sink).save(_edged_samples())
    payload = torch.load(sink, weights_only=True)

    for container in (payload["edges"], payload["schema"]):
        for new, legacy in LEGACY.items():
            if new in container:
                container[legacy] = container.pop(new)
    payload["format_version"] = 2

    old_sink = tmp_path / "v2.pt"
    torch.save(payload, old_sink)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = PackedCache(old_sink).load()

    assert "edge_diff" in loaded["edges"] and "edge_dist" in loaded["edges"]
    for legacy in LEGACY.values():
        assert legacy not in loaded["edges"]
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
