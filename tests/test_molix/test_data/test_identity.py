"""Tests for :func:`molix.data.compute_cache_identity`."""

from __future__ import annotations

from molix.data import (
    InMemorySource,
    compute_cache_identity,
    pipeline,
)


def _pipe(name: str = "p"):
    return pipeline(name).build()


def _src(name: str = "src", n: int = 3):
    samples = [{"Z": i} for i in range(n)]
    return InMemorySource(samples, name=name)


class TestCacheIdentity:
    def test_stable_across_calls(self):
        p, s = _pipe(), _src()
        assert compute_cache_identity(p, s) == compute_cache_identity(p, s)

    def test_pipeline_name_changes_identity(self):
        p1, p2 = _pipe("a"), _pipe("b")
        s = _src()
        assert compute_cache_identity(p1, s) != compute_cache_identity(p2, s)

    def test_source_changes_identity(self):
        p = _pipe()
        assert compute_cache_identity(p, _src("a")) != compute_cache_identity(
            p, _src("b")
        )

    def test_fit_source_changes_identity(self):
        p = _pipe()
        s = _src("main")
        fit_a = _src("fit_a")
        assert compute_cache_identity(p, s) != compute_cache_identity(
            p, s, fit_source=fit_a
        )

    def test_fit_source_defaults_to_source(self):
        p = _pipe()
        s = _src()
        assert compute_cache_identity(p, s) == compute_cache_identity(
            p, s, fit_source=s
        )

    def test_extra_changes_identity(self):
        p, s = _pipe(), _src()
        assert compute_cache_identity(p, s) != compute_cache_identity(
            p, s, extra={"impl": "v2"}
        )

    def test_extra_key_order_does_not_matter(self):
        p, s = _pipe(), _src()
        a = compute_cache_identity(p, s, extra={"a": "1", "b": "2"})
        b = compute_cache_identity(p, s, extra={"b": "2", "a": "1"})
        assert a == b

    def test_is_short_hex(self):
        ident = compute_cache_identity(_pipe(), _src())
        assert len(ident) == 12
        int(ident, 16)  # valid hex
