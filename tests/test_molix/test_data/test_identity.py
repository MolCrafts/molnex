"""Tests for :meth:`molix.data.PipelineSpec.cache_identity`."""

from __future__ import annotations

from molix.data import InMemorySource, Pipeline


def _pipe(name: str = "p"):
    return Pipeline(name).build()


def _src(name: str = "src", n: int = 3):
    samples = [{"Z": i} for i in range(n)]
    return InMemorySource(samples, name=name)


class TestCacheIdentity:
    def test_stable_across_calls(self):
        p, s = _pipe(), _src()
        assert p.cache_identity(s) == p.cache_identity(s)

    def test_pipeline_name_changes_identity(self):
        p1, p2 = _pipe("a"), _pipe("b")
        s = _src()
        assert p1.cache_identity(s) != p2.cache_identity(s)

    def test_source_changes_identity(self):
        p = _pipe()
        assert p.cache_identity(_src("a")) != p.cache_identity(_src("b"))

    def test_fit_source_changes_identity(self):
        p = _pipe()
        s = _src("main")
        fit_a = _src("fit_a")
        assert p.cache_identity(s) != p.cache_identity(s, fit_source=fit_a)

    def test_fit_source_defaults_to_source(self):
        p = _pipe()
        s = _src()
        assert p.cache_identity(s) == p.cache_identity(s, fit_source=s)

    def test_extra_changes_identity(self):
        p, s = _pipe(), _src()
        assert p.cache_identity(s) != p.cache_identity(s, extra={"impl": "v2"})

    def test_extra_key_order_does_not_matter(self):
        p, s = _pipe(), _src()
        a = p.cache_identity(s, extra={"a": "1", "b": "2"})
        b = p.cache_identity(s, extra={"b": "2", "a": "1"})
        assert a == b

    def test_is_short_hex(self):
        ident = _pipe().cache_identity(_src())
        assert len(ident) == 12
        int(ident, 16)  # valid hex
