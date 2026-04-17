"""Tests for :func:`molix.data.cache.cache_key`."""

from __future__ import annotations

from molix.data.cache import cache_key


class TestCacheKey:
    def test_stable_across_calls(self):
        a = cache_key(pipeline_id="p", source_id="s")
        b = cache_key(pipeline_id="p", source_id="s")
        assert a == b

    def test_pipeline_id_changes_key(self):
        a = cache_key(pipeline_id="p1", source_id="s")
        b = cache_key(pipeline_id="p2", source_id="s")
        assert a != b

    def test_source_id_changes_key(self):
        a = cache_key(pipeline_id="p", source_id="s1")
        b = cache_key(pipeline_id="p", source_id="s2")
        assert a != b

    def test_fit_source_id_changes_key(self):
        a = cache_key(pipeline_id="p", source_id="s")
        b = cache_key(pipeline_id="p", source_id="s", fit_source_id="fit")
        assert a != b

    def test_fit_source_defaults_to_source(self):
        a = cache_key(pipeline_id="p", source_id="s")
        b = cache_key(pipeline_id="p", source_id="s", fit_source_id="s")
        assert a == b

    def test_extra_changes_key(self):
        a = cache_key(pipeline_id="p", source_id="s")
        b = cache_key(pipeline_id="p", source_id="s", extra={"impl": "v2"})
        assert a != b

    def test_extra_key_order_is_stable(self):
        a = cache_key(pipeline_id="p", source_id="s",
                      extra={"a": "1", "b": "2"})
        b = cache_key(pipeline_id="p", source_id="s",
                      extra={"b": "2", "a": "1"})
        assert a == b

    def test_return_is_short_hex(self):
        k = cache_key(pipeline_id="p", source_id="s")
        assert len(k) == 12
        int(k, 16)
