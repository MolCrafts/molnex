"""Tests for :mod:`molix.data.ddp`."""

from __future__ import annotations

import pytest

from molix.data.ddp import rank, wait_for_ready


class TestRank:
    def test_missing_rank_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        assert rank() == 0

    def test_rank_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        assert rank() == 3

    def test_malformed_rank_env_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv("RANK", "not-a-number")
        assert rank() == 0


class TestWaitForReady:
    def test_returns_immediately_when_ready(self, tmp_path):
        sink = tmp_path / "ready.pt"
        sink.write_bytes(b"data")
        # No raise → test passes.
        wait_for_ready(sink, timeout=1.0, poll_interval=0.01)

    def test_raises_on_timeout(self, tmp_path):
        with pytest.raises(TimeoutError, match="Timed out"):
            wait_for_ready(
                tmp_path / "never.pt",
                timeout=0.3,
                poll_interval=0.05,
            )
