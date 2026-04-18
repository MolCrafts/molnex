"""Tests for ``molix.logging`` channel-routing primitives + ``configure_run``."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from mollog import Level, TextFormatter

from molix import logging as mlogging
from molix.logging import (
    ChannelFilter,
    CSVMetricsFormatter,
    EventFormatter,
    EVENTS_LOGGER_NAME,
    KindFilter,
    METRICS_LOGGER_NAME,
    PrettyTextFormatter,
    configure_run,
    events_logger,
    get_table_width,
    has_effective_handlers,
    metrics_logger,
    set_table_width,
)


# ---------------------------------------------------------------------------
# Fixture: ensure each test starts with a clean molix logger tree.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_molix_logger():
    mlogging.shutdown()
    # Shared table-width is module-level state; reset to the documented
    # default so test ordering never affects assertions.
    set_table_width(80)
    yield
    mlogging.shutdown()
    set_table_width(80)


# ---------------------------------------------------------------------------
# Table width sharing
# ---------------------------------------------------------------------------


class TestTableWidth:
    def test_defaults_to_80(self):
        # Fresh state → default
        assert get_table_width() == 80

    def test_set_and_get(self):
        set_table_width(125)
        assert get_table_width() == 125

    def test_ignores_non_positive(self):
        set_table_width(125)
        set_table_width(0)  # no-op
        set_table_width(-5)  # no-op
        assert get_table_width() == 125


# ---------------------------------------------------------------------------
# has_effective_handlers
# ---------------------------------------------------------------------------


class TestHasEffectiveHandlers:
    def test_false_when_molix_unconfigured(self):
        # Default mollog root has a fallback StreamHandler but molix itself
        # starts clean — we must *not* treat the mollog default as molix
        # configuration.
        assert has_effective_handlers() is False

    def test_true_after_basicConfig(self):
        mlogging.basicConfig(level=Level.INFO)
        assert has_effective_handlers() is True

    def test_true_after_configure_run(self, tmp_path):
        configure_run(tmp_path / "run")
        assert has_effective_handlers() is True


# ---------------------------------------------------------------------------
# PrettyTextFormatter: kind-dispatched rendering
# ---------------------------------------------------------------------------


def _make_record(
    extra: dict,
    message: str = "",
    logger_name: str = "molix.metrics",
) -> object:
    from mollog.record import LogRecord
    from datetime import datetime

    return LogRecord(
        level=Level.INFO,
        logger_name=logger_name,
        message=message,
        timestamp=datetime(2026, 4, 18, 12, 0, 0),
        extra=extra,
    )


class TestPrettyTextFormatter:
    def test_header_renders_aligned_columns(self):
        fmt = PrettyTextFormatter(col_width=10)
        rec = _make_record({"kind": "header", "columns": ["step", "loss"]})
        out = fmt.format(rec)
        assert out == "      step       loss"

    def test_row_renders_aligned_values(self):
        fmt = PrettyTextFormatter(col_width=10, row_fmt="{:>10.4g}")
        rec = _make_record({
            "kind": "row",
            "columns": ["step", "loss"],
            "values": {"step": 42, "loss": 0.125},
        })
        out = fmt.format(rec)
        assert out.split() == ["42", "0.125"]

    def test_row_missing_value_becomes_nan(self):
        fmt = PrettyTextFormatter(col_width=10, row_fmt="{:>10.4g}")
        rec = _make_record({
            "kind": "row",
            "columns": ["step", "missing"],
            "values": {"step": 1},
        })
        assert "nan" in fmt.format(rec)

    def test_epoch_sep_is_full_width_dashes(self):
        fmt = PrettyTextFormatter()
        rec = _make_record({"kind": "epoch_sep", "table_width": 20})
        assert fmt.format(rec) == "─" * 20

    def test_announce_renders_inline_separator(self):
        fmt = PrettyTextFormatter()
        rec = _make_record(
            {"kind": "announce", "table_width": 40},
            message="hello",
        )
        out = fmt.format(rec)
        assert out.startswith("─── hello ")
        assert out.endswith("─")
        assert len(out) == 40

    def test_unknown_kind_falls_back_to_text_formatter(self):
        fmt = PrettyTextFormatter()
        rec = _make_record({}, message="plain info")
        out = fmt.format(rec)
        # Fallback is TextFormatter — should contain the message verbatim.
        assert "plain info" in out


# ---------------------------------------------------------------------------
# CSVMetricsFormatter: stateful header then rows
# ---------------------------------------------------------------------------


class TestCSVMetricsFormatter:
    def test_header_produces_csv_header(self):
        f = CSVMetricsFormatter()
        rec = _make_record({"kind": "header", "columns": ["step", "train/loss"]})
        assert f.format(rec) == "step,train/loss"

    def test_row_after_header_produces_csv_data(self):
        f = CSVMetricsFormatter()
        f.format(_make_record({"kind": "header", "columns": ["step", "loss"]}))
        data = f.format(_make_record({
            "kind": "row",
            "columns": ["step", "loss"],
            "values": {"step": 10, "loss": 0.5},
        }))
        assert data == "10,0.5"

    def test_other_kinds_drop_to_empty_string(self):
        f = CSVMetricsFormatter()
        assert f.format(_make_record({"kind": "epoch_sep"})) == ""
        assert f.format(_make_record({"kind": "announce"}, message="x")) == ""

    def test_csv_cell_quotes_commas_and_newlines(self):
        f = CSVMetricsFormatter()
        f.format(_make_record({"kind": "header", "columns": ["a", "note"]}))
        rec = _make_record({
            "kind": "row",
            "columns": ["a", "note"],
            "values": {"a": 1, "note": "a,b\nc"},
        })
        out = f.format(rec)
        assert out == '1,"a,b\nc"'

    def test_nan_becomes_empty_cell(self):
        f = CSVMetricsFormatter()
        f.format(_make_record({"kind": "header", "columns": ["x"]}))
        rec = _make_record({
            "kind": "row",
            "columns": ["x"],
            "values": {"x": float("nan")},
        })
        assert f.format(rec) == ""

    def test_formatter_always_writes_header(self):
        """Header dedup is a filter-layer concern (see configure_run)."""
        f = CSVMetricsFormatter()
        rec = _make_record({"kind": "header", "columns": ["step", "loss"]})
        assert f.format(rec) == "step,loss"
        assert f.format(rec) == "step,loss"


# ---------------------------------------------------------------------------
# EventFormatter
# ---------------------------------------------------------------------------


class TestEventFormatter:
    def test_renders_timestamp_plus_message(self):
        f = EventFormatter()
        rec = _make_record({}, message="ckpt saved")
        out = f.format(rec)
        assert out.endswith(" ckpt saved")
        assert "2026-04-18" in out


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestChannelFilter:
    def test_matches_exact_and_children(self):
        f = ChannelFilter("molix.metrics")
        assert f.filter(_make_record({}, "x", logger_name="molix.metrics")) is True
        assert f.filter(_make_record({}, "x", logger_name="molix.metrics.csv")) is True
        assert f.filter(_make_record({}, "x", logger_name="molix.events")) is False


class TestKindFilter:
    def test_passes_only_listed_kinds(self):
        f = KindFilter("header", "row")
        assert f.filter(_make_record({"kind": "header"})) is True
        assert f.filter(_make_record({"kind": "row"})) is True
        assert f.filter(_make_record({"kind": "epoch_sep"})) is False
        assert f.filter(_make_record({})) is False


# ---------------------------------------------------------------------------
# configure_run — end-to-end routing
# ---------------------------------------------------------------------------


class TestConfigureRun:
    def test_creates_all_four_files(self, tmp_path):
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        # Files exist after creation (they're opened for write).
        for name in ("train.log", "metrics.csv", "events.log", "warnings.log"):
            assert (run_dir / name).exists(), f"{name} not created"

    def test_metrics_channel_lands_in_metrics_csv_only(self, tmp_path):
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        m = metrics_logger()
        m.info("h", kind="header", columns=["step", "loss"])
        m.info("r", kind="row", columns=["step", "loss"],
               values={"step": 1, "loss": 0.5})
        mlogging.shutdown()

        csv = (run_dir / "metrics.csv").read_text()
        assert csv.splitlines() == ["step,loss", "1,0.5"]
        # events.log must stay empty — metrics shouldn't leak there.
        assert (run_dir / "events.log").read_text() == ""

    def test_metrics_csv_dedupes_repeated_headers(self, tmp_path):
        """Periodic header reprints don't re-emit the CSV header line."""
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        m = metrics_logger()
        m.info("h", kind="header", columns=["step", "loss"])
        m.info("r", kind="row", columns=["step", "loss"],
               values={"step": 1, "loss": 0.5})
        # Periodic reprint — same columns.
        m.info("h", kind="header", columns=["step", "loss"])
        m.info("r", kind="row", columns=["step", "loss"],
               values={"step": 2, "loss": 0.3})
        mlogging.shutdown()

        lines = (run_dir / "metrics.csv").read_text().splitlines()
        assert lines == ["step,loss", "1,0.5", "2,0.3"]

    def test_events_channel_lands_in_events_log_only(self, tmp_path):
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        events_logger().info("ckpt: foo", kind="announce", table_width=40)
        mlogging.shutdown()

        events = (run_dir / "events.log").read_text().strip()
        assert events.endswith(" ckpt: foo")
        # metrics.csv shouldn't collect events.
        assert (run_dir / "metrics.csv").read_text() == ""

    def test_warnings_land_in_warnings_log(self, tmp_path):
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        mlogging.getLogger("x").warning("careful")
        mlogging.shutdown()

        warns = (run_dir / "warnings.log").read_text()
        assert "careful" in warns
        assert "WARNING" in warns

    def test_info_lands_in_train_log(self, tmp_path):
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        mlogging.getLogger("x").info("hello")
        mlogging.shutdown()

        train = (run_dir / "train.log").read_text()
        assert "hello" in train

    def test_warning_also_lands_in_train_log(self, tmp_path):
        """train.log is the full audit — warnings hit it too, not just warnings.log."""
        run_dir = tmp_path / "run"
        configure_run(run_dir)
        mlogging.getLogger("x").warning("hey")
        mlogging.shutdown()

        assert "hey" in (run_dir / "train.log").read_text()
        assert "hey" in (run_dir / "warnings.log").read_text()

    def test_stdout_receives_metrics_and_events_regardless_of_level(self, tmp_path):
        """metrics/events channels bypass the console level gate."""
        run_dir = tmp_path / "run"
        buf = io.StringIO()
        configure_run(run_dir, console_level="CRITICAL", stream=buf)

        metrics_logger().info("h", kind="header", columns=["step"])
        events_logger().info("evt", kind="announce", table_width=20)
        mlogging.getLogger("x").info("dropped")  # below CRITICAL → filtered
        mlogging.shutdown()

        out = buf.getvalue()
        assert "step" in out
        assert "─── evt " in out
        assert "dropped" not in out
