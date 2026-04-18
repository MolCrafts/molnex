"""Stdlib-compatible logging facade for molix with per-channel routing.

Drop-in replacement for ``import logging`` inside the molix stack.  The
public API mirrors the standard library's ``logging`` module — level
constants (``DEBUG``, ``INFO``, ``WARNING`` …), ``getLogger(name)``,
``basicConfig(...)`` — but records flow through mollog.  Everything is
scoped to the ``"molix"`` logger with ``propagate=False``, so
configuring it never mutates the mollog root logger that other packages
in the process may rely on.

Beyond the stdlib surface, this module defines an opinionated **channel
model** so training output is split across multiple destinations:

==================  ====================  =============================================
Channel             Logger name           Typical sink
==================  ====================  =============================================
``metrics``         ``molix.metrics``     ``metrics.csv`` (CSV) + stdout (aligned row)
``events``          ``molix.events``      ``events.log``  + stdout (``─── msg ───``)
``info`` (generic)  ``molix.<module>``    ``train.log``
``warn``            any logger            ``warnings.log`` + stdout
==================  ====================  =============================================

Records carry an ``extra["kind"]`` tag that formatters dispatch on:

* ``header`` / ``row`` — structured table emission by :class:`molix.core.hooks.Log`
* ``epoch_sep`` — full-width ``────`` separator at epoch boundaries
* ``announce`` — thin ``─── message ───`` separator for intermittent events

The dispatch machinery lives in :class:`PrettyTextFormatter` (for the
console view) and :class:`CSVMetricsFormatter` (for the structured sink).
See :func:`configure_run` for the one-call wiring of all sinks.

Typical use::

    from molix import logging

    logging.configure_run(run_dir="runs/exp1", console_level="WARNING")
    log = logging.getLogger(__name__)
    log.info("starting run", steps=200)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Any

from mollog import FileHandler, Filter, Level, Logger, StreamHandler, TextFormatter
from mollog import get_logger as _mollog_get_logger
from mollog.formatter import Formatter
from mollog.record import LogRecord

__all__ = [
    "MOLIX_LOGGER_NAME",
    # Level constants (stdlib parity)
    "TRACE",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    # Stdlib-style API
    "getLogger",
    "basicConfig",
    "shutdown",
    # Snake-case aliases (used by code that prefers mollog spelling)
    "get_logger",
    "configure",
    # Channel model — used by Log / CheckpointHook and any custom hook
    # that wants its events to participate in the split routing.
    "METRICS_LOGGER_NAME",
    "EVENTS_LOGGER_NAME",
    "metrics_logger",
    "events_logger",
    "has_effective_handlers",
    "set_table_width",
    "get_table_width",
    # Formatters / filters
    "PrettyTextFormatter",
    "CSVMetricsFormatter",
    "EventFormatter",
    "ChannelFilter",
    "KindFilter",
    # One-call router setup
    "configure_run",
]

MOLIX_LOGGER_NAME = "molix"

# ── Level constants (stdlib parity) ─────────────────────────────────────────
# Exported so callers can write `logging.INFO` instead of importing Level
# directly — matches the `logging.INFO` convention.
TRACE = Level.TRACE
DEBUG = Level.DEBUG
INFO = Level.INFO
WARNING = Level.WARNING
ERROR = Level.ERROR
CRITICAL = Level.CRITICAL


def _qualify(name: str) -> str:
    if not name or name == MOLIX_LOGGER_NAME:
        return MOLIX_LOGGER_NAME
    if name.startswith(MOLIX_LOGGER_NAME + "."):
        return name
    return f"{MOLIX_LOGGER_NAME}.{name}"


def getLogger(name: str = "") -> Logger:  # noqa: N802 — stdlib naming
    """Return a logger under the ``molix`` namespace.

    Mirrors :func:`logging.getLogger`.  ``getLogger()`` returns the
    ``molix`` root; ``getLogger("foo.bar")`` returns ``molix.foo.bar``
    (the prefix is added automatically if missing).  Child loggers
    propagate to ``molix`` by default and inherit the handlers set by
    :func:`basicConfig`.
    """
    return _mollog_get_logger(_qualify(name))


def basicConfig(  # noqa: N802 — stdlib naming
    *,
    level: Level | str | int = Level.INFO,
    filename: str | Path | None = None,
    filemode: str = "a",
    stream: IO[str] | None = None,
    formatter: Formatter | None = None,
    stream_formatter: Formatter | None = None,
    file_formatter: Formatter | None = None,
    file_level: Level | str | int | None = None,
    encoding: str = "utf-8",
    force: bool = True,
) -> Logger:
    """Configure the ``molix`` logger without touching the mollog root.

    Mirrors :func:`logging.basicConfig` with the dual-destination
    convenience we need for training runs: if *filename* is given, both
    a :class:`StreamHandler` (on *stream* or ``sys.stderr``) **and** a
    :class:`FileHandler` are attached — records land in both.
    *file_level* lets the file accept a looser threshold (e.g. DEBUG)
    than the console.

    Sets ``propagate=False`` on the ``molix`` logger so records never
    reach other mollog consumers' root-logger handlers.

    Formatters
    ----------
    By default the stream handler is wired to :class:`PrettyTextFormatter`
    so :class:`molix.core.hooks.Log`'s ``metrics`` / ``events`` records
    render as aligned tables and ``─── message ───`` separators out of
    the box; the file handler uses the structured :class:`TextFormatter`.
    Pass *stream_formatter* / *file_formatter* (or *formatter* as a
    single-formatter override that applies to both) to change either.
    Plain :class:`TextFormatter` everywhere is recovered by passing
    ``formatter=TextFormatter()``.

    Parameters
    ----------
    force:
        If ``True`` (default), existing handlers on the ``molix`` logger
        are closed and replaced.  Matches the stdlib ``force=True``
        semantics but flipped to default-on, because training scripts
        call ``basicConfig`` once per run and expect a clean slate.
    """

    lvl = Level.coerce(level)
    file_lvl = Level.coerce(file_level) if file_level is not None else lvl

    if formatter is not None:
        resolved_stream_fmt = formatter
        resolved_file_fmt = formatter
    else:
        resolved_stream_fmt = stream_formatter or PrettyTextFormatter()
        resolved_file_fmt = file_formatter or TextFormatter()

    logger = _mollog_get_logger(MOLIX_LOGGER_NAME)
    logger.level = lvl
    logger.propagate = False

    if force:
        logger.clear_handlers(close=True)

    stream_handler = StreamHandler(stream=stream or sys.stderr, level=lvl)
    stream_handler.set_formatter(resolved_stream_fmt)
    logger.add_handler(stream_handler)

    if filename is not None:
        file_handler = FileHandler(
            Path(filename),
            mode=filemode,
            encoding=encoding,
            level=file_lvl,
        )
        file_handler.set_formatter(resolved_file_fmt)
        logger.add_handler(file_handler)

    return logger


def shutdown() -> None:
    """Close all handlers attached to the ``molix`` logger."""
    logger = _mollog_get_logger(MOLIX_LOGGER_NAME)
    logger.clear_handlers(close=True)


# ── Snake-case aliases ──────────────────────────────────────────────────────
# Kept so mollog-style callers (`logging.configure(...)`) keep working.
configure = basicConfig
get_logger = getLogger


# ---------------------------------------------------------------------------
# Channel model — loggers, table-width share, formatters, filters, configure_run
# ---------------------------------------------------------------------------

METRICS_LOGGER_NAME = f"{MOLIX_LOGGER_NAME}.metrics"
EVENTS_LOGGER_NAME = f"{MOLIX_LOGGER_NAME}.events"


def metrics_logger() -> Logger:
    """Dedicated logger for the training-table channel (``molix.metrics``)."""
    return _mollog_get_logger(METRICS_LOGGER_NAME)


def events_logger() -> Logger:
    """Dedicated logger for the inline-event channel (``molix.events``)."""
    return _mollog_get_logger(EVENTS_LOGGER_NAME)


def has_effective_handlers(logger: Logger | None = None) -> bool:
    """True if the molix logger tree has any handler attached directly.

    Walks ``logger`` (defaulting to the ``molix`` root) up to — but
    **not** including — the mollog root, whose default stderr
    ``StreamHandler`` would otherwise always return True and prevent
    the :class:`~molix.core.hooks.Log` hook from printing under a
    zero-config / unit-test harness. A False return means "no
    molix-level handler is attached", so callers should fall back to
    :func:`print` for visible output.
    """
    cur: Logger | None = logger or _mollog_get_logger(MOLIX_LOGGER_NAME)
    while cur is not None:
        if getattr(cur, "handlers", None):
            return True
        name = getattr(cur, "name", "")
        if name == MOLIX_LOGGER_NAME:
            return False  # stopped at molix root with no handlers
        cur = getattr(cur, "parent", None)
    return False


# Module-level shared rendering width, populated by the Log hook at
# ``on_train_start`` so that off-hook emitters (e.g. ``CheckpointHook``)
# can still lay out their ``─── msg ───`` separators flush with the
# training table's column stride. Falls back to 80 columns if never set.
_table_width: int = 80


def set_table_width(width: int) -> None:
    """Publish the training-table width for pretty-formatter alignment."""
    global _table_width
    if width > 0:
        _table_width = int(width)


def get_table_width() -> int:
    """Current table width advertised by the Log hook (80 if unset)."""
    return _table_width


def _csv_cell(value: Any) -> str:
    """Render one CSV cell — quote only when needed, elide NaN as empty."""
    if value is None:
        return ""
    if isinstance(value, float):
        # NaN / inf → empty cells so pandas / numpy readers don't trip.
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        return repr(value)
    if isinstance(value, (int, bool)):
        return str(int(value) if isinstance(value, bool) else value)
    s = str(value)
    if any(c in s for c in (",", '"', "\n")):
        return '"' + s.replace('"', '""') + '"'
    return s


class PrettyTextFormatter(Formatter):
    """Channel-aware console formatter.

    Dispatch table based on ``record.extra["kind"]``:

    ============  ===========================================================
    ``kind``      Rendering
    ============  ===========================================================
    ``header``    Aligned column header — uses ``extra["columns"]``.
    ``row``       Aligned data row — uses ``extra["columns"]`` + ``values``.
    ``epoch_sep`` Full-width ``"─" * table_width``.
    ``announce``  ``─── message ─────────`` flush to table width.
    *(absent)*    Delegates to the underlying :class:`TextFormatter`.
    ============  ===========================================================

    Parameters
    ----------
    col_width:
        Width of each column in ``header`` / ``row`` rendering.  Must
        match the ``fmt`` width used by :class:`molix.core.hooks.Log`.
    row_fmt:
        Numeric format applied to each value in a ``row`` record
        (``"{:>12.4g}"`` by default — width + general-precision 4).
    """

    def __init__(
        self,
        *,
        col_width: int = 12,
        row_fmt: str = "{:>12.4g}",
    ) -> None:
        self._col_width = col_width
        self._row_fmt = row_fmt
        self._fallback = TextFormatter()

    def format(self, record: LogRecord) -> str:
        extra = record.extra or {}
        kind = extra.get("kind")
        if kind == "header":
            cols = extra["columns"]
            return " ".join(f"{c:>{self._col_width}}" for c in cols)
        if kind == "row":
            cols = extra["columns"]
            values = extra["values"]
            parts: list[str] = []
            for c in cols:
                v = values.get(c)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                    parts.append(self._row_fmt.format(v))
                else:
                    parts.append(f"{'nan':>{self._col_width}}")
            return " ".join(parts)
        width = int(extra.get("table_width", get_table_width()))
        if kind == "epoch_sep":
            return "─" * width
        if kind == "announce":
            msg = record.message
            prefix = f"─── {msg} "
            pad = max(3, width - len(prefix))
            return prefix + "─" * pad
        return self._fallback.format(record)


class CSVMetricsFormatter(Formatter):
    """Stateful: emit a CSV header on the first ``header`` record, rows after.

    The formatter silently drops non-metrics records (``None`` return is
    filtered by the handler layer). Pair with a :class:`ChannelFilter`
    on ``molix.metrics`` + a :class:`KindFilter` on ``{"header", "row"}``
    so the sink file never contains stray lines.
    """

    def __init__(self) -> None:
        self._columns: list[str] | None = None

    def format(self, record: LogRecord) -> str:
        extra = record.extra or {}
        kind = extra.get("kind")
        if kind == "header":
            cols = list(extra["columns"])
            self._columns = cols
            return ",".join(_csv_cell(c) for c in cols)
        if kind == "row":
            cols = self._columns or list(extra.get("columns", []))
            if not cols:
                return ""
            values = extra["values"]
            return ",".join(_csv_cell(values.get(c)) for c in cols)
        return ""


class EventFormatter(Formatter):
    """Timestamped plain-text events for ``events.log``.

    Renders as ``"<ISO-timestamp> <message>"`` — no level / module
    prefix because the file's purpose is a human-scannable event
    timeline, not a structured audit trail (that's ``train.log``).
    """

    def format(self, record: LogRecord) -> str:
        return f"{record.timestamp.isoformat(timespec='seconds')} {record.message}"


class ChannelFilter(Filter):
    """Pass records whose logger name starts with *prefix*.

    ``ChannelFilter("molix.metrics")`` passes every record emitted from
    the metrics channel (including nested child loggers, should any be
    added later).
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def filter(self, record: LogRecord) -> bool:
        name = record.logger_name or ""
        return name == self._prefix or name.startswith(self._prefix + ".")


class KindFilter(Filter):
    """Pass records whose ``extra["kind"]`` is in the allow-list."""

    def __init__(self, *kinds: str) -> None:
        self._kinds = frozenset(kinds)

    def filter(self, record: LogRecord) -> bool:
        extra = record.extra or {}
        return extra.get("kind") in self._kinds


class _HeaderOncePerColumnSet(Filter):
    """Drop ``kind=header`` records whose column set matches the previous one.

    The :class:`molix.core.hooks.Log` hook re-emits a ``header`` record
    at every periodic reprint and after each epoch boundary so the
    console view stays readable. For a CSV sink those repeats would
    produce duplicate header lines, confusing ``pandas.read_csv`` and
    other row-oriented readers. This filter keeps one header per unique
    column layout and lets every row through untouched.
    """

    def __init__(self) -> None:
        self._last_cols: tuple[str, ...] | None = None

    def filter(self, record: LogRecord) -> bool:
        extra = record.extra or {}
        if extra.get("kind") != "header":
            return True
        cols = tuple(extra.get("columns", ()))
        if cols == self._last_cols:
            return False
        self._last_cols = cols
        return True


class _StdoutConsoleFilter(Filter):
    """Gate for the stdout stream handler.

    Admits everything from the metrics / events channels (pretty output
    is why they exist), plus any record at ``console_level`` or above —
    so warnings and errors still show up alongside the training table.
    """

    def __init__(self, console_level: Level) -> None:
        self._lvl = console_level

    def filter(self, record: LogRecord) -> bool:
        name = record.logger_name or ""
        if name == METRICS_LOGGER_NAME or name.startswith(METRICS_LOGGER_NAME + "."):
            return True
        if name == EVENTS_LOGGER_NAME or name.startswith(EVENTS_LOGGER_NAME + "."):
            return True
        return record.level >= self._lvl


def configure_run(
    run_dir: str | Path,
    *,
    console_level: Level | str | int = Level.WARNING,
    file_level: Level | str | int = Level.INFO,
    col_width: int = 12,
    row_fmt: str = "{:>12.4g}",
    stream: IO[str] | None = None,
) -> Logger:
    """One-shot router setup for a training run.

    Configures the ``molix`` logger with five handlers whose filters
    route records to channel-specific sinks:

    ==================  ===============  ====================================
    Sink                Level            Filter(s)
    ==================  ===============  ====================================
    stdout              any              metrics OR events OR level ≥ console
    ``train.log``       ``file_level``   (none — full audit)
    ``metrics.csv``     INFO             channel=metrics, kind∈{header,row}
    ``events.log``      INFO             channel=events
    ``warnings.log``    WARNING          (none — level threshold handles it)
    ==================  ===============  ====================================

    The console stream uses :class:`PrettyTextFormatter`; ``metrics.csv``
    uses :class:`CSVMetricsFormatter`; ``events.log`` uses
    :class:`EventFormatter`; the rest use :class:`TextFormatter`.

    Existing handlers on the ``molix`` logger are closed and replaced so
    each call produces a clean routing graph.

    Parameters
    ----------
    run_dir:
        Directory for the four log files; created if missing.
    console_level:
        Level gate for non-metric / non-event records on stdout.
        Defaults to ``WARNING`` so the console shows only the training
        table, events, and warnings+ — the textbook "clean console"
        pattern from hooks.md § Console Output Design.
    file_level:
        Level gate for ``train.log``. ``INFO`` captures every structured
        event including hook activations; drop to ``DEBUG`` for deep
        traces.
    col_width / row_fmt:
        Passed straight through to :class:`PrettyTextFormatter`. Must
        match the :class:`molix.core.hooks.Log` ``fmt`` width.
    stream:
        Override for the stdout stream (defaults to ``sys.stdout``).

    Returns
    -------
    The configured molix root :class:`~mollog.Logger`.
    """
    console_lvl = Level.coerce(console_level)
    file_lvl = Level.coerce(file_level)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    root = _mollog_get_logger(MOLIX_LOGGER_NAME)
    root.level = Level.DEBUG  # root lets everything through; handlers gate.
    root.propagate = False
    root.clear_handlers(close=True)

    # 1. stdout — pretty view (metrics + events + WARNING+ only).
    stdout_handler = StreamHandler(stream=stream or sys.stdout, level=Level.DEBUG)
    stdout_handler.set_formatter(
        PrettyTextFormatter(col_width=col_width, row_fmt=row_fmt)
    )
    stdout_handler.add_filter(_StdoutConsoleFilter(console_lvl))
    root.add_handler(stdout_handler)

    # 2. train.log — full structured audit trail.
    train_handler = FileHandler(
        run_dir / "train.log", mode="w", level=file_lvl, encoding="utf-8",
    )
    train_handler.set_formatter(TextFormatter())
    root.add_handler(train_handler)

    # 3. metrics.csv — structured training-table sink.
    csv_handler = FileHandler(
        run_dir / "metrics.csv", mode="w", level=Level.INFO, encoding="utf-8",
    )
    csv_handler.set_formatter(CSVMetricsFormatter())
    csv_handler.add_filter(ChannelFilter(METRICS_LOGGER_NAME))
    csv_handler.add_filter(KindFilter("header", "row"))
    csv_handler.add_filter(_HeaderOncePerColumnSet())
    root.add_handler(csv_handler)

    # 4. events.log — inline-announce timeline.
    events_handler = FileHandler(
        run_dir / "events.log", mode="w", level=Level.INFO, encoding="utf-8",
    )
    events_handler.set_formatter(EventFormatter())
    events_handler.add_filter(ChannelFilter(EVENTS_LOGGER_NAME))
    root.add_handler(events_handler)

    # 5. warnings.log — everything WARNING+, structured.
    warn_handler = FileHandler(
        run_dir / "warnings.log", mode="w", level=Level.WARNING, encoding="utf-8",
    )
    warn_handler.set_formatter(TextFormatter())
    root.add_handler(warn_handler)

    return root
