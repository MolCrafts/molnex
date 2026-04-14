"""Stdlib-compatible logging facade for molix.

Drop-in replacement for ``import logging`` inside the molix stack.  The
public API mirrors the standard library's ``logging`` module — level
constants (``DEBUG``, ``INFO``, ``WARNING`` …), ``getLogger(name)``,
``basicConfig(...)`` — but records flow through mollog.  Everything is
scoped to the ``"molix"`` logger with ``propagate=False``, so
configuring it never mutates the mollog root logger that other packages
in the process may rely on.

Typical use::

    from molix import logging

    logging.basicConfig(
        level=logging.INFO,
        filename=run_ctx.logs_dir / "train.log",
        filemode="w",
    )
    log = logging.getLogger(__name__)
    log.info("starting run", steps=200)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO

from mollog import FileHandler, Level, Logger, StreamHandler, TextFormatter
from mollog import get_logger as _mollog_get_logger
from mollog.formatter import Formatter

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
    shared_formatter = formatter or TextFormatter()

    logger = _mollog_get_logger(MOLIX_LOGGER_NAME)
    logger.level = lvl
    logger.propagate = False

    if force:
        logger.clear_handlers(close=True)

    stream_handler = StreamHandler(stream=stream or sys.stderr, level=lvl)
    stream_handler.set_formatter(shared_formatter)
    logger.add_handler(stream_handler)

    if filename is not None:
        file_handler = FileHandler(
            Path(filename),
            mode=filemode,
            encoding=encoding,
            level=file_lvl,
        )
        file_handler.set_formatter(shared_formatter)
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
