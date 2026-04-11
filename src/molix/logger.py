"""Centralized logging for molix using external mollog library."""

from __future__ import annotations

import logging

from mollog import get_logger


def getLogger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return get_logger(name)  # type: ignore
