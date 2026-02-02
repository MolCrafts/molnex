"""Centralized logging for molix using external mollog library."""
from __future__ import annotations

from mollog import get_logger

def getLogger(name: str):
    """Get a logger instance with the specified name."""
    return get_logger(name)
