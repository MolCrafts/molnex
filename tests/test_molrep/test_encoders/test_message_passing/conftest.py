"""Shared fixtures for the message-passing encoder tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def atom_dim() -> int:
    return 8


@pytest.fixture
def bond_dim() -> int:
    return 8
