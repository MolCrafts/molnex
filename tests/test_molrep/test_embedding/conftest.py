"""Shared fixtures for the :mod:`molrep.embedding` unit suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from molix import config


@pytest.fixture
def fp64() -> Iterator[None]:
    """Run the case under the global fp64 precision, restoring the previous one.

    Every layer in :mod:`molrep.embedding` bakes ``config["ftype"]`` into its
    parameters and buffers at construction time (the contract documented in
    :mod:`molix.config`), so the precision has to be switched *before* the
    module is built and handed back afterwards.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")
