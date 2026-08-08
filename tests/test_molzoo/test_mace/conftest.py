"""MACE-specific fixtures for the ``molzoo.mace`` sub-package tests.

Only MACE-family constants / specs live here. Batch construction goes
through :func:`tests.conftest.make_graph_batch` — this package deliberately
does not grow a second batch builder.

The tiny configurations below mirror ``tests/test_molzoo/test_mace_matpes.py``
so ``MACEEncoder`` can be diffed against the flat variants key-for-key.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from molix import config

#: Element table (z-table) shared by every model built in this package.
#: Ascending and duplicate-free — ``torch.searchsorted`` requires it.
ATOMIC_NUMBERS: list[int] = [1, 6, 8]

#: Per-element reference energies ``E0`` in eV/atom, same order as
#: :data:`ATOMIC_NUMBERS`.
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]

#: Tiny MACE-MatPES hyper-parameters (l_max=1, 16 channels): fast on CPU and
#: structurally identical to the shipped model. Passed verbatim to both
#: ``MACEMatpesSpec`` and the flat ``MACEMatpes`` constructor, which is what
#: makes the ``state_dict`` parity assertion meaningful.
TINY_MATPES_KWARGS: dict[str, Any] = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 1,
    "num_features": 16,
    "max_hidden_l": 1,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "radial_mlp": [8],
    "use_fallback": True,  # CPU tests: fused kernels need a GPU + ops wheel
}

#: Tiny MACE-OMOL hyper-parameters (l_max=1, 16 channels). ``use_fallback`` is
#: absent: the flat ``MACEOMol`` hard-codes the fused path, so the spec must
#: default/carry ``use_fallback=False`` for the two stacks to agree key-wise.
TINY_OMOL_KWARGS: dict[str, Any] = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 1,
    "num_features": 16,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "edge_channels": 8,
}


@pytest.fixture(autouse=True)
def fp64():
    """Build every model in this package at fp64.

    cuEquivariance freezes its working precision at construction, so a model
    built under the default fp32 and then ``.double()``-d still contracts in
    float32 — ~1e-8 eV of noise, which is fine for inference but swamps the
    invariance assertions here. Foundation-weight use is fp64 anyway, so the
    tests exercise that path.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


@pytest.fixture
def tiny_matpes_spec():
    """Tiny :class:`molzoo.mace.spec.MACEMatpesSpec`.

    Imported inside the fixture on purpose: ``molzoo.mace.spec`` does not
    exist yet, and a module-level import would break collection of every
    other test file in this package instead of only the cases that need it.
    """
    from molzoo.mace.spec import MACEMatpesSpec

    return MACEMatpesSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_MATPES_KWARGS,
    )


@pytest.fixture
def tiny_omol_spec():
    """Tiny :class:`molzoo.mace.spec.MACEOMolSpec` (see :func:`tiny_matpes_spec`)."""
    from molzoo.mace.spec import MACEOMolSpec

    return MACEOMolSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        use_fallback=False,  # the flat MACEOMol hard-codes the fused path
        **TINY_OMOL_KWARGS,
    )
