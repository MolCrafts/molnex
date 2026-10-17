"""Shared fixtures for the ``molzoo.mace`` sub-package tests.

Only MACE-family constants, geometries, specs and models live here. Batch
construction goes through :func:`tests.conftest.make_graph_batch` — this
package deliberately does not grow a second batch builder.

Everything a second module in this package needs is defined here exactly once:

* :data:`CLUSTER_POS` / :data:`CLUSTER_Z` and the ``cluster`` /
  ``periodic_cluster`` / ``omol_cluster`` / ``pair_batch`` fixtures — the
  fixed geometries (no RNG anywhere);
* :func:`full_edge_index` — the ordered intra-graph pair list;
* the ``tiny_*_spec`` fixtures — one tiny spec per shipped family;
* ``matpes_model`` / ``omol_model`` — the tiny energy + force products.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from molzoo.mace.potential import MaceMatpes, MaceOmol
from molzoo.mace.spec import (
    MACELesSpec,
    MACEMatpesSpec,
    MACEOffSpec,
    MACEOMolSpec,
    MACEPolarSpec,
)
from tests.conftest import make_graph_batch

#: Element table (z-table) shared by every model built in this package.
#: Ascending and duplicate-free — ``torch.searchsorted`` requires it.
ATOMIC_NUMBERS: list[int] = [1, 6, 8]

#: Per-element reference energies ``E0`` in eV/atom, same order as
#: :data:`ATOMIC_NUMBERS`.
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]

#: Tiny MACE-MatPES hyper-parameters (l_max=1, 16 channels): fast on CPU and
#: structurally identical to the shipped model.
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
}

#: Tiny MACE-OMOL hyper-parameters (l_max=1, 16 channels).
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

#: Tiny MACE-OFF hyper-parameters (l_max=1, 16 channels), shared with MACE-LES.
TINY_OFF_KWARGS: dict[str, Any] = {
    "avg_num_neighbors": 5.0,
    "r_max": 5.0,
    "num_bessel": 4,
    "l_max": 1,
    "num_features": 16,
    "correlation": 2,
    "mlp_dim": 8,
    "radial_mlp": [8],
}

#: Tiny Polar backbone: ``l_max=3`` so a wrong ``hidden_l = l_max-1`` schedule
#: would emit last-layer width 144 instead of 64.
TINY_POLAR_KWARGS: dict[str, Any] = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 3,
    "num_features": 16,
    "max_hidden_l": 1,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "edge_channels": 8,
}

#: A five-atom cluster at literal coordinates (Å) — no RNG anywhere.
CLUSTER_POS: list[list[float]] = [
    [0.00, 0.00, 0.00],
    [0.95, 0.00, 0.00],
    [-0.24, 0.93, 0.00],
    [0.00, 0.00, 1.40],
    [1.20, 1.10, 0.60],
]

#: Atomic numbers of :data:`CLUSTER_POS`, all inside the ``[1, 6, 8]`` table.
CLUSTER_Z: list[int] = [8, 1, 1, 6, 1]

#: Graph membership of :data:`CLUSTER_POS` — one graph, ``B = 1``.
SINGLE_GRAPH: list[int] = [0] * len(CLUSTER_Z)

#: Two clusters (4 + 3 atoms) in one batch — exercises the per-graph reduction.
PAIR_POS: list[list[float]] = CLUSTER_POS[:4] + [
    [5.00, 5.00, 5.00],
    [5.95, 5.00, 5.00],
    [5.00, 5.90, 5.00],
]
PAIR_Z: list[int] = [8, 1, 1, 6, 8, 1, 1]
PAIR_BATCH: list[int] = [0, 0, 0, 0, 1, 1, 1]

#: Periodic shift added to the first two edges (Å) — ``unit_shifts @ cell``.
PERIODIC_SHIFT = 2.0


def full_edge_index(batch: Sequence[int]) -> torch.Tensor:
    """All ordered intra-graph atom pairs as ``(E, 2)`` ``[source, target]``.

    Args:
        batch: Graph index per atom; ``len(batch)`` is the atom count.

    Returns:
        ``(E, 2)`` edge index on the repo convention (``[:, 0]`` = source,
        ``[:, 1]`` = target), self-pairs and cross-graph pairs excluded.
    """
    n_atoms = len(batch)
    pairs = [
        [i, j] for i in range(n_atoms) for j in range(n_atoms) if i != j and batch[i] == batch[j]
    ]
    return torch.tensor(pairs, dtype=torch.long)


def wake_zero_init_readout(model: torch.nn.Module) -> None:
    """Pin the OMOL readout to a private generator, off the global RNG.

    ``molzoo.mace.heads._ScalarO3Linear`` draws its weight on the global RNG
    and zero-initialises its bias; filling both scalar linears from a fixed
    private generator makes the OMOL model's interaction energy depend on
    that generator alone, not on how much global RNG the construction
    consumed before the readout was built. Checkpoint use is unaffected —
    official weights overwrite these entries.

    Args:
        model: An OMOL-configured model exposing ``readout.linear_mid`` /
            ``readout.linear_2``.
    """
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for linear in (model.readout.linear_mid, model.readout.linear_2):
            linear.weight.normal_(generator=generator)
            linear.bias.normal_(generator=generator)


@pytest.fixture(autouse=True)
def _build_at_fp64(fp64):
    """Every MACE test builds at fp64 (``tests.conftest.fp64``)."""


@pytest.fixture
def tiny_matpes_spec() -> MACEMatpesSpec:
    """Tiny :class:`molzoo.mace.spec.MACEMatpesSpec`."""
    return MACEMatpesSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_MATPES_KWARGS,
    )


@pytest.fixture
def tiny_omol_spec() -> MACEOMolSpec:
    """Tiny :class:`molzoo.mace.spec.MACEOMolSpec`."""
    return MACEOMolSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_OMOL_KWARGS,
    )


@pytest.fixture
def tiny_off_spec() -> MACEOffSpec:
    """Tiny :class:`molzoo.mace.spec.MACEOffSpec`."""
    return MACEOffSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_OFF_KWARGS,
    )


@pytest.fixture
def tiny_les_spec() -> MACELesSpec:
    """Tiny :class:`molzoo.mace.spec.MACELesSpec` on the MACE-OFF body, residual first."""
    return MACELesSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        residual_first=True,
        **TINY_OFF_KWARGS,
    )


@pytest.fixture
def tiny_polar_spec() -> MACEPolarSpec:
    """Tiny :class:`molzoo.mace.spec.MACEPolarSpec`."""
    return MACEPolarSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **TINY_POLAR_KWARGS,
    )


@pytest.fixture
def cluster() -> TensorDict:
    """A shift-free five-atom, one-graph batch (no charge/spin conditioning)."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
    )


@pytest.fixture
def periodic_cluster() -> TensorDict:
    """The same cluster with a non-zero image shift folded into ``pairs.pair_diff`` (two edges)."""
    edge_index = full_edge_index(SINGLE_GRAPH)
    shifts = torch.zeros(edge_index.shape[0], 3, dtype=torch.float64)
    shifts[0, 0] = PERIODIC_SHIFT
    shifts[1, 0] = -PERIODIC_SHIFT
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=edge_index,
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        shifts=shifts,
    )


@pytest.fixture
def pair_batch() -> TensorDict:
    """Two separated clusters in one batch (``B = 2``)."""
    return make_graph_batch(
        pos=torch.tensor(PAIR_POS, dtype=torch.float64),
        Z=torch.tensor(PAIR_Z, dtype=torch.long),
        pair_index=full_edge_index(PAIR_BATCH),
        batch=torch.tensor(PAIR_BATCH, dtype=torch.long),
    )


@pytest.fixture
def omol_cluster() -> TensorDict:
    """The cluster with explicit OMOL conditioning (neutral closed-shell singlet)."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs={
            "total_charge": torch.zeros(1, dtype=torch.long),
            "total_spin": torch.ones(1, dtype=torch.long),
        },
    )


@pytest.fixture
def matpes_model(tiny_matpes_spec: MACEMatpesSpec) -> MaceMatpes:
    """Tiny energy + force :class:`~molzoo.mace.potential.MaceMatpes`, seeded."""
    torch.manual_seed(0)
    return MaceMatpes(tiny_matpes_spec, wrt=("atoms.pos",)).eval()


@pytest.fixture
def omol_model(tiny_omol_spec: MACEOMolSpec) -> MaceOmol:
    """Tiny energy + force :class:`~molzoo.mace.potential.MaceOmol` with a woken readout.

    See :func:`wake_zero_init_readout` for why the readout is filled.
    """
    torch.manual_seed(0)
    model = MaceOmol(tiny_omol_spec, wrt=("atoms.pos",)).eval()
    wake_zero_init_readout(model)
    return model
