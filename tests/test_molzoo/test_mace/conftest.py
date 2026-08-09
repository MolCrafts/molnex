"""Shared fixtures for the ``molzoo.mace`` sub-package tests.

Only MACE-family constants, geometries, models and specs live here. Batch
construction goes through :func:`tests.conftest.make_graph_batch` — this
package deliberately does not grow a second batch builder (the three private
builders of the pre-cutover flat test files were folded into it by
``mace-subpackage-restructure-06-wire``, and none may come back).

The tiny configurations below are the ones the pre-cutover flat variant tests
used, so ``MACEEncoder`` / ``MACEPotential`` can be diffed against the keyword
variants key-for-key.

Everything a second module in this package needs is defined here exactly once:

* :data:`CLUSTER_POS` / :data:`CLUSTER_Z` and the ``cluster`` /
  ``periodic_cluster`` / ``omol_cluster`` / ``pair_batch`` fixtures — the
  fixed geometries (no RNG anywhere);
* :func:`full_edge_index` — the ordered intra-graph pair list;
* :func:`wake_zero_init_readout` — pins the OMOL readout to a private
  generator, so state-transfer parity does not ride on the global RNG;
* ``matpes_variant`` / ``omol_variant`` — the keyword-constructed foundation
  models, used both as the subject of ``test_variants.py`` and as the
  raw-tensor (``energy_forces``) oracle of ``test_potential.py`` /
  ``test_encoder.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from molix import config
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec
from molzoo.mace.variants import MACEMatpes, MACEOMol
from tests.conftest import make_graph_batch

#: Element table (z-table) shared by every model built in this package.
#: Ascending and duplicate-free — ``torch.searchsorted`` requires it.
ATOMIC_NUMBERS: list[int] = [1, 6, 8]

#: Per-element reference energies ``E0`` in eV/atom, same order as
#: :data:`ATOMIC_NUMBERS`.
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]

#: Tiny MACE-MatPES hyper-parameters (l_max=1, 16 channels): fast on CPU and
#: structurally identical to the shipped model. Passed verbatim to both
#: ``MACEMatpesSpec`` and the keyword ``MACEMatpes`` constructor, which is what
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
#: absent: :class:`~molzoo.mace.variants.MACEOMol` takes no such keyword, so the
#: spec must default/carry ``use_fallback=False`` for the two stacks to agree
#: key-wise.
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


def raw_tensors(
    batch: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """``(pos, Z, edge_index, atom_batch, num_graphs)`` off a post-collate batch.

    The raw-tensor seams (``energy_core`` / ``energy_forces``) take these five
    in this order; every case that crosses from the batch schema to a flat call
    goes through here rather than unpacking by hand.

    Args:
        batch: Post-collate ``TensorDict`` with ``atoms`` / ``edges``.

    Returns:
        Positions ``(N, 3)`` Å, atomic numbers ``(N,)``, edge index ``(E, 2)``,
        graph index per atom ``(N,)`` and the graph count ``B``.
    """
    return (
        batch["atoms", "pos"],
        batch["atoms", "Z"],
        batch["edges", "edge_index"],
        batch["atoms", "batch"],
        int(batch["atoms", "batch"].max()) + 1,
    )


def wake_zero_init_readout(model: torch.nn.Module) -> None:
    """Pin the OMOL readout to a private generator, off the global RNG.

    ``molrep.readout.mace._ScalarO3Linear`` now draws its weight from ``N(0, 1)``
    on the **global** RNG and zero-initialises only its bias, so a fresh OMOL
    ``NonLinearBiasReadout`` is already non-trivial and the assertions here are
    no longer at risk of being vacuous. (They were: the weight used to be
    zero-initialised too, which made an untrained model's interaction energy
    position-independent and every force identically zero, so parity, physics
    and equivalence assertions all reduced to ``0 == 0``.)

    The helper stays as belt-and-braces determinism for the state-transfer
    parity tests: it fills both scalar linears from a private, fixed
    ``torch.Generator``, so the values a test compares across two models depend
    on that generator alone and not on how much global RNG each model happened
    to consume before its readout was built. Checkpoint use is unaffected —
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
        use_fallback=False,  # MACEOMol takes no such keyword; the spec defaults
        **TINY_OMOL_KWARGS,
    )


@pytest.fixture
def cluster() -> TensorDict:
    """A shift-free five-atom, one-graph batch (no charge/spin conditioning)."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
    )


@pytest.fixture
def periodic_cluster() -> TensorDict:
    """The same cluster with a non-zero ``edges.shifts`` on the first two edges."""
    edge_index = full_edge_index(SINGLE_GRAPH)
    shifts = torch.zeros(edge_index.shape[0], 3, dtype=torch.float64)
    shifts[0, 0] = PERIODIC_SHIFT
    shifts[1, 0] = -PERIODIC_SHIFT
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=edge_index,
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        shifts=shifts,
    )


@pytest.fixture
def pair_batch() -> TensorDict:
    """Two separated clusters in one batch (``B = 2``)."""
    return make_graph_batch(
        pos=torch.tensor(PAIR_POS, dtype=torch.float64),
        Z=torch.tensor(PAIR_Z, dtype=torch.long),
        edge_index=full_edge_index(PAIR_BATCH),
        batch=torch.tensor(PAIR_BATCH, dtype=torch.long),
    )


@pytest.fixture
def omol_cluster() -> TensorDict:
    """The cluster with explicit OMOL conditioning (neutral closed-shell singlet)."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs={
            "total_charge": torch.zeros(1, dtype=torch.long),
            "total_spin": torch.ones(1, dtype=torch.long),
        },
    )


@pytest.fixture
def matpes_variant() -> MACEMatpes:
    """Tiny :class:`~molzoo.mace.variants.MACEMatpes` in the keyword shape.

    ``atomic_energies`` is a **tensor**, as ``scripts/matpes_port/run_nve.py``
    passes it (``torch.tensor(cfg["atomic_energies"], dtype=config.ftype)``),
    even though :class:`~molzoo.mace.spec.MACEMatpesSpec` holds a list.
    """
    torch.manual_seed(0)
    return MACEMatpes(
        atomic_numbers=list(ATOMIC_NUMBERS),
        atomic_energies=torch.tensor(ATOMIC_ENERGIES),
        **TINY_MATPES_KWARGS,
    ).eval()


@pytest.fixture
def omol_variant() -> MACEOMol:
    """Tiny :class:`~molzoo.mace.variants.MACEOMol` with a woken readout.

    See :func:`wake_zero_init_readout` for why the readout is filled.
    """
    torch.manual_seed(0)
    model = MACEOMol(
        atomic_numbers=list(ATOMIC_NUMBERS),
        atomic_energies=torch.tensor(ATOMIC_ENERGIES),
        **TINY_OMOL_KWARGS,
    ).eval()
    wake_zero_init_readout(model)
    return model
