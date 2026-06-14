"""Shared fixtures for the Sonata physics-invariant integration suite.

This conftest provides one float64 ``Sonata`` pipeline (Allegro encoder +
``PermMultipoleHead`` + ``EwaldMultipoleEnergy``), three deterministic
``TensorDict`` samples (free / periodic / charged-periodic), a callable
SO(3) random-rotation factory, and a callable random-traceless-quadrupole
factory. All tensors are float64 — the algebraic-identity tolerances
(1e-10 / 1e-12) the suite asserts are below float32 ULP at the eV scale.

Layout decisions:

* ``sonata_pipeline`` is **module-scoped** so each test file rebuilds the
  encoder + head once, not once per test (≥10× speedup on the periodic
  finite-difference tests, which call ``sonata`` 50+ times).
* Float64 precision is set inside the fixture body (``config.ftype =
  float64``) and restored at teardown, so existing tests in this
  directory (``test_sonata.py``, ``test_byteff_heads.py``,
  ``test_composition.py``) are unaffected — the swap only happens for
  tests that depend on ``sonata_pipeline``.
* Sample batches use deterministic seeded random positions in Å and
  intentionally non-trivial (non-symmetric) atom layouts so symmetry
  tests can detect a regression.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tensordict import TensorDict

from molix.config import config
from molpot.composition import Sonata, build_sonata
from molzoo import Allegro

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_bidirectional_edges(offsets: list[tuple[int, int]]) -> torch.Tensor:
    """Build a full intra-graph edge_index (no self-loops) from per-graph slices.

    ``offsets`` is a list of ``(start, n_atoms)`` tuples; for each tuple
    we add all ``n_atoms · (n_atoms - 1)`` directed pairs in that range.

    Returns ``(E, 2)`` long tensor.
    """
    pairs: list[list[int]] = []
    for start, n in offsets:
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append([start + i, start + j])
    return torch.tensor(pairs, dtype=torch.long)


def _make_batch_fp64(
    *,
    pos: torch.Tensor,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    batch_idx: torch.Tensor,
    total_charge: torch.Tensor,
    cell: torch.Tensor | None = None,
) -> TensorDict:
    """Assemble a float64 ``TensorDict`` with per-graph ``total_charge``."""
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1)
    n_atoms = pos.shape[0]
    n_edges = edge_index.shape[0]
    n_graphs = int(total_charge.shape[0])

    num_atoms = torch.zeros(n_graphs, dtype=torch.long)
    num_atoms.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))

    graphs_kwargs: dict = {
        "num_atoms": num_atoms,
        "total_charge": total_charge,
        "batch_size": [n_graphs],
    }
    if cell is not None:
        graphs_kwargs["cell"] = cell

    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[n_atoms]),
        edges=TensorDict(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[n_edges],
        ),
        graphs=TensorDict(**graphs_kwargs),
        batch_size=[],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sonata_pipeline() -> Sonata:
    """Float64 Sonata pipeline: Allegro(l_max=2, expose_tensor_track=True)
    → PermMultipoleHead(charge + dipole + quadrupole) → EwaldMultipoleEnergy.

    No short-range head — the integration suite asserts on the
    Ewald-only contributions so we keep the pipeline minimal.
    """
    orig_ftype = config["ftype"]
    config["ftype"] = torch.float64
    try:
        torch.manual_seed(0)
        encoder = Allegro(
            num_elements=10,
            num_scalar_features=16,
            num_tensor_features=4,
            r_max=5.0,
            num_bessel=8,
            num_layers=2,
            l_max=2,
            type_embed_dim=16,
            latent_mlp_depth=1,
            latent_mlp_width=32,
            avg_num_neighbors=12.0,
            expose_tensor_track=True,
        )
        sonata = build_sonata(
            encoder,
            sigma=1.0,
            dl=2.0,
            charge=True,
            dipole=True,
            quadrupole=True,
            constrain_total_charge=True,
            avg_num_neighbors=12.0,
        )
        # cuequivariance_torch.Linear ignores `config.ftype` and creates
        # float32 weights regardless; cast the whole module tree to
        # float64 after construction so the head's `cuet.Linear`
        # collapse paths line up with the float64 batch inputs.
        sonata = sonata.double()
        sonata.eval()
        yield sonata
    finally:
        config["ftype"] = orig_ftype


# ---------------------------------------------------------------------------
# Sample batches — every fixture is float64, deterministic, and small
# enough that finite-difference loops are tractable.
# ---------------------------------------------------------------------------


def _two_graph_pos_Z(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two graphs (4 atoms + 4 atoms) with non-trivial, asymmetric layouts.

    Returns ``(pos, Z, batch_idx)`` all on the CPU in float64 / int64.
    """
    torch.manual_seed(seed)
    pos = torch.tensor(
        [
            # graph 0 — 4 atoms, ~1.4 Å bonds
            [0.10, 0.20, 0.05],
            [1.55, 0.15, 0.10],
            [0.85, 1.30, -0.05],
            [-0.65, 1.25, 0.20],
            # graph 1 — 4 atoms, square-ish
            [0.05, 0.10, -0.10],
            [1.45, -0.05, 0.05],
            [0.10, 1.40, 0.15],
            [1.50, 1.45, -0.10],
        ],
        dtype=torch.float64,
    )
    Z = torch.tensor([1, 6, 8, 7, 1, 6, 8, 7], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return pos, Z, batch_idx


@pytest.fixture
def sample_neutral_batch_open() -> TensorDict:
    """Two 4-atom graphs, no cell, total_charge = 0 per graph."""
    pos, Z, batch_idx = _two_graph_pos_Z(seed=0)
    edge_index = _full_bidirectional_edges([(0, 4), (4, 4)])
    total_charge = torch.zeros(2, dtype=torch.float64)
    return _make_batch_fp64(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch_idx=batch_idx,
        total_charge=total_charge,
        cell=None,
    )


@pytest.fixture
def sample_neutral_batch_periodic() -> TensorDict:
    """Two 4-atom graphs, cubic cells (L = 10 Å each), total_charge = 0."""
    pos, Z, batch_idx = _two_graph_pos_Z(seed=0)
    edge_index = _full_bidirectional_edges([(0, 4), (4, 4)])
    total_charge = torch.zeros(2, dtype=torch.float64)
    cell = 10.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    return _make_batch_fp64(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch_idx=batch_idx,
        total_charge=total_charge,
        cell=cell,
    )


@pytest.fixture
def sample_charged_batch_periodic() -> TensorDict:
    """Two 4-atom graphs, cubic cells, distinct non-zero per-graph total_charge.

    Graph 0 carries ``+1 e``; graph 1 carries ``-2 e``. Charges are real-
    space net charges; the LES Ewald path tolerates non-neutrality (Flag
    #3, k=0 background) so this is a valid input — and exercising the
    head's ``constrain_total_charge`` projection on a non-zero target is
    the whole point of the per-graph projection contract.
    """
    pos, Z, batch_idx = _two_graph_pos_Z(seed=0)
    edge_index = _full_bidirectional_edges([(0, 4), (4, 4)])
    total_charge = torch.tensor([1.0, -2.0], dtype=torch.float64)
    cell = 10.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    return _make_batch_fp64(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch_idx=batch_idx,
        total_charge=total_charge,
        cell=cell,
    )


# ---------------------------------------------------------------------------
# Random rotations + traceless quadrupoles
# ---------------------------------------------------------------------------


@pytest.fixture
def random_rotation_matrix() -> Callable[[], torch.Tensor]:
    """Factory for a fresh float64 SO(3) rotation matrix via QR.

    QR of a Gaussian matrix is uniform on O(3); we flip the sign of one
    column when the determinant is negative to land in SO(3) (proper
    rotations). Each call uses the current ``torch.manual_seed`` state,
    so seed deterministically before invoking.
    """

    def _make() -> torch.Tensor:
        A = torch.randn(3, 3, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        if torch.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    return _make


@pytest.fixture
def random_traceless_Q() -> Callable[[int], torch.Tensor]:
    """Factory for synthetic ``(N, 5)`` cuet-2e quadrupole moments.

    The cuet ``2e`` real-spherical basis is by construction the 5
    independent components of a symmetric traceless ℓ=2 tensor — any
    ``(N, 5)`` Gaussian draw maps to a symmetric traceless ``(N, 3, 3)``
    Cartesian quadrupole under the LES convention. Useful for direct
    contract checks on ``_theta_to_cartesian_quadrupole`` independent of
    the model output.
    """

    def _make(n: int) -> torch.Tensor:
        return torch.randn(n, 5, dtype=torch.float64)

    return _make
