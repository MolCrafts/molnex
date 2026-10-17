"""Shared fixtures for the Sonata suite.

One float64 :class:`~molpot.composition.Sonata` (Allegro encoder +
``PermMultipoleHead`` + ``EwaldMultipoleEnergy``, no short-range head), built
at an explicit ``dtype=torch.float64`` and bound to the 10 Å cubic cell every
periodic sample here carries; three deterministic ``TensorDict`` samples
(free / periodic / charged-periodic); a callable SO(3) random-rotation factory
and a callable random-traceless-quadrupole factory. All tensors are float64 —
the algebraic-identity tolerances (1e-10 / 1e-12) the suite asserts are below
float32 ULP at the eV scale.

``sonata_pipeline`` is **module-scoped** so each test file rebuilds the
encoder + head once, not once per test (the finite-difference tests call the
energy dozens of times). Sample batches use deterministic seeded random
positions in Å and intentionally non-trivial (non-symmetric) atom layouts so
symmetry tests can detect a regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from tensordict import TensorDict

from molpot.composition import Sonata, SonataSpec
from molpot.heads import PermMultipoleHeadSpec
from molpot.potentials import EwaldMultipoleEnergySpec
from molzoo.allegro import AllegroEncoder, AllegroSpec

#: Edge length (Å) of the cubic cell of every periodic sample in this suite.
CELL_LENGTH = 10.0

#: Dataset-wide ⟨|N(i)|⟩ the encoder and the head normalise with.
AVG_NUM_NEIGHBORS = 12.0


def allegro_encoder(
    *, dtype: torch.dtype, l_max: int = 2, expose_tensor_track: bool = True
) -> AllegroEncoder:
    """A small Allegro body, seeded, exposing its tensor track by default."""
    torch.manual_seed(0)
    return AllegroEncoder(
        AllegroSpec(
            atomic_numbers=[1, 6, 7, 8],
            num_scalar_features=16,
            num_tensor_features=4,
            r_max=5.0,
            num_bessels=8,
            num_layers=2,
            l_max=l_max,
            radial_chemical_embed_dim=16,
            allegro_mlp_hidden_layers_depth=1,
            allegro_mlp_hidden_layers_width=32,
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            expose_tensor_track=expose_tensor_track,
        ),
        dtype=dtype,
    )


def sonata_spec(encoder: AllegroEncoder) -> SonataSpec:
    """Charge + dipole + quadrupole head on *encoder*'s width, LES-default Ewald."""
    return SonataSpec(
        head=PermMultipoleHeadSpec(
            input_dim=encoder.output_dim,
            avg_num_neighbors=AVG_NUM_NEIGHBORS,
            charge=True,
            dipole=True,
            quadrupole=True,
            constrain_total_charge=True,
        ),
        ewald=EwaldMultipoleEnergySpec(sigma=1.0, dl=2.0),
    )


def sonata_fp64(*, wrt: Sequence[str] = ("atoms.pos",)) -> Sonata:
    """A float64 Sonata in eval mode, its reciprocal grid still unbound."""
    encoder = allegro_encoder(dtype=torch.float64)
    return Sonata(sonata_spec(encoder), encoder, wrt=wrt, dtype=torch.float64).eval()


def full_bidirectional_edges(offsets: list[tuple[int, int]]) -> torch.Tensor:
    """Build a full intra-graph pair_index (no self-loops) from per-graph slices.

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


def make_batch(
    *,
    pos: torch.Tensor,
    Z: torch.Tensor,
    pair_index: torch.Tensor,
    batch_idx: torch.Tensor,
    total_charge: torch.Tensor,
    cell: torch.Tensor | None = None,
    minimum_image: bool = False,
) -> TensorDict:
    """Assemble a post-collate batch with per-graph ``total_charge``.

    The edge geometry is recomputed from ``pos``; with ``minimum_image`` and an
    orthorhombic ``cell`` ``(B, 3, 3)`` it is wrapped by
    ``Δr → Δr − L·round(Δr/L)``. Dtypes follow the tensors handed in.
    """
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    if minimum_image and cell is not None:
        pair_batch = batch_idx[pair_index[:, 0]]
        cell_diag = torch.diagonal(cell[pair_batch], dim1=-2, dim2=-1)  # (E, 3)
        pair_diff = pair_diff - cell_diag * torch.round(pair_diff / cell_diag)
    pair_dist = pair_diff.norm(dim=-1)
    n_graphs = int(total_charge.shape[0])
    num_atoms = torch.zeros(n_graphs, dtype=torch.long)
    num_atoms.scatter_add_(0, batch_idx, torch.ones_like(batch_idx))

    graphs = TensorDict(num_atoms=num_atoms, total_charge=total_charge, batch_size=[n_graphs])
    if cell is not None:
        graphs["cell"] = cell
    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[pos.shape[0]]),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=pair_diff,
            pair_dist=pair_dist,
            batch_size=[pair_index.shape[0]],
        ),
        graphs=graphs,
        batch_size=[],
    )


@pytest.fixture(scope="module")
def sonata_pipeline() -> Sonata:
    """Float64 Sonata (``wrt=("atoms.pos",)``) bound to the suite's 10 Å cubic cell.

    No short-range head — the suite asserts on the Ewald-only contributions.
    Binding freezes the reciprocal grid, so an evaluation on a rotated or
    translated copy of a periodic sample enumerates exactly the same k-vectors.
    """
    sonata = sonata_fp64()
    sonata.bind_cell(CELL_LENGTH * torch.eye(3, dtype=torch.float64))
    return sonata


def _two_graph_pos_Z() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two graphs (4 atoms + 4 atoms) with non-trivial, asymmetric layouts.

    Returns ``(pos, Z, batch_idx)`` all on the CPU in float64 / int64.
    """
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


def _two_graph_batch(total_charge: torch.Tensor, *, periodic: bool) -> TensorDict:
    pos, Z, batch_idx = _two_graph_pos_Z()
    cell = CELL_LENGTH * torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    return make_batch(
        pos=pos,
        Z=Z,
        pair_index=full_bidirectional_edges([(0, 4), (4, 4)]),
        batch_idx=batch_idx,
        total_charge=total_charge,
        cell=cell if periodic else None,
    )


@pytest.fixture
def sample_neutral_batch_open() -> TensorDict:
    """Two 4-atom graphs, no cell, total_charge = 0 per graph."""
    return _two_graph_batch(torch.zeros(2, dtype=torch.float64), periodic=False)


@pytest.fixture
def sample_neutral_batch_periodic() -> TensorDict:
    """Two 4-atom graphs, cubic cells (L = 10 Å each), total_charge = 0."""
    return _two_graph_batch(torch.zeros(2, dtype=torch.float64), periodic=True)


@pytest.fixture
def sample_charged_batch_periodic() -> TensorDict:
    """Two 4-atom graphs, cubic cells, distinct non-zero per-graph total_charge.

    Graph 0 carries ``+1 e``; graph 1 carries ``-2 e``. Charges are real-
    space net charges; the LES Ewald path tolerates non-neutrality (Flag
    #3, k=0 background) so this is a valid input — and exercising the
    head's ``constrain_total_charge`` projection on a non-zero target is
    the whole point of the per-graph projection contract.
    """
    return _two_graph_batch(torch.tensor([1.0, -2.0], dtype=torch.float64), periodic=True)


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
