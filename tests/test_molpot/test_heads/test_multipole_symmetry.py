"""Symmetry tests for the Allegro + ``PermMultipoleHead`` ("permanent multipole")
composition — readout-only.

Three physical symmetries that every output of the composed pipeline must
satisfy — separate from the encoder-only checks in
``tests/test_molzoo/test_symmetry.py``.

==================================  ==============  ==============  ==============
output                              translation     rotation        permutation
==================================  ==============  ==============  ==============
``("atoms", "atomic_charges")``     invariant       invariant       equivariant
``("graphs", "molecular_dipole")``  invariant¹      equivariant     invariant
``("atoms", "atomic_dipoles")``     invariant       l=1 equivariant equivariant
``("atoms", "atomic_quadrupoles")`` invariant       l=2 equivariant equivariant
==================================  ==============  ==============  ==============

¹ Translation invariance of ``μ_mol`` only holds when total predicted
  charge is zero. ``PermMultipoleHead.constrain_total_charge=True`` projects
  ``Σ q_i = total_charge``; for a neutral target (``= 0``) that erases the
  shift term ``(Σ q_i)·t``.

This file deliberately does **not** test ``energy_es`` or autograd forces:
the screened-Coulomb / Ewald multipole energy now lives in
:class:`molpot.potentials.EwaldMultipoleEnergy`, whose own physics-oracle
suite at ``tests/test_molpot/test_potentials/test_ewald_multipole.py``
covers translation / rotation / permutation invariance of the energy and
the corresponding ``F = -∂E/∂pos`` autograd forces.

For l=2 (quadrupole) rotation tests, plain 3×3 matrices no longer suffice:
the output transforms under the Wigner ``D⁽²⁾(R)`` representation, so the
rotation tests parametrise on Euler angles and apply ``cuet.Rotation`` to
the output. Translation/permutation tests still use a 3×3 matrix because
those symmetries are independent of the irrep order.

The graph-transform helpers come from ``tests.symmetry_helpers`` so the
encoder-only and pipeline tests share one definition of "translate / rotate
/ permute a TensorDict".
"""

from __future__ import annotations

import math

import cuequivariance as cue
import cuequivariance_torch as cuet
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.heads import PermMultipoleHead
from molrep.utils.equivariance import random_rotation_matrix, rotate_vectors
from molzoo import Allegro
from tests.symmetry_helpers import (
    make_graph_batch,
    permute_graph,
    recompute_edge_geometry,
    rotate_graph,
    translate_graph,
)

SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Rotation helpers — the quadrupole tests need Euler angles paired with the
# matching 3×3 rotation matrix (one for ``rotate_graph``, one for the
# Wigner-``D⁽²⁾`` rotation of the output via ``cuet.Rotation``). Convention
# is Y-X-Y, matching ``cuet.Rotation(γ, β, α, x)``.
# ---------------------------------------------------------------------------


def _rotation_matrix_yxy(
    gamma: float,
    beta: float,
    alpha: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """3×3 rotation matrix for ``Rᵧ(α) Rₓ(β) Rᵧ(γ)`` (Y-X-Y intrinsic)."""
    cg, sg = math.cos(gamma), math.sin(gamma)
    cb, sb = math.cos(beta), math.sin(beta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    Ry_g = torch.tensor([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]], dtype=dtype)
    Rx_b = torch.tensor([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]], dtype=dtype)
    Ry_a = torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=dtype)
    return Ry_a @ Rx_b @ Ry_g


def _rotate_l2(features: torch.Tensor, gamma: float, beta: float, alpha: float) -> torch.Tensor:
    """Apply the Wigner ``D⁽²⁾(R)`` representation to a ``(N, 5)`` tensor.

    The 5 components are interpreted as a single ``1·2e`` channel in
    ``cue.ir_mul`` layout, which is the same convention
    :class:`PermMultipoleHead` writes its quadrupole output in.
    """
    out_irreps = cue.Irreps(cue.O3, [(1, "2e")])
    rot = cuet.Rotation(out_irreps, layout=cue.ir_mul).to(features.dtype)
    n = features.shape[0]
    g = torch.full((n,), gamma, dtype=features.dtype, device=features.device)
    b = torch.full((n,), beta, dtype=features.dtype, device=features.device)
    a = torch.full((n,), alpha, dtype=features.dtype, device=features.device)
    return rot(g, b, a, features)


# Three Euler triples covering pure-axis and generic rotations. Used by the
# l=2 quadrupole equivariance tests where we need both a 3×3 ``R`` for
# ``rotate_graph`` and the matching ``cuet.Rotation`` for the output.
_EULER_CASES = (
    (0.0, 0.0, math.pi / 2),
    (math.pi / 3, 0.4, -0.7),
    (1.234, 0.567, -0.891),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_molecule_neutral() -> TensorDict:
    """5-atom chain, two molecules (3+2), each with ``total_charge=0``."""
    torch.manual_seed(42)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.3, 0.0],
            [2.5, 0.0, 0.1],
            [4.0, 0.5, 0.0],
            [5.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    Z = torch.tensor([6, 1, 8, 6, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    total_charge = torch.tensor([0.0, 0.0], dtype=torch.float32)
    return make_graph_batch(
        pos=pos,
        Z=Z,
        edge_index=edge_index,
        batch=batch,
        graphs={"total_charge": total_charge},
    )


def _build_allegro(*, expose_tensor_track: bool) -> Allegro:
    torch.manual_seed(0)
    encoder = Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=8.0,
        num_layers=2,
        type_embed_dim=16,
        latent_mlp_depth=1,
        latent_mlp_width=16,
        avg_num_neighbors=4.0,
        expose_tensor_track=expose_tensor_track,
    )
    encoder.eval()
    return encoder


@pytest.fixture
def charge_only_pipeline():
    """Allegro (scalar-track only) → PermMultipoleHead(charge=True)."""
    encoder = _build_allegro(expose_tensor_track=False)
    head = PermMultipoleHead(
        input_dim=encoder.num_scalar_features * (encoder.num_layers + 1),
        avg_num_neighbors=4.0,
        charge=True,
        dipole=False,
        quadrupole=False,
        constrain_total_charge=True,
        hidden_dim=16,
    ).eval()
    return _PipelineModule(encoder, head)


@pytest.fixture
def charge_dipole_pipeline():
    """Allegro (tensor track exposed) → PermMultipoleHead(charge=True, dipole=True)."""
    encoder = _build_allegro(expose_tensor_track=True)
    head = PermMultipoleHead(
        input_dim=encoder.num_scalar_features * (encoder.num_layers + 1),
        avg_num_neighbors=4.0,
        charge=True,
        dipole=True,
        quadrupole=False,
        constrain_total_charge=True,
        hidden_dim=16,
        tensor_irreps=encoder.tensor_track_irreps,
    ).eval()
    return _PipelineModule(encoder, head)


@pytest.fixture
def full_multipole_pipeline():
    """Allegro (tensor track exposed) → PermMultipoleHead with q + μ + Θ all on."""
    encoder = _build_allegro(expose_tensor_track=True)
    head = PermMultipoleHead(
        input_dim=encoder.num_scalar_features * (encoder.num_layers + 1),
        avg_num_neighbors=4.0,
        charge=True,
        dipole=True,
        quadrupole=True,
        constrain_total_charge=True,
        hidden_dim=16,
        tensor_irreps=encoder.tensor_track_irreps,
    ).eval()
    return _PipelineModule(encoder, head)


class _PipelineModule(nn.Module):
    """Minimal Allegro→PermMultipoleHead pipeline.

    Re-derives ``bond_diff`` / ``bond_dist`` from ``pos`` inside ``forward``
    so that the encoder always sees fresh edge geometry under
    rotate / translate / permute transforms.
    """

    def __init__(self, encoder: Allegro, head: PermMultipoleHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        recompute_edge_geometry(batch)
        batch = self.encoder(batch)
        return self.head(batch)


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------


class TestTranslationInvariance:
    """Rigid shift of all atoms must not change the predicted moments.

    For neutral systems the total-charge projection enforces Σ q_proj = 0,
    so the molecular dipole's origin-dependent piece (Σ q_i)·t vanishes
    and ``μ_mol`` is also translation-invariant.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_charge_pipeline(self, charge_only_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = charge_only_pipeline(small_molecule_neutral.clone())
            shifted = charge_only_pipeline(translate_graph(small_molecule_neutral, t))

        # Float32 ULP — translation amplifies pos magnitudes by ~10×, so
        # bond_diff differs at ~1e-5; tolerances mirror the encoder tests.
        assert torch.allclose(
            ref["atomic_charges"], shifted["atomic_charges"], atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            ref["molecular_dipole"], shifted["molecular_dipole"], atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_charge_dipole_pipeline(self, charge_dipole_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = charge_dipole_pipeline(small_molecule_neutral.clone())
            shifted = charge_dipole_pipeline(translate_graph(small_molecule_neutral, t))

        assert torch.allclose(
            ref["atomic_charges"], shifted["atomic_charges"], atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            ref["atomic_dipoles"], shifted["atomic_dipoles"], atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            ref["molecular_dipole"], shifted["molecular_dipole"], atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_full_multipole_pipeline(self, full_multipole_pipeline, small_molecule_neutral, seed):
        """All four moment outputs (q, μ_atom, μ_mol, Θ_atom) under T."""
        torch.manual_seed(seed)
        t = torch.randn(3) * 10.0

        with torch.no_grad():
            ref = full_multipole_pipeline(small_molecule_neutral.clone())
            shifted = full_multipole_pipeline(translate_graph(small_molecule_neutral, t))

        for key in (
            "atomic_charges",
            "atomic_dipoles",
            "molecular_dipole",
            "atomic_quadrupoles",
        ):
            assert torch.allclose(ref[key], shifted[key], atol=1e-4, rtol=1e-4), key


# ---------------------------------------------------------------------------
# Rotation equivariance
# ---------------------------------------------------------------------------


class TestRotationEquivariance:
    """Scalars are SO(3)-invariant, vectors rotate with the same R."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_scalar_invariance(self, charge_only_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = charge_only_pipeline(small_molecule_neutral.clone())
            rot = charge_only_pipeline(rotate_graph(small_molecule_neutral, R))

        assert torch.allclose(ref["atomic_charges"], rot["atomic_charges"], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_molecular_dipole_equivariance(
        self, charge_only_pipeline, small_molecule_neutral, seed
    ):
        """``μ_mol(R·x) == R · μ_mol(x)``."""
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = charge_only_pipeline(small_molecule_neutral.clone())
            rot = charge_only_pipeline(rotate_graph(small_molecule_neutral, R))

        mu_rotated = rotate_vectors(ref["molecular_dipole"], R)
        assert torch.allclose(mu_rotated, rot["molecular_dipole"], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_atomic_dipole_equivariance(self, charge_dipole_pipeline, small_molecule_neutral, seed):
        """``μ_atom(R·x) == R · μ_atom(x)``."""
        torch.manual_seed(seed)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = charge_dipole_pipeline(small_molecule_neutral.clone())
            rot = charge_dipole_pipeline(rotate_graph(small_molecule_neutral, R))

        mu_rotated = rotate_vectors(ref["atomic_dipoles"], R)
        assert torch.allclose(mu_rotated, rot["atomic_dipoles"], atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("euler", _EULER_CASES)
    def test_atomic_quadrupole_equivariance(
        self, full_multipole_pipeline, small_molecule_neutral, euler
    ):
        """``Θ_atom(R·x) == D⁽²⁾(R) · Θ_atom(x)``.

        Parametrised on Euler angles (Y-X-Y, matching ``cuet.Rotation``).
        We need the matching 3×3 ``R`` for ``rotate_graph`` and the
        Wigner ``D⁽²⁾`` for the output rotation; both come from the same
        ``(γ, β, α)``.
        """
        gamma, beta, alpha = euler
        R = _rotation_matrix_yxy(gamma, beta, alpha)

        with torch.no_grad():
            ref = full_multipole_pipeline(small_molecule_neutral.clone())
            rot = full_multipole_pipeline(rotate_graph(small_molecule_neutral, R))

        theta_rotated = _rotate_l2(ref["atomic_quadrupoles"], gamma, beta, alpha)
        assert torch.allclose(theta_rotated, rot["atomic_quadrupoles"], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Permutation equivariance
# ---------------------------------------------------------------------------


class TestPermutationEquivariance:
    """Per-graph scalars / vectors are permutation-invariant. Per-atom outputs
    permute with the relabelling."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_per_graph_invariance(self, charge_only_pipeline, small_molecule_neutral, seed):
        torch.manual_seed(seed)
        n = small_molecule_neutral["atoms", "Z"].shape[0]
        perm = torch.randperm(n)

        with torch.no_grad():
            ref = charge_only_pipeline(small_molecule_neutral.clone())
            permuted = charge_only_pipeline(permute_graph(small_molecule_neutral, perm))

        assert torch.allclose(
            ref["molecular_dipole"], permuted["molecular_dipole"], atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_atomic_charges_equivariance(self, charge_only_pipeline, small_molecule_neutral, seed):
        """``q(perm(x))[i] == q(x)[perm[i]]``."""
        torch.manual_seed(seed)
        n = small_molecule_neutral["atoms", "Z"].shape[0]
        perm = torch.randperm(n)

        with torch.no_grad():
            ref = charge_only_pipeline(small_molecule_neutral.clone())
            permuted = charge_only_pipeline(permute_graph(small_molecule_neutral, perm))

        assert torch.allclose(
            ref["atomic_charges"][perm],
            permuted["atomic_charges"],
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_atomic_dipoles_equivariance(
        self, charge_dipole_pipeline, small_molecule_neutral, seed
    ):
        """``μ_atom(perm(x))[i] == μ_atom(x)[perm[i]]``."""
        torch.manual_seed(seed)
        n = small_molecule_neutral["atoms", "Z"].shape[0]
        perm = torch.randperm(n)

        with torch.no_grad():
            ref = charge_dipole_pipeline(small_molecule_neutral.clone())
            permuted = charge_dipole_pipeline(permute_graph(small_molecule_neutral, perm))

        assert torch.allclose(
            ref["atomic_dipoles"][perm],
            permuted["atomic_dipoles"],
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_atomic_quadrupoles_equivariance(
        self, full_multipole_pipeline, small_molecule_neutral, seed
    ):
        """``Θ_atom(perm(x))[i] == Θ_atom(x)[perm[i]]``."""
        torch.manual_seed(seed)
        n = small_molecule_neutral["atoms", "Z"].shape[0]
        perm = torch.randperm(n)

        with torch.no_grad():
            ref = full_multipole_pipeline(small_molecule_neutral.clone())
            permuted = full_multipole_pipeline(permute_graph(small_molecule_neutral, perm))

        assert torch.allclose(
            ref["atomic_quadrupoles"][perm],
            permuted["atomic_quadrupoles"],
            atol=1e-5,
            rtol=1e-5,
        )
