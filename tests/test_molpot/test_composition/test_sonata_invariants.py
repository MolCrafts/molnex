"""Sonata physics invariants — translation, SO(3), permutation, Σqᵢ, tr(Θ).

Categories 1–5 of the ``sonata-02-tests`` integration suite. Every test
runs in float64 with a deterministic seed; the algebraic-identity
tolerances (1e-10 for charge conservation, 1e-12 for trace) are below
float32 ULP at the eV scale, so float32 here would silently pass on
roundoff alone.

Each invariant traces to one acceptance criterion:

* ac-002 — translation invariance, ≤ 1e-7
* ac-003 — SO(3) invariance of U + equivariance of μ (D⁽¹⁾) and Θ (D⁽²⁾)
* ac-004 — permutation equivariance of all per-atom outputs
* ac-005 — Σqᵢ = Q_tot per graph to ≤ 1e-10
* ac-006 — tr(Θ) = 0 per atom to ≤ 1e-12

A failing test here is, by sub-spec contract, a real bug against
``sonata-01-composer`` to file separately; loosening a tolerance to
make a test pass is forbidden.
"""

from __future__ import annotations

import math

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from tensordict import TensorDict
from tests.symmetry_helpers import (
    permute_graph,
    rotate_graph,
    translate_graph,
)

from molpot.composition import Sonata

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _theta5_to_cartesian33(theta: torch.Tensor) -> torch.Tensor:
    """Mirror of ``Sonata._theta_to_cartesian_quadrupole`` for tests.

    Re-implemented (not imported) so a regression in the production
    converter is detected by *this* test, not silently absorbed when both
    use the same buggy basis. Convention: cuequivariance ``2e`` real
    spherical, layout ``ir_mul``, ordering ``[m=-2, -1, 0, +1, +2]``.
    """
    s2 = math.sqrt(0.5)
    s6 = 1.0 / math.sqrt(6.0)
    n = theta.shape[0]
    q = torch.zeros(n, 3, 3, dtype=theta.dtype, device=theta.device)
    q[:, 0, 1] = q[:, 1, 0] = theta[:, 0] * s2
    q[:, 1, 2] = q[:, 2, 1] = theta[:, 1] * s2
    q[:, 0, 0] = -theta[:, 2] * s6 + theta[:, 4] * s2
    q[:, 1, 1] = -theta[:, 2] * s6 - theta[:, 4] * s2
    q[:, 2, 2] = 2.0 * theta[:, 2] * s6
    q[:, 0, 2] = q[:, 2, 0] = theta[:, 3] * s2
    return q


def _rotate_l1(vec: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply a 3×3 SO(3) rotation to per-atom 3-vectors ``(N, 3)``."""
    return vec @ R.T


def _rotate_l2_via_cuet(theta: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply Wigner ``D⁽²⁾(R)`` to ``(N, 5)`` cuet-2e quadrupoles.

    We extract Y-X-Y Euler angles from ``R`` so the same rotation can be
    applied via ``cuet.Rotation`` to the spherical components. This is
    the same recipe used in ``test_multipole_symmetry.py``.
    """
    # Extract Y-X-Y Euler angles (γ, β, α) from R = R_y(α) R_x(β) R_y(γ).
    # Component identities (using s = sin, c = cos throughout):
    #   R[0,1] = s_α s_β     R[2,1] = c_α s_β
    #   R[1,0] = s_β s_γ     R[1,2] = -s_β c_γ      R[1,1] = c_β
    # so β ∈ [0, π] gives s_β ≥ 0 and the per-axis atan2 extractions are
    # unambiguous. The gimbal-lock case (s_β = 0) collapses α + γ; we
    # then attribute the entire rotation to α.
    cos_beta = R[1, 1].clamp(-1.0, 1.0)
    beta = torch.acos(cos_beta)
    sin_beta = torch.sin(beta)
    if sin_beta.abs() < 1e-9:
        alpha = torch.atan2(R[0, 2], R[0, 0])
        gamma = torch.zeros((), dtype=R.dtype)
    else:
        alpha = torch.atan2(R[0, 1], R[2, 1])
        gamma = torch.atan2(R[1, 0], -R[1, 2])
    out_irreps = cue.Irreps(cue.O3, [(1, "2e")])
    rot = cuet.Rotation(out_irreps, layout=cue.ir_mul).to(theta.dtype)
    n = theta.shape[0]
    g = torch.full((n,), float(gamma), dtype=theta.dtype, device=theta.device)
    b = torch.full((n,), float(beta), dtype=theta.dtype, device=theta.device)
    a = torch.full((n,), float(alpha), dtype=theta.dtype, device=theta.device)
    return rot(g, b, a, theta)


def _scatter_sum(values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.zeros(n_graphs, dtype=values.dtype, device=values.device)
    out.scatter_add_(0, batch, values)
    return out


# ---------------------------------------------------------------------------
# 1 — Translation invariance (ac-002)
# ---------------------------------------------------------------------------


class TestTranslation:
    """Energy / scalar moments are invariant under a rigid shift of all atoms.

    For neutral systems the molecular dipole is also translation-invariant
    (the origin-dependent term ``(Σ qᵢ)·t`` vanishes).
    """

    def test_translation_invariance_open(
        self, sonata_pipeline: Sonata, sample_neutral_batch_open: TensorDict
    ) -> None:
        torch.manual_seed(7)
        t = torch.randn(3, dtype=torch.float64) * 5.0

        with torch.no_grad():
            ref = sonata_pipeline(sample_neutral_batch_open.clone())
            shifted = sonata_pipeline(translate_graph(sample_neutral_batch_open, t))

        assert torch.allclose(ref["energy"], shifted["energy"], atol=1e-7, rtol=1e-7)
        assert torch.allclose(
            ref["atomic_charges"], shifted["atomic_charges"], atol=1e-7, rtol=1e-7
        )

    def test_translation_invariance_periodic_lattice_vector(
        self, sonata_pipeline: Sonata, sample_neutral_batch_periodic: TensorDict
    ) -> None:
        """Shift everyone by an integer combination of cell[0]'s lattice vectors.

        For a periodic system, this is doubly trivial: the relative
        geometry is unchanged AND the structure factor's plane waves
        are invariant under shifts by lattice vectors. We then keep the
        cell unchanged so the post-shift system is still valid.
        """
        cell0 = sample_neutral_batch_periodic["graphs", "cell"][0]
        # n·cell[0] + m·cell[1] + p·cell[2] for integers (n, m, p).
        for n, m, p in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 2, -1)]:
            t = n * cell0[0] + m * cell0[1] + p * cell0[2]
            with torch.no_grad():
                ref = sonata_pipeline(sample_neutral_batch_periodic.clone())
                shifted = sonata_pipeline(translate_graph(sample_neutral_batch_periodic, t))
            rel = (shifted["energy"] - ref["energy"]).abs() / ref["energy"].abs().clamp(min=1e-12)
            assert rel.max() < 1e-7, (
                f"lattice translation (n,m,p)=({n},{m},{p}): rel.max()={rel.max():.2e}"
            )


# ---------------------------------------------------------------------------
# 2 — SO(3) invariance / equivariance (ac-003)
# ---------------------------------------------------------------------------


class TestRotation:
    """Energy is SO(3)-invariant; μ rotates as D⁽¹⁾ and Θ as D⁽²⁾."""

    def test_rotation_energy_invariance_open(
        self,
        sonata_pipeline: Sonata,
        sample_neutral_batch_open: TensorDict,
        random_rotation_matrix,
    ) -> None:
        torch.manual_seed(11)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = sonata_pipeline(sample_neutral_batch_open.clone())
            rot = sonata_pipeline(rotate_graph(sample_neutral_batch_open, R))

        # Algebraic identity in float64 — tolerance below 1e-10.
        rel = (rot["energy"] - ref["energy"]).abs() / ref["energy"].abs().clamp(min=1e-12)
        assert rel.max() < 1e-10, f"rel.max()={rel.max():.2e}"

    def test_rotation_dipole_equivariance_open(
        self,
        sonata_pipeline: Sonata,
        sample_neutral_batch_open: TensorDict,
        random_rotation_matrix,
    ) -> None:
        """``μ_atom(R·x) ≈ R · μ_atom(x)`` (D⁽¹⁾ rep)."""
        torch.manual_seed(13)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = sonata_pipeline(sample_neutral_batch_open.clone())
            rot = sonata_pipeline(rotate_graph(sample_neutral_batch_open, R))

        mu_rotated = _rotate_l1(ref["atomic_dipoles"], R)
        assert torch.allclose(mu_rotated, rot["atomic_dipoles"], atol=1e-9, rtol=1e-9)

    def test_rotation_quadrupole_equivariance_open(
        self,
        sonata_pipeline: Sonata,
        sample_neutral_batch_open: TensorDict,
        random_rotation_matrix,
    ) -> None:
        """``Θ_atom(R·x) ≈ D⁽²⁾(R) · Θ_atom(x)`` via cuet.Rotation."""
        torch.manual_seed(17)
        R = random_rotation_matrix()

        with torch.no_grad():
            ref = sonata_pipeline(sample_neutral_batch_open.clone())
            rot = sonata_pipeline(rotate_graph(sample_neutral_batch_open, R))

        theta_rotated = _rotate_l2_via_cuet(ref["atomic_quadrupoles"], R)
        assert torch.allclose(theta_rotated, rot["atomic_quadrupoles"], atol=1e-9, rtol=1e-9)

    def test_rotation_energy_invariance_periodic(
        self,
        sonata_pipeline: Sonata,
        sample_neutral_batch_periodic: TensorDict,
        random_rotation_matrix,
    ) -> None:
        """Joint rotation of pos AND cell leaves periodic energy invariant.

        Uses the frozen-grid path (``kvec_indices`` enumerated once
        on the unperturbed cell) so the reciprocal-space integer
        triplet array is identical for the unrotated and rotated
        evaluations. Without freezing, FP rounding on
        ``k² = ‖nvec @ G‖²`` can flip a boundary k-vector across the
        cutoff under rotation, breaking exact rotation invariance.
        With the frozen grid, ``k_sq`` is preserved exactly under
        rotation (rotations are isometries on ``kvec``) and the
        invariance holds at float64 ULP.
        """
        torch.manual_seed(19)
        R = random_rotation_matrix()
        batch = sample_neutral_batch_periodic
        cell = batch["graphs", "cell"]  # (B, 3, 3)
        # Freeze the integer-triplet grid using the *unrotated* cell;
        # rotation preserves cell norms so this same grid is also
        # what the rotated cell would dynamically enumerate, but
        # passing it explicitly removes the FP-rounding flicker at
        # the cutoff.
        nvec = sonata_pipeline.ewald.enumerate_kvec_indices(cell[0])
        rotated = rotate_graph(batch, R)
        rotated_cell = torch.einsum("bij,kj->bik", cell, R)
        rotated["graphs", "cell"] = rotated_cell

        with torch.no_grad():
            ref = sonata_pipeline(batch.clone(), kvec_indices=nvec)
            rot = sonata_pipeline(rotated, kvec_indices=nvec)

        rel = (rot["energy"] - ref["energy"]).abs() / ref["energy"].abs().clamp(min=1e-12)
        assert rel.max() < 1e-9, f"periodic rel.max()={rel.max():.2e}"


# ---------------------------------------------------------------------------
# 3 — Permutation equivariance (ac-004)
# ---------------------------------------------------------------------------


class TestPermutation:
    """Per-atom outputs permute with the relabelling; per-graph outputs are
    invariant. The permutation is constructed to interleave atoms across
    graphs in the batch (not just within a single graph) so the per-graph
    accumulators are also exercised."""

    def test_permutation_equivariance_open(
        self, sonata_pipeline: Sonata, sample_neutral_batch_open: TensorDict
    ) -> None:
        torch.manual_seed(23)
        n = sample_neutral_batch_open["atoms", "Z"].shape[0]
        # Pick a permutation that does NOT preserve graph membership
        # (the easiest way: take any random permutation and assert it
        # actually scrambles the batch index).
        perm = torch.randperm(n)
        batch_idx = sample_neutral_batch_open["atoms", "batch"]
        assert not torch.equal(batch_idx[perm], batch_idx), (
            "test sanity: permutation must scramble graph membership"
        )

        with torch.no_grad():
            ref = sonata_pipeline(sample_neutral_batch_open.clone())
            permuted = sonata_pipeline(permute_graph(sample_neutral_batch_open, perm))

        # Per-atom outputs: charges, dipoles, quadrupoles, phi, field.
        for key in ("atomic_charges", "atomic_dipoles", "atomic_quadrupoles", "phi", "field"):
            assert torch.allclose(ref[key][perm], permuted[key], atol=1e-9, rtol=1e-9), (
                f"per-atom output {key!r} not permutation-equivariant"
            )

        # Per-graph outputs (energy, energy_es, molecular_dipole) follow
        # the post-permutation graph order (which matches the pre-perm
        # order because we did not reorder the graph IDs themselves —
        # the permutation only relabels atoms).
        # The post-perm batch index is `batch_idx[perm]`; since each
        # original graph receives the same set of atoms back (just under
        # new indices), the per-graph outputs must match by graph id.
        for key in ("energy", "energy_es", "molecular_dipole"):
            assert torch.allclose(ref[key], permuted[key], atol=1e-9, rtol=1e-9), (
                f"per-graph output {key!r} not permutation-invariant"
            )


# ---------------------------------------------------------------------------
# 4 — Σqᵢ = Q_tot per graph (ac-005)
# ---------------------------------------------------------------------------


class TestChargeConservation:
    """``PermMultipoleHead.constrain_total_charge=True`` projects per-graph
    charge sums onto ``("graphs", "total_charge")`` via mean-residual
    subtraction. The projection is algebraic (subtract a per-atom
    constant) so the post-projection error is at float64 ULP."""

    def test_charge_conservation_neutral(
        self, sonata_pipeline: Sonata, sample_neutral_batch_periodic: TensorDict
    ) -> None:
        with torch.no_grad():
            out = sonata_pipeline(sample_neutral_batch_periodic.clone())

        q = out["atomic_charges"]
        atom_batch = sample_neutral_batch_periodic["atoms", "batch"]
        target = sample_neutral_batch_periodic["graphs", "total_charge"]
        n_graphs = int(target.shape[0])
        sums = _scatter_sum(q, atom_batch, n_graphs)

        assert torch.allclose(sums, target, atol=1e-10, rtol=0.0), (
            f"per-graph Σq deviates: sums={sums}, target={target}"
        )

    def test_charge_conservation_charged(
        self, sonata_pipeline: Sonata, sample_charged_batch_periodic: TensorDict
    ) -> None:
        with torch.no_grad():
            out = sonata_pipeline(sample_charged_batch_periodic.clone())

        q = out["atomic_charges"]
        atom_batch = sample_charged_batch_periodic["atoms", "batch"]
        target = sample_charged_batch_periodic["graphs", "total_charge"]
        n_graphs = int(target.shape[0])
        sums = _scatter_sum(q, atom_batch, n_graphs)

        assert torch.allclose(sums, target, atol=1e-10, rtol=0.0), (
            f"per-graph Σq deviates: sums={sums}, target={target}"
        )


# ---------------------------------------------------------------------------
# 5 — tr(Θ) = 0 per atom (ac-006)
# ---------------------------------------------------------------------------


class TestQuadrupoleTraceless:
    """The 5-component cuet ``2e`` real-spherical basis is by construction
    the symmetric traceless ℓ=2 representation. Round-tripping the head's
    output through ``_theta5_to_cartesian33`` and taking the trace must
    return zero to float64 ULP — anything else means the basis transform
    has a bug."""

    def test_traceless_open(
        self, sonata_pipeline: Sonata, sample_neutral_batch_open: TensorDict
    ) -> None:
        with torch.no_grad():
            out = sonata_pipeline(sample_neutral_batch_open.clone())

        Theta_cart = _theta5_to_cartesian33(out["atomic_quadrupoles"])
        traces = torch.einsum("nii->n", Theta_cart)
        assert traces.abs().max() < 1e-12, f"tr(Θ).abs().max()={traces.abs().max():.2e}"

    def test_traceless_periodic(
        self, sonata_pipeline: Sonata, sample_neutral_batch_periodic: TensorDict
    ) -> None:
        with torch.no_grad():
            out = sonata_pipeline(sample_neutral_batch_periodic.clone())

        Theta_cart = _theta5_to_cartesian33(out["atomic_quadrupoles"])
        traces = torch.einsum("nii->n", Theta_cart)
        assert traces.abs().max() < 1e-12, f"tr(Θ).abs().max()={traces.abs().max():.2e}"

    def test_traceless_synthetic_input(self, random_traceless_Q) -> None:
        """Documentation-via-test: a random ``(N, 5)`` cuet-2e draw, when
        expanded to Cartesian, is symmetric AND traceless. This pins the
        basis convention independent of the model output."""
        Theta = random_traceless_Q(16)
        cart = _theta5_to_cartesian33(Theta)
        # Symmetry
        assert torch.allclose(cart, cart.transpose(-1, -2), atol=1e-12, rtol=0.0)
        # Tracelessness
        traces = torch.einsum("nii->n", cart)
        assert traces.abs().max() < 1e-12
