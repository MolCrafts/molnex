"""Tests for GTOHartreeEnergy (isolated real-space and periodic k-space)."""

from __future__ import annotations

import math

import pytest
import torch

from molix import config
from molix.compile import Compiler
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.hartree import (
    GTOHartreeEnergy,
    complete_directed_pairs,
    gto_self_overlap,
)
from molpot.potentials.gto.kspace import KSpaceGrid

#: Madelung (Wigner) constant of the simple-cubic lattice with a neutralising
#: background: ``E_self = -½ ξ q² / L`` in Gaussian units. Hard-coded golden
#: from the literature value ξ = 2.837297 (Wigner 1934), not a live oracle.
CUBIC_MADELUNG = 2.837297

#: Three atoms within ~3 Å of each other (Å), placed by each test.
PROBE_POS: list[list[float]] = [
    [0.0, 0.0, 0.0],
    [1.4, 0.3, -0.2],
    [-0.9, 1.6, 0.7],
]

#: Matching ``ir_mul`` multipoles ``(q, μ_y, μ_z, μ_x)`` in e / e·Å. The
#: monopoles sum to zero, so the reciprocal sum needs no charged-cell caveat.
PROBE_FEATS: list[list[float]] = [
    [0.4, 0.20, -0.10, 0.05],
    [-0.3, -0.15, 0.22, -0.08],
    [-0.1, 0.05, 0.12, 0.18],
]

#: Polar dump reciprocal cutoff (Å⁻¹) at ``σ = 1.5 Å``, ``l_max = 1``,
#: ``κ = 1.0`` — the official checkpoint's ``kspace_cutoff`` buffer.
K_CUT = 1.8467166200173746

# -- goldens ---------------------------------------------------------
#
# Two unit charges 1 Å apart in a 10 Å cubic cell, ``σ = 1.5 Å``, ``l_max = 1``
# (zero dipoles). Hard-coded fp64 literals captured from *this* tree before the
# fixed-capacity k-grid rewrite (molnex @ ``feat/oplsaa-parameterizer``,
# ``PYTHONPATH=src python capture_kgrid_goldens.py``, 2026-09-03). Strain
# derivative from ``pos @ (I+ε)ᵀ`` / ``cell @ (I+ε)ᵀ``. No live oracle.

#: ``include_self_interaction=False`` energy in eV.
AC008_ENERGY_NOSI = -2.1055022353716124

#: ``include_self_interaction=True`` energy in eV.
AC008_ENERGY_SI = 3.3105844216987386

#: ``∂E/∂pos`` (2, 3) in eV/Å, ``include_self_interaction=False``.
AC008_DPOS_NOSI: list[list[float]] = [
    [0.3128153444964179, -5.963111948670274e-19, 2.168404344971009e-18],
    [-0.3128153444964179, -7.047314121155779e-19, -2.168404344971009e-18],
]

#: ``∂E/∂ε`` (3, 3) in eV, ``include_self_interaction=False``.
AC008_DEPS_NOSI: list[list[float]] = [
    [1.470186763029792, 2.874766613593052e-17, 1.4169465773160073e-17],
    [-9.416870756969064e-17, 1.9180416642629678, -2.0709219206926254e-17],
    [3.0530362823859914e-17, -2.0709219206926254e-17, 1.9180416642629678],
]

#: ``∂E/∂pos`` (2, 3) in eV/Å, ``include_self_interaction=True``.

#: ``∂E/∂ε`` (3, 3) in eV, ``include_self_interaction=True``.

# -- PROBE geometry goldens (same capture) ----------------------------------

#: Periodic energy (eV) of :data:`PROBE_POS` in a 10 Å cubic cell.
PROBE_PERIODIC10_NOSI = -0.6989869329101701
PROBE_PERIODIC10_SI = 0.041031040667996545

#: Periodic energy (eV) of ``PROBE_POS + 10`` in a 20 Å cubic cell.
PROBE_PERIODIC20_NOSI = -0.6972598758290532
PROBE_PERIODIC20_SI = 0.04275809774911341

#: Isolated ``l_max=1`` energy (eV) of ``PROBE_POS + 10``.
PROBE_ISOLATED_NOSI = -0.6968842461149762
PROBE_ISOLATED_SI = 0.04313372746319044

#: Isolated ``l_max=0`` energy (eV) of ``PROBE_POS + 10`` (monopoles only).
PROBE_ISOLATED_L0_NOSI = -0.6665851178517852
PROBE_ISOLATED_L0_SI = 0.03750614756736037


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestCompleteDirectedPairs:
    """The all-pairs list is a pure function of the row count."""

    @pytest.mark.parametrize("n", [1, 4])
    def test_enumerates_every_ordered_pair_once(self, n: int) -> None:
        """``n (n - 1)`` rows, each ordered pair ``i != j`` exactly once; ``n = 1`` is empty."""
        src, dst = complete_directed_pairs(n, torch.device("cpu"))
        assert src.shape == dst.shape == (n * (n - 1),)
        assert src.dtype == dst.dtype == torch.long
        pairs = list(zip(src.tolist(), dst.tolist(), strict=True))
        assert len(set(pairs)) == len(pairs)
        assert set(pairs) == {(i, j) for i in range(n) for j in range(n) if i != j}


class TestGTOHartreeEnergy:
    """l=0 pair Hartree ``k_e q_i q_j erf(r/(2σ))/r``, plus the ``cell`` branch."""

    def test_two_unit_charges_one_angstrom_apart(self):
        """σ=1.5 Å, r=1 Å, q=1,1 → eV_A * erf(1/3)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.ones(2, 1, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = energy(q, pos, batch, num_graphs=1)
        expect = eV_A * math.erf(1.0 / 3.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(expect, rel=1e-10)

    def test_self_pairs_are_excluded(self):
        """A single charge has no pair energy (self-interaction is a separate term)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=False)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        assert energy(q, pos, batch, num_graphs=1).item() == pytest.approx(0.0)

    def test_unit_monopole_self_energy(self):
        """Dump σ=1.5 Å: ½ q² eV_A/(σ√π) for a lone unit charge."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=True)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        expect = 0.5 * eV_A / (1.5 * math.sqrt(math.pi))
        assert energy(q, pos, batch, num_graphs=1).item() == pytest.approx(expect, rel=1e-8)

    def test_self_overlap_polar_dump_sigma(self):
        """Polar-1-M dump overlap_constants at σ=1.5 Å, l_max=1."""
        omega = gto_self_overlap(1.5, 1)
        assert omega.shape == (4,)
        assert omega[0].item() == pytest.approx(eV_A / (1.5 * math.sqrt(math.pi)), rel=1e-10)
        assert omega[1].item() == pytest.approx(0.401191604, rel=1e-6)
        assert torch.allclose(omega[1:], omega[1:2].expand(3))

    def test_zero_dipole_l1_matches_l0(self):
        """Finite-difference copies of a pure monopole must not add extra pairs."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        q0 = torch.ones(2, 1, dtype=torch.float64)
        q1 = torch.zeros(2, 4, dtype=torch.float64)
        q1[:, 0] = 1.0
        e0 = GTOHartreeEnergy(sigma=1.5, l_max=0)(q0, pos, batch, num_graphs=1)
        e1 = GTOHartreeEnergy(sigma=1.5, l_max=1)(q1, pos, batch, num_graphs=1)
        assert torch.allclose(e0, e1, rtol=1e-10, atol=1e-12)

    def test_two_graph_batch_matches_explicit_pair_loop(self):
        """``l_max=0``: the batch energy is the loop over same-graph pairs ``i ≠ j``.

        Graph 1 starts at the origin like graph 0, so the masked cross-graph
        pair ``(0, 3)`` sits at zero distance — evaluated, zeroed, harmless.
        """
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor(PROBE_POS + [[0.0, 0.0, 0.0], [1.1, -0.4, 0.8]], dtype=torch.float64)
        q = torch.tensor([0.4, -0.3, -0.1, 0.7, -0.7], dtype=torch.float64)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        out = energy(q.unsqueeze(-1), pos, batch, num_graphs=2)
        expect = torch.zeros(2, dtype=torch.float64)
        for i in range(5):
            for j in range(5):
                if i == j or batch[i] != batch[j]:
                    continue
                r = (pos[i] - pos[j]).norm()
                expect[batch[i]] += 0.5 * eV_A * q[i] * q[j] * torch.erf(r / 3.0) / r
        assert out.shape == (2,)
        assert torch.allclose(out, expect, rtol=1e-12, atol=0.0)

    def test_copy_layout_matches_explicit_pair_loop(self):
        """``orig`` masks the pairs of one atom's copies; every other same-graph pair is summed.

        Three atoms in two graphs, four copies each; copies ``0`` and ``3`` of
        an atom coincide, so masked zero-distance pairs occur inside a graph
        as well as across graphs.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64).repeat_interleave(4, dim=0)
        pos[1::4, 0] += 0.3
        pos[2::4, 1] -= 0.2
        orig = torch.arange(3).repeat_interleave(4)
        batch = torch.tensor([0, 0, 1], dtype=torch.long).repeat_interleave(4)
        q = torch.linspace(-0.6, 0.6, 12, dtype=torch.float64)
        out = model._pair_energy(q, pos, batch, 2, orig)
        expect = torch.zeros(2, dtype=torch.float64)
        for i in range(12):
            for j in range(12):
                if i == j or batch[i] != batch[j] or orig[i] == orig[j]:
                    continue
                r = (pos[i] - pos[j]).norm()
                expect[batch[i]] += 0.5 * eV_A * q[i] * q[j] * torch.erf(r / 3.0) / r
        assert out.shape == (2,)
        assert torch.allclose(out, expect, rtol=1e-12, atol=0.0)

    def test_coincident_graphs_keep_energy_and_force_finite(self):
        """Two graphs at the same coordinates: the masked zero-distance pairs are inert.

        Every cross-graph pair ``(i, i')`` has ``r = 0``. The batch energy and
        its position gradient must equal those of the graph alone — and in
        particular carry no NaN from ``0 · inf`` in either direction.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        leaf = torch.cat([pos, pos]).requires_grad_(True)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        energy = model(torch.cat([feats, feats]), leaf, batch, num_graphs=2)
        (grad,) = torch.autograd.grad(energy.sum(), leaf)
        single = pos.clone().requires_grad_(True)
        alone = model(feats, single, torch.zeros(3, dtype=torch.long), num_graphs=1)
        (grad_alone,) = torch.autograd.grad(alone.sum(), single)
        assert bool(torch.isfinite(grad).all())
        assert torch.allclose(energy, torch.cat([alone, alone]), rtol=1e-12, atol=0.0)
        assert torch.allclose(grad, torch.cat([grad_alone, grad_alone]), rtol=1e-12, atol=0.0)

    @pytest.mark.slow
    def test_isolated_branch_compiles_to_one_graph(self) -> None:
        """A pair list that is a function of ``N`` alone → 1 graph, 0 breaks, two graphs in."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS + PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS + PROBE_POS, dtype=torch.float64)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        explanation = Compiler.explain(model, feats, pos, batch, num_graphs=2)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )

    def test_large_cell_kspace_matches_isolated(self):
        """A 60 Å cell reproduces the isolated energy — images are all that differ.

        Both branches are asked for the *same* physical quantity with
        ``include_self_interaction=True``: the pair sum plus the on-site GTO
        overlap. The reciprocal sum already runs over ``i = j``, so the k-space
        branch must return its raw value; adding ``_self_energy`` on top would
        double-count ~0.74 eV here, 17× the total energy. The residual is the
        periodic images (~2e-5 eV at L = 60 Å), well inside ``abs=1e-2``.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64) + 30.0
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 60.0
        periodic = model(feats, pos, batch, cell=cell, num_graphs=1)
        isolated = model(feats, pos, batch, num_graphs=1)
        assert periodic.shape == (1,)
        assert periodic.item() == pytest.approx(isolated.item(), abs=1e-2)

    def test_kspace_lone_charge_is_self_plus_madelung(self):
        """One unit monopole in a 40 Å cube: on-site term + Wigner background.

        The reciprocal sum already contains the on-site ``i = j`` term
        ``½ eV_A q²/(σ√π)``; the second term is the cubic Madelung / Wigner
        energy ``−½ ξ q² eV_A / L`` of the neutralising background that
        dropping ``k = 0`` implies. Truncating the k-sum at
        ``κ · 2.25/σ · 2^0.3`` leaves ~1e-3 relative.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        feats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 40.0
        expect = 0.5 * eV_A / (1.5 * math.sqrt(math.pi)) - 0.5 * CUBIC_MADELUNG * eV_A / 40.0
        assert model(feats, pos, batch, cell=cell, num_graphs=1).item() == pytest.approx(
            expect, rel=3e-3
        )

    def test_kspace_self_flag_subtracts_not_adds(self):
        """Periodic branch: ``True`` is the raw sum, ``False`` removes the on-site term.

        Opposite sense to the isolated branch, where ``True`` *adds* it — the
        reciprocal sum starts with ``i = j`` included (SI V.6).
        """
        pos = torch.tensor(PROBE_POS, dtype=torch.float64) + 10.0
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        without = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        e_true = model(feats, pos, batch, cell=cell, num_graphs=1)
        e_false = without(feats, pos, batch, cell=cell, num_graphs=1)
        overlap = model.self_overlap
        on_site = 0.5 * (overlap * feats * feats).sum()
        assert on_site.item() > 0.1  # not a vacuous difference
        assert (e_true - e_false).sum().item() == pytest.approx(on_site.item(), rel=1e-10)
        raw = model._kspace_energy(feats, pos, batch, cell, num_graphs=1)
        assert e_true.sum().item() == pytest.approx(raw.sum().item(), rel=1e-12)


class TestGTOKspaceHartree:
    """Periodic branch of :class:`GTOHartreeEnergy` (Polar SI V.6 direct sum).

    Migrated from ``test_kspace.py`` by ``polar-compile-clean-01-kgrid``: the
    free functions these used to call (``hartree_energy``) are folded into
    their single caller, so the cases now go through the public ``forward``.
    ``include_self_interaction=True`` is the raw reciprocal sum, i.e. the exact
    quantity the retired free function returned.
    """

    def test_two_charges_energy_is_finite(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        out = energy(q, pos, batch, cell=cell, num_graphs=1)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_forces_obey_newtons_third_law(self):
        """Opposite charges in a cube: F_1 = −F_0, attraction along the bond."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 14.0
        leaf = pos.detach().requires_grad_(True)
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)(q, leaf, batch, cell=cell, num_graphs=1).sum()
        (d_energy,) = torch.autograd.grad(energy, leaf)
        assert torch.allclose(d_energy.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)
        # Attraction: moving atom 0 toward atom 1 lowers E (∂E/∂x_0 < 0).
        assert d_energy[0, 0].item() < 0.0
        assert d_energy[1, 0].item() > 0.0

    def test_origin_translation_leaves_energy(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[1.0, 2.0, 3.0], [2.4, 2.0, 3.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 14.0
        e0 = energy(q, pos, batch, cell=cell, num_graphs=1)
        e1 = energy(q, pos + 1.7, batch, cell=cell, num_graphs=1)
        assert torch.allclose(e0, e1, atol=1e-9, rtol=0.0)

    def test_zero_dipole_matches_monopole(self):
        """A zero dipole column contributes nothing to the structure factor.

        Both calls go through the same ``l_max=1`` module, so both use the same
        ``k_cutoff``; only the feature width differs, which is what selects the
        analytic dipole term.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.3, -0.2]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 16.0
        monopole = torch.tensor([[0.4], [-0.4]], dtype=torch.float64)
        with_dipole = torch.zeros(2, 4, dtype=torch.float64)
        with_dipole[:, 0] = monopole[:, 0]
        e0 = model(monopole, pos, batch, cell=cell, num_graphs=1)
        e1 = model(with_dipole, pos, batch, cell=cell, num_graphs=1)
        assert torch.allclose(e0, e1, atol=1e-10, rtol=0.0)

    def test_half_sphere_matches_full_sphere(self):
        """Conjugate-symmetry factor 2 reproduces the full k ≠ 0 sum."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=True)
        pos = torch.tensor([[0.1, 0.2, 0.0], [1.0, 0.2, 0.4]], dtype=torch.float64)
        q = torch.tensor([[0.7], [-0.7]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 11.0
        sigma = 1.5
        k_cut = model.k_cutoff
        half = model(q, pos, batch, cell=cell, num_graphs=1)
        g = (2.0 * math.pi) * torch.linalg.inv(cell).T
        nk = max(1, int(11.0 * k_cut / (2.0 * math.pi)) + 1)
        n = torch.arange(-nk, nk + 1)
        nvec = torch.stack(torch.meshgrid(n, n, n, indexing="ij"), dim=-1).reshape(-1, 3)
        kvec = nvec.to(cell.dtype) @ g
        k_sq = (kvec * kvec).sum(dim=1)
        keep = (k_sq > 0.0) & (k_sq < k_cut * k_cut)
        kvec, k_sq = kvec[keep], k_sq[keep]
        kr = pos @ kvec.T
        s_real = q[:, 0] @ torch.cos(kr)
        s_imag = q[:, 0] @ torch.sin(kr)
        s_sq = s_real * s_real + s_imag * s_imag
        kfac = torch.exp(-(sigma * sigma) * k_sq) / k_sq
        full = eV_A * (2.0 * math.pi) / torch.det(cell).abs() * (kfac * s_sq).sum()
        assert half.sum().item() == pytest.approx(full.item(), rel=1e-10)

    def test_cell_gradient_is_nonzero(self):
        """Hartree depends on the cell, so strain autograd yields a virial."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=True)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = (torch.eye(3, dtype=torch.float64) * 13.0).requires_grad_(True)
        energy = model(q, pos, batch, cell=cell, num_graphs=1)
        (grad,) = torch.autograd.grad(energy.sum(), cell)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0.0

    def test_batched_graphs_are_rejected(self):
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.zeros(4, 3, dtype=torch.float64)
        q = torch.ones(4, 1, dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        with pytest.raises(ValueError, match="single-system"):
            energy(q, pos, batch, cell=cell, num_graphs=2)


class TestGTOHartreeEnergyKspace:
    """Fixed-capacity k-grid contract: ``bind_cell``, ``num_graphs``, 1 graph."""

    def test_bind_cell_freezes_the_cutoff_on_the_grid(self) -> None:
        """``k_cutoff`` is computed once, in ``__init__``, and handed to the grid."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1)
        assert model.k_cutoff == K_CUT
        assert model.kgrid.k_cutoff == K_CUT
        model.bind_cell(torch.eye(3, dtype=torch.float64) * 10.0)
        assert model.kgrid.capacity == (4, 4, 4)

    def test_k_cutoff_reads_the_grid_rather_than_recomputing_it(self) -> None:
        """One home for the cutoff: the property delegates to :attr:`kgrid`.

        Swapping the grid is the documented way to trade head-room for
        k-points, so a property that re-derives ``kspace_cutoff(sigma, l_max,
        κ)`` would report a cutoff the sum no longer uses.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1)
        model.kgrid = KSpaceGrid(k_cutoff=0.5)
        assert model.k_cutoff == 0.5

    def test_capacity_factor_keyword_reaches_the_grid(self) -> None:
        """``kspace_capacity_factor`` is forwarded verbatim to the grid."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, kspace_capacity_factor=1.0)
        assert model.kgrid.capacity_factor == 1.0

    def test_capacity_factor_does_not_change_the_periodic_energy(self) -> None:
        """D3: the extra half-box rows carry weight ``0`` and add exactly ``+0.0``.

        Head-room is an allocation decision, never a physics one. Padding rows
        contribute exactly ``+0.0`` per term, but the two grids sum different
        row counts (171 against 364 here), so the reduction order — and hence
        the last ulp — is platform-dependent (x86 bitwise, aarch64 one ulp
        apart); ``atol=1e-12`` is two orders below the fp64 energy contract.
        """
        feats = torch.zeros(2, 4, dtype=torch.float64)
        feats[:, 0] = 1.0
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        tight = GTOHartreeEnergy(sigma=1.5, l_max=1, kspace_capacity_factor=1.0)
        default = GTOHartreeEnergy(sigma=1.5, l_max=1, kspace_capacity_factor=1.2)
        out_tight = tight(feats, pos, batch, cell=cell, num_graphs=1)
        out_default = default(feats, pos, batch, cell=cell, num_graphs=1)
        assert tight.kgrid.capacity != default.kgrid.capacity  # not a self-comparison
        assert torch.allclose(out_tight, out_default, atol=1e-12, rtol=0.0)

    def test_capacity_factor_below_one_is_rejected(self) -> None:
        """The grid's own ``>= 1`` guard must not be bypassed by the keyword."""
        with pytest.raises(ValueError, match="capacity_factor must be >= 1"):
            GTOHartreeEnergy(sigma=1.5, l_max=1, kspace_capacity_factor=0.5)

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, AC008_ENERGY_NOSI), (True, AC008_ENERGY_SI)],
    )
    def test_two_charges_periodic_energy_matches_pre_change_literal(
        self, include_self_interaction: bool, expected: float
    ) -> None:
        """fp64 periodic energy is unchanged by the fixed-capacity grid."""
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.zeros(2, 4, dtype=torch.float64)
        feats[:, 0] = 1.0
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        out = model(feats, pos, batch, cell=cell, num_graphs=1)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, AC008_DPOS_NOSI), (True, AC008_DPOS_NOSI)],
    )
    def test_two_charges_position_gradient_matches_pre_change_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        """``∂E/∂pos`` in eV/Å against the pre-change eager path."""
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.zeros(2, 4, dtype=torch.float64)
        feats[:, 0] = 1.0
        leaf = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        leaf.requires_grad_(True)
        batch = torch.zeros(2, dtype=torch.long)
        eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        strain = torch.eye(3, dtype=torch.float64) + eps
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        energy = model(feats, leaf @ strain.T, batch, cell=cell @ strain.T, num_graphs=1)
        dpos, _ = torch.autograd.grad(energy.sum(), (leaf, eps))
        assert torch.allclose(
            dpos, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, AC008_DEPS_NOSI), (True, AC008_DEPS_NOSI)],
    )
    def test_two_charges_strain_gradient_matches_pre_change_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        """``∂E/∂ε`` in eV — the half-box weights carry no strain."""
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.zeros(2, 4, dtype=torch.float64)
        feats[:, 0] = 1.0
        leaf = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        leaf.requires_grad_(True)
        batch = torch.zeros(2, dtype=torch.long)
        eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        strain = torch.eye(3, dtype=torch.float64) + eps
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        energy = model(feats, leaf @ strain.T, batch, cell=cell @ strain.T, num_graphs=1)
        _, deps = torch.autograd.grad(energy.sum(), (leaf, eps))
        assert torch.allclose(
            deps, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_periodic_energy_and_position_gradient_are_finite(self) -> None:
        """the guarded denominator keeps ``0 · inf`` out of the sum."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        leaf = torch.tensor(PROBE_POS, dtype=torch.float64).requires_grad_(True)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        energy = model(feats, leaf, batch, cell=cell, num_graphs=1)
        (dpos,) = torch.autograd.grad(energy.sum(), leaf)
        assert bool(torch.isfinite(energy).all())
        assert bool(torch.isfinite(dpos).all())

    def test_over_capacity_cell_poisons_energy_and_force(self) -> None:
        """an over-capacity grid returns NaN in-graph, never truncates.

        ``_kspace_energy`` is the compiled route: ``forward``'s eager
        ``bind_cell`` guard is skipped there, exactly as
        ``torch.compiler.is_compiling()`` skips it under dynamo. ``2×`` is the
        first round scale that trips D4's ``floor(x) > N_c`` on a 10 Å cell
        bound at ``f = 1.2`` (``x = 2.9391``, ``N_c = 4``, guard fires at
        ``x ≥ 5``); ``1.5×`` is still complete and must *not* be poisoned.
        """
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=True)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        leaf = torch.tensor(PROBE_POS, dtype=torch.float64).requires_grad_(True)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        energy = model._kspace_energy(feats, leaf, batch, cell * 2.0, num_graphs=1)
        (dpos,) = torch.autograd.grad(energy.sum(), leaf)
        assert bool(torch.isnan(energy).all())
        assert bool(torch.isnan(dpos).all())

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, PROBE_PERIODIC10_NOSI), (True, PROBE_PERIODIC10_SI)],
    )
    def test_probe_energy_in_a_10_angstrom_cell_matches_literal(
        self, include_self_interaction: bool, expected: float
    ) -> None:
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        out = model(feats, pos, batch, cell=cell, num_graphs=1)
        assert out.item() == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, PROBE_PERIODIC20_NOSI), (True, PROBE_PERIODIC20_SI)],
    )
    def test_probe_energy_in_a_20_angstrom_cell_matches_literal(
        self, include_self_interaction: bool, expected: float
    ) -> None:
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64) + 10.0
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        out = model(feats, pos, batch, cell=cell, num_graphs=1)
        assert out.item() == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, PROBE_ISOLATED_NOSI), (True, PROBE_ISOLATED_SI)],
    )
    def test_isolated_l1_energy_matches_literal(
        self, include_self_interaction: bool, expected: float
    ) -> None:
        """``scatter_sum_compile_safe`` swaps the reduction kernel.

        ``index_add_`` instead of ``scatter_add_`` is not a bitwise contract on
        CUDA, hence ``atol=1e-12`` rather than ``torch.equal``.
        """
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=1, include_self_interaction=include_self_interaction
        )
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64) + 10.0
        batch = torch.zeros(3, dtype=torch.long)
        out = model(feats, pos, batch, num_graphs=1)
        assert torch.allclose(
            out, torch.tensor([expected], dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, PROBE_ISOLATED_L0_NOSI), (True, PROBE_ISOLATED_L0_SI)],
    )
    def test_isolated_l0_energy_matches_literal(
        self, include_self_interaction: bool, expected: float
    ) -> None:
        model = GTOHartreeEnergy(
            sigma=1.5, l_max=0, include_self_interaction=include_self_interaction
        )
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)[:, :1]
        pos = torch.tensor(PROBE_POS, dtype=torch.float64) + 10.0
        batch = torch.zeros(3, dtype=torch.long)
        out = model(feats, pos, batch, num_graphs=1)
        assert torch.allclose(
            out, torch.tensor([expected], dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    @pytest.mark.slow
    def test_periodic_branch_compiles_to_one_graph(self) -> None:
        """bound grid + explicit ``num_graphs`` → 1 graph, 0 breaks."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        explanation = Compiler.explain(model, feats, pos, batch, cell=cell, num_graphs=1)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    def test_batch_cell_energy_is_bitwise_equal(self) -> None:
        """``(1, 3, 3)`` and ``(3, 3)`` differ by a squeeze, nothing else."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        flat = model(feats, pos, batch, cell=cell, num_graphs=1)
        batched = model(feats, pos, batch, cell=cell.unsqueeze(0), num_graphs=1)
        assert torch.equal(flat, batched)

    @pytest.mark.slow
    def test_compiling_without_bind_cell_names_bind_cell(self) -> None:
        """the unbound grid fails at trace time with its own message."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        # Dynamo re-wraps user exceptions, so match on the message, not the type.
        with pytest.raises(Exception) as excinfo:
            compiled(feats, pos, batch, cell=cell, num_graphs=1)
        assert "bind_cell" in str(excinfo.value)
