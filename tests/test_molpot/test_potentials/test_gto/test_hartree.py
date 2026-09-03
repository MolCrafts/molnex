"""Tests for GTOHartreeEnergy (isolated real-space and periodic k-space)."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from molix import config
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto import hartree as hartree_module
from molpot.potentials.gto.hartree import GTOHartreeEnergy, gto_self_overlap
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

# -- ac-008 goldens ---------------------------------------------------------
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
AC008_DPOS_SI: list[list[float]] = [
    [0.3128153444964179, -5.963111948670274e-19, 2.168404344971009e-18],
    [-0.3128153444964179, -7.047314121155779e-19, -2.168404344971009e-18],
]

#: ``∂E/∂ε`` (3, 3) in eV, ``include_self_interaction=True``.
AC008_DEPS_SI: list[list[float]] = [
    [1.470186763029792, 2.874766613593052e-17, 1.4169465773160073e-17],
    [-9.416870756969064e-17, 1.9180416642629678, -2.0709219206926254e-17],
    [3.0530362823859914e-17, -2.0709219206926254e-17, 1.9180416642629678],
]

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

# -- Retained isolated-branch enumeration (ac-007) --------------------------
#
# ``inspect.getsource`` of the four pair-enumeration helpers, captured from the
# same pre-change tree. They are *not* touched by the k-grid work (their break
# is a named, deliberate debt owned by ``polar-compile-clean-04-pairs``), so a
# character-for-character comparison is the guard. A line-range git diff is not
# usable: the imports above them shift.

_RETAINED_SOURCE: dict[str, str] = {
    "_skip_diagonal": '''\
def _skip_diagonal(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All directed pairs ``(i, j)`` with ``i != j``, no ``N×N`` mask."""
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    idx = torch.arange(n * (n - 1), device=device)
    src = torch.div(idx, n - 1, rounding_mode="floor")
    dst = idx.remainder(n - 1)
    return src, dst + (dst >= src).to(dtype=dst.dtype)
''',
    "_skip_equal_orig_regular": '''\
def _skip_equal_orig_regular(
    n_atoms: int, copies: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs of ``n_atoms * copies`` sites whose original atom differs.

    Layout is ``orig = arange(n_atoms).repeat_interleave(copies)``.
    """
    atom_src, atom_dst = _skip_diagonal(n_atoms, device)
    copy = torch.arange(copies, device=device)
    src, dst = torch.broadcast_tensors(
        atom_src[:, None, None] * copies + copy[None, :, None],
        atom_dst[:, None, None] * copies + copy[None, None, :],
    )
    return src.reshape(-1), dst.reshape(-1)
''',
    "_complete_directed_pairs": '''\
def _complete_directed_pairs(
    batch: torch.Tensor,
    orig: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs that share a graph and come from different original atoms.

    ``orig`` is the pre-duplication atom id (identity for ``l_max=0``; repeated
    4× for the dipole finite-difference copies). Copies of the same atom never
    interact — that is the optional self-interaction term.
    """
    n = int(batch.shape[0])
    device = batch.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    if n == 0:
        return empty, empty
    single = bool((batch == batch[0]).all().item())
    if orig is None:
        src, dst = _skip_diagonal(n, device)
        if single:
            return src, dst
        keep = batch[src] == batch[dst]
        return src[keep], dst[keep]

    if single:
        first = orig[0]
        copies = int((orig == first).sum().item())
        n_atoms = n // copies if copies else 0
        if copies > 1 and n_atoms * copies == n:
            expected = torch.arange(n_atoms, device=device, dtype=orig.dtype).repeat_interleave(
                copies
            )
            if torch.equal(orig, expected):
                return _skip_equal_orig_regular(n_atoms, copies, device)
        src, dst = _skip_diagonal(n, device)
        keep = orig[src] != orig[dst]
        return src[keep], dst[keep]

    src, dst = _skip_diagonal(n, device)
    keep = (batch[src] == batch[dst]) & (orig[src] != orig[dst])
    return src[keep], dst[keep]
''',
    "cached_complete_directed_pairs": '''\
def cached_complete_directed_pairs(
    owner: object,
    batch: torch.Tensor,
    orig: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse pair indices while ``N``, graph membership and orig layout stay put."""
    osum = 0 if orig is None else int(orig.sum().item()) + 1
    key = (int(batch.shape[0]), int(batch.sum().item()), osum, batch.device)
    cached = getattr(owner, "_pair_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    src, dst = _complete_directed_pairs(batch, orig)
    owner._pair_cache = (key, src, dst)
    return src, dst
''',
}


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestGTOHartreeEnergy:
    """l=0 pair Hartree ``k_e q_i q_j erf(r/(2σ))/r``, plus the ``cell`` branch."""

    def test_two_unit_charges_one_angstrom_apart(self):
        """σ=1.5 Å, r=1 Å, q=1,1 → eV_A * erf(1/3)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        q = torch.ones(2, 1, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = energy(q, pos, batch)
        expect = eV_A * math.erf(1.0 / 3.0)
        assert out.shape == (1,)
        assert out.item() == pytest.approx(expect, rel=1e-10)

    def test_self_pairs_are_excluded(self):
        """A single charge has no pair energy (self-interaction is a separate term)."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=False)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        assert energy(q, pos, batch).item() == pytest.approx(0.0)

    def test_unit_monopole_self_energy(self):
        """Dump σ=1.5 Å: ½ q² eV_A/(σ√π) for a lone unit charge."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0, include_self_interaction=True)
        pos = torch.zeros(1, 3, dtype=torch.float64)
        q = torch.ones(1, 1, dtype=torch.float64)
        batch = torch.zeros(1, dtype=torch.long)
        expect = 0.5 * eV_A / (1.5 * math.sqrt(math.pi))
        assert energy(q, pos, batch).item() == pytest.approx(expect, rel=1e-8)

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
        e0 = GTOHartreeEnergy(sigma=1.5, l_max=0)(q0, pos, batch)
        e1 = GTOHartreeEnergy(sigma=1.5, l_max=1)(q1, pos, batch)
        assert torch.allclose(e0, e1, rtol=1e-10, atol=1e-12)

    def test_l1_pair_count_skips_same_atom_copies(self):
        """2 atoms × 4 copies → 32 directed pairs, not 8×7."""
        from molpot.potentials.gto.hartree import _complete_directed_pairs

        orig = torch.arange(2).repeat_interleave(4)
        batch = torch.zeros(8, dtype=torch.long)
        src, dst = _complete_directed_pairs(batch, orig)
        assert src.numel() == 32
        assert int((orig[src] == orig[dst]).sum()) == 0

    def test_graphs_do_not_mix(self):
        """Two isolated pairs in different graphs keep their own energies."""
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        q = torch.ones(4, 1, dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        out = energy(q, pos, batch)
        expect = eV_A * math.erf(1.0 / 3.0)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.full((2,), expect, dtype=torch.float64), rtol=1e-10)

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
        periodic = model(feats, pos, batch, cell=cell)
        isolated = model(feats, pos, batch)
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
        assert model(feats, pos, batch, cell=cell).item() == pytest.approx(expect, rel=3e-3)

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
        e_true = model(feats, pos, batch, cell=cell)
        e_false = without(feats, pos, batch, cell=cell)
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
        out = energy(q, pos, batch, cell=cell)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_forces_obey_newtons_third_law(self):
        """Opposite charges in a cube: F_1 = −F_0, attraction along the bond."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        q = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 14.0
        leaf = pos.detach().requires_grad_(True)
        energy = GTOHartreeEnergy(sigma=1.5, l_max=0)(q, leaf, batch, cell=cell).sum()
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
        e0 = energy(q, pos, batch, cell=cell)
        e1 = energy(q, pos + 1.7, batch, cell=cell)
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
        e0 = model(monopole, pos, batch, cell=cell)
        e1 = model(with_dipole, pos, batch, cell=cell)
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
        half = model(q, pos, batch, cell=cell)
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
        energy = model(q, pos, batch, cell=cell)
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
            energy(q, pos, batch, cell=cell)


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

    def test_capacity_factor_one_sizes_a_tighter_box(self) -> None:
        """NVT head-room: ``f = 1.0`` on a 10 Å cell is ``(3, 3, 3)``, not ``(4, 4, 4)``.

        ``x = ‖a_i‖ k_cutoff / 2π = 2.9391`` at the dump settings, so
        ``N_c = ceil(f x)`` is 3 at ``f = 1.0`` and 4 at the default ``f = 1.2``.
        """
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        tight = GTOHartreeEnergy(sigma=1.5, l_max=1, kspace_capacity_factor=1.0)
        tight.bind_cell(cell)
        default = GTOHartreeEnergy(sigma=1.5, l_max=1)
        default.bind_cell(cell)
        assert tight.kgrid.capacity == (3, 3, 3)
        assert default.kgrid.capacity == (4, 4, 4)

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
        """ac-008: fp64 periodic energy is unchanged by the fixed-capacity grid."""
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
        [(False, AC008_DPOS_NOSI), (True, AC008_DPOS_SI)],
    )
    def test_two_charges_position_gradient_matches_pre_change_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        """ac-008: ``∂E/∂pos`` in eV/Å against the pre-change eager path."""
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
        [(False, AC008_DEPS_NOSI), (True, AC008_DEPS_SI)],
    )
    def test_two_charges_strain_gradient_matches_pre_change_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        """ac-008 / D7: ``∂E/∂ε`` in eV — the half-box weights carry no strain."""
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
        """ac-003: the guarded denominator keeps ``0 · inf`` out of the sum."""
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
        """ac-010: an over-capacity grid returns NaN in-graph, never truncates.

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
        """ac-007: ``scatter_sum_compile_safe`` swaps the reduction kernel.

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

    def test_periodic_branch_compiles_to_one_graph(self) -> None:
        """ac-006: bound grid + explicit ``num_graphs`` → 1 graph, 0 breaks."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model)(feats, pos, batch, cell=cell, num_graphs=1)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    def test_batch_cell_compiles_to_one_graph(self) -> None:
        """ac-014: ``(1, 3, 3)`` is squeezed inside the traced region."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0)
        model.bind_cell(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model)(feats, pos, batch, cell=cell, num_graphs=1)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    def test_batch_cell_energy_is_bitwise_equal(self) -> None:
        """ac-014: ``(1, 3, 3)`` and ``(3, 3)`` differ by a squeeze, nothing else."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        flat = model(feats, pos, batch, cell=cell, num_graphs=1)
        batched = model(feats, pos, batch, cell=cell.unsqueeze(0), num_graphs=1)
        assert torch.equal(flat, batched)

    def test_missing_num_graphs_keeps_the_documented_break(self) -> None:
        """``num_graphs=None`` is the eager-only fallback and must stay visible."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model)(feats, pos, batch, cell=cell)
        assert explanation.graph_break_count > 0, (
            "num_graphs=None is the documented eager-only fallback: it keeps "
            "int(batch.max().item()) and must not silently become compile-clean."
        )

    def test_isolated_branch_still_breaks_by_design(self) -> None:
        """ac-007: the retained all-pairs enumeration is a *named* break."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=1, include_self_interaction=False)
        feats = torch.tensor(PROBE_FEATS, dtype=torch.float64)
        pos = torch.tensor(PROBE_POS, dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(model)(feats, pos, batch, num_graphs=1)
        assert explanation.graph_break_count > 0, (
            "the isolated all-pairs enumeration is retained on purpose (D6); "
            "its break is owned by spec polar-compile-clean-04-pairs and must "
            "not become an accidental silent contract."
        )

    def test_compiling_without_bind_cell_names_bind_cell(self) -> None:
        """ac-011: the unbound grid fails at trace time with its own message."""
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

    def test_two_graph_batch_is_rejected_with_explicit_num_graphs(self) -> None:
        """ac-014: the owner's own single-system guard survives ``num_graphs``."""
        model = GTOHartreeEnergy(sigma=1.5, l_max=0)
        pos = torch.zeros(4, 3, dtype=torch.float64)
        q = torch.ones(4, 1, dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        with pytest.raises(ValueError, match="single-system"):
            model(q, pos, batch, cell=cell, num_graphs=2)

    def test_retained_pair_helpers_are_byte_identical(self) -> None:
        """ac-007: the four isolated-branch helpers are untouched, character-exact."""
        for name, source in _RETAINED_SOURCE.items():
            assert inspect.getsource(getattr(hartree_module, name)) == source, name

    def test_periodic_call_chain_has_no_host_sync(self) -> None:
        """ac-005: no ``.item()``, no ``int()``/``float()``, no boolean-mask index."""
        path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "molpot"
            / "potentials"
            / "gto"
            / "hartree.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: node
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == "GTOHartreeEnergy"
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in ("_kspace_energy", "_self_energy"):
            assert name in methods, name
            for node in ast.walk(methods[name]):
                if isinstance(node, ast.Call):
                    func = node.func
                    assert not (isinstance(func, ast.Attribute) and func.attr == "item"), name
                    assert not (isinstance(func, ast.Name) and func.id in {"int", "float"}), name
                if isinstance(node, ast.Subscript):
                    assert not isinstance(node.slice, ast.Compare | ast.BoolOp), name

    def test_scatter_import_is_compile_safe(self) -> None:
        """ac-005: ``scatter_sum`` hides an ``.item()`` on its ``dim_size=None`` path."""
        path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "molpot"
            / "potentials"
            / "gto"
            / "hartree.py"
        )
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("from molix.F.scatter import")
        ]
        assert lines == ["from molix.F.scatter import scatter_sum_compile_safe"]

    def test_kspace_is_imported_by_public_name_only(self) -> None:
        """No private reach-through into :mod:`molpot.potentials.gto.kspace`.

        ``_as_cell`` / ``_TWOPI`` are that module's own internals; importing
        them here welds the two modules into one unit with an underscore
        between them, and freezes a name the owner is free to rename. Whatever
        this module needs, ``kspace`` must expose publicly.
        """
        path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "molpot"
            / "potentials"
            / "gto"
            / "hartree.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "molpot.potentials.gto.kspace"
            for alias in node.names
        ]
        assert imported, "hartree.py must still import from the kspace module"
        assert [name for name in imported if name.startswith("_")] == []
