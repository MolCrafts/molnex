"""Tests for GTOFieldFeatures."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from molix import config
from molpot.potentials.gto.features import GTOFieldFeatures
from molpot.potentials.gto.kspace import KSpaceGrid

#: Two atoms 6 Å apart (Å), placed by each test.
PAIR_POS: list[list[float]] = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]

#: Source ``ir_mul`` multipoles ``(q, μ_y, μ_z, μ_x)`` in e / e·Å, Σq = 0.
PAIR_FEATS: list[list[float]] = [
    [0.5, 0.10, -0.20, 0.30],
    [-0.5, -0.15, 0.05, 0.25],
]

#: Shared reciprocal cutoff (Å⁻¹) of the dump block: ``σ_min = min(1.5,
#: min(σ_comb)) = 1.5``, ``l = 1``, ``κ = 1`` — the same value the Hartree
#: block freezes, hard-coded from the pre-change capture (molnex @
#: ``feat/oplsaa-parameterizer``, 2026-09-03).
FEAT_K_CUT = 1.8467166200173746

# -- fp64 feature goldens ---------------------------------------------------
#
# ``_dump_block(include_self_interaction=<flag>)`` evaluated on
# :data:`PAIR_FEATS`. Hard-coded literals captured from *this* tree before the
# fixed-capacity k-grid rewrite (``PYTHONPATH=src python
# capture_kgrid_goldens.py``, 2026-09-03). No live oracle.

#: ``PAIR_POS`` in a 10 Å cubic cell, ``include_self_interaction=False``.
FEAT_PERIODIC10_NOSI: list[list[float]] = [
    [
        -6.815618193730831,
        -5.637932583997482,
        -0.15940007827634178,
        -0.1298221414394755,
        0.11201522724282903,
        0.14522970123074813,
        0.31920431148168915,
        -0.03534900734337931,
    ],
    [
        6.768027751977043,
        5.629911847673534,
        0.13570765275958546,
        0.13752592133511182,
        -0.18309250379309827,
        -0.12211836154383926,
        0.26694294142639563,
        -0.027362410914853974,
    ],
]

#: ``PAIR_POS`` in a 10 Å cubic cell, ``include_self_interaction=True``.
FEAT_PERIODIC10_SI: list[list[float]] = [
    [
        2.784145458552945,
        0.4334910441051978,
        0.07214662246550481,
        -0.012667747992813961,
        -0.3510781742408641,
        -0.08907908566257497,
        1.0138444137072289,
        0.3161141729966053,
    ],
    [
        -2.8317359003067333,
        -0.4415117804291457,
        -0.2116123983531844,
        -0.038205668834880496,
        -0.06731915342217497,
        -0.06354116482050848,
        0.8458096932810121,
        0.2655235727017999,
    ],
]

#: ``PAIR_POS + 7`` in a 20 Å cubic cell, ``include_self_interaction=False``.
FEAT_PERIODIC20_NOSI: list[list[float]] = [
    [
        -4.7952771524785,
        -4.3889879331677495,
        -0.05609283045044561,
        -0.07497631389500747,
        0.0265604834884825,
        0.03949702702421154,
        -1.1205021626998157,
        -1.274478439942442,
    ],
    [
        4.852670330049407,
        4.423989681336093,
        0.04132665696946408,
        0.05723667045960967,
        -0.07085900393142705,
        -0.09271595733040551,
        -1.1525291100162054,
        -1.2918550991747488,
    ],
]

#: ``PAIR_POS + 7`` in a 20 Å cubic cell, ``include_self_interaction=True``.
FEAT_PERIODIC20_SI: list[list[float]] = [
    [
        4.804486499805277,
        1.68243569493493,
        0.17545387029140097,
        0.04217807955165408,
        -0.43653291799521066,
        -0.19481175986911156,
        -0.42586206047427594,
        -0.9230152596024574,
    ],
    [
        -4.747093322234369,
        -1.6474339467665862,
        -0.3059933941433058,
        -0.11849491971038265,
        0.04491434643949624,
        -0.034138760607074735,
        -0.5736623581615891,
        -0.9989691155580951,
    ],
]

#: Isolated ``PAIR_POS``, ``include_self_interaction=False``.
FEAT_ISOLATED_NOSI: list[list[float]] = [
    [
        -4.5663116367539756,
        -4.164955796007061,
        -0.043890480005031436,
        -0.05980086187309938,
        0.029492920432122382,
        0.03839753240766618,
        -1.2673365001103016,
        -1.5474934088198644,
    ],
    [
        4.646458086045259,
        4.2140079693722265,
        0.02501539584250865,
        0.03489257913748825,
        -0.08505970481322209,
        -0.11240501228373698,
        -1.268959829361834,
        -1.5487659836473036,
    ],
]

#: Isolated ``PAIR_POS``, ``include_self_interaction=True``.
FEAT_ISOLATED_SI: list[list[float]] = [
    [
        -4.5663116367539756,
        -4.164955796007061,
        -0.043890480005031436,
        -0.05980086187309938,
        0.029492920432122382,
        0.03839753240766618,
        -1.2673365001103016,
        -1.5474934088198644,
    ],
    [
        4.646458086045259,
        4.2140079693722265,
        0.02501539584250865,
        0.03489257913748825,
        -0.08505970481322209,
        -0.11240501228373698,
        -1.268959829361834,
        -1.5487659836473036,
    ],
]


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


def _dump_block(*, include_self_interaction: bool = False) -> GTOFieldFeatures:
    """Polar-1-M dump widths (σ_ρ = 1.5 Å, widths 1.5 / 3.0 Å, l = 1)."""
    return GTOFieldFeatures(
        density_sigma=1.5,
        density_lmax=1,
        feature_lmax=1,
        feature_widths=[1.5, 3.0],
        include_self_interaction=include_self_interaction,
    )


class TestGTOFieldFeatures:
    """Isolated-molecule field features for Polar-1-M dump widths."""

    def test_dump_widths_emit_ir_mul_dim_8(self):
        """Two widths, lmax=1 → 2x0e+2x1o dim 8."""
        block = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
        )
        assert block.out_dim == 8
        n = 3
        out = block(
            torch.randn(n, 4, dtype=torch.float64),
            torch.randn(n, 3, dtype=torch.float64),
            torch.zeros(n, dtype=torch.long),
        )
        assert out.shape == (n, 8)

    def test_translation_invariance(self):
        """Relative geometry: a rigid translation must not change features."""
        block = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
        )
        torch.manual_seed(0)
        q = torch.randn(4, 4, dtype=torch.float64)
        pos = torch.randn(4, 3, dtype=torch.float64)
        batch = torch.zeros(4, dtype=torch.long)
        a = block(q, pos, batch)
        b = block(q, pos + 1.7, batch)
        assert torch.allclose(a, b, atol=1e-8)

    def test_periodic_self_overlap_constants(self):
        """``r → 0`` limits of the two real-space kernels, per feature width.

        With ``σ_c = sqrt((σ_ρ² + w²)/2)``: ``erf(r/2σ_c)/r → 1/(σ_c√π)`` for
        the potential, and a Gaussian dipole's own field at its centre is
        ``μ/(6√π σ_c³)``. Goldens are those closed forms evaluated at
        ``σ_ρ = 1.5 Å``, ``w = [1.5, 3.0] Å``, ``offset = 0.1 Å`` — hard-coded,
        no live oracle.
        """
        block = _dump_block()
        assert block.include_self_interaction is False
        assert block.self_overlap_l0.shape == (2,)
        assert block.self_overlap_l1.shape == (2,)
        assert block.self_overlap_l0.dtype == torch.float64
        assert block.self_overlap_l1.dtype == torch.float64
        assert block.self_overlap_l0[0].item() == pytest.approx(19.199527, rel=1e-6)
        assert block.self_overlap_l0[1].item() == pytest.approx(12.142847, rel=1e-6)
        assert block.self_overlap_l1[0].item() == pytest.approx(2.315467, rel=1e-6)
        assert block.self_overlap_l1[1].item() == pytest.approx(1.171544, rel=1e-6)

    def test_large_cell_kspace_features_match_isolated(self):
        """A 60 Å cell reproduces the isolated features once the own site is dropped.

        The real-space pair sum never pairs an atom with itself, so the
        reciprocal path must remove the ``i = j`` term to answer the same
        question. What is left over is not that term (~9.6 eV on the scalar
        columns, ``self_overlap_l0 · q``): it is the periodic images (~1e-2,
        falling as ``L⁻³``) plus the structural gap between the real-space
        finite difference ``φ(r+δ) − φ(r)`` at ``δ = 0.1 Å`` and the analytic
        ``∇φ`` the k-space path uses (~1e-2, independent of ``L``). Hence
        ``atol=3e-2``, still ~300× below the term under test — a permuted
        ``(μ_y, μ_z, μ_x)`` self-subtraction misses by 0.7 eV and fails here.
        """
        block = _dump_block()
        pos = torch.tensor(PAIR_POS, dtype=torch.float64) + 30.0
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 60.0
        periodic = block(feats, pos, batch, cell=cell)
        isolated = block(feats, pos, batch)
        own_site = (block.self_overlap_l0 * feats[:, 0:1].abs()).max().item()
        assert own_site > 3.0  # the tolerance below is not vacuous
        assert torch.allclose(periodic, isolated, atol=3e-2, rtol=0.0)

    def test_include_self_interaction_keeps_own_contribution(self):
        """``True`` − ``False`` is exactly the atom's own multipole at its own site.

        Scalar columns take ``self_overlap_l0 · q_i``; the vector block is
        ``self_overlap_l1 · μ_{i,m}`` in the ``ir_mul`` order
        ``(m = y, m = z, m = x)`` of the source multipoles ``feats[:, 1:4]``.
        Cell size is irrelevant to this difference, so a cheap 20 Å box is used.
        """
        keep = _dump_block(include_self_interaction=True)
        drop = _dump_block(include_self_interaction=False)
        pos = torch.tensor(PAIR_POS, dtype=torch.float64) + 7.0
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        delta = keep(feats, pos, batch, cell=cell) - drop(feats, pos, batch, cell=cell)
        scalars = drop.self_overlap_l0 * feats[:, 0:1]
        vectors = torch.stack(
            [drop.self_overlap_l1 * feats[:, m : m + 1] for m in (1, 2, 3)], dim=1
        ).reshape(feats.shape[0], -1)
        expect = torch.cat([scalars, vectors], dim=-1)
        assert delta.shape == (2, drop.out_dim)
        assert expect.abs().max().item() > 1.0  # not a comparison of two zeros
        assert torch.allclose(delta, expect, atol=1e-9, rtol=0.0)


class TestGTOFieldFeaturesKspace:
    """Fixed-capacity k-grid contract for the field-feature block."""

    def test_k_cutoff_is_frozen_at_construction(self) -> None:
        """``combined_sigma.min()`` is a ctor-time Python float, not a forward read."""
        block = _dump_block()
        assert block.kgrid.k_cutoff == FEAT_K_CUT

    def test_bind_cell_sizes_the_grid(self) -> None:
        block = _dump_block()
        block.bind_cell(torch.eye(3, dtype=torch.float64) * 10.0)
        assert block.kgrid.capacity == (4, 4, 4)

    def test_k_cutoff_property_reports_the_dump_cutoff(self) -> None:
        """The block answers "what cutoff am I summing to?" like its Hartree twin."""
        assert _dump_block().k_cutoff == FEAT_K_CUT

    def test_k_cutoff_reads_the_grid_rather_than_recomputing_it(self) -> None:
        """One home for the cutoff: the property delegates to :attr:`kgrid`.

        Swapping the grid is the documented way to trade head-room for
        k-points, so a property that re-derives ``kspace_cutoff(σ_min, l, κ)``
        would report a cutoff the sum no longer uses.
        """
        block = _dump_block()
        block.kgrid = KSpaceGrid(k_cutoff=0.5)
        assert block.k_cutoff == 0.5

    def test_capacity_factor_keyword_reaches_the_grid(self) -> None:
        """``kspace_capacity_factor`` is forwarded verbatim to the grid."""
        block = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
            kspace_capacity_factor=1.0,
        )
        assert block.kgrid.capacity_factor == 1.0

    def test_capacity_factor_one_sizes_a_tighter_box(self) -> None:
        """NVT head-room: ``f = 1.0`` on a 10 Å cell is ``(3, 3, 3)``, not ``(4, 4, 4)``.

        ``x = ‖a_i‖ k_cutoff / 2π = 2.9391`` at the dump settings, so
        ``N_c = ceil(f x)`` is 3 at ``f = 1.0`` and 4 at the default ``f = 1.2``.
        """
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        tight = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
            kspace_capacity_factor=1.0,
        )
        tight.bind_cell(cell)
        default = _dump_block()
        default.bind_cell(cell)
        assert tight.kgrid.capacity == (3, 3, 3)
        assert default.kgrid.capacity == (4, 4, 4)

    def test_capacity_factor_does_not_change_the_periodic_features(self) -> None:
        """D3: the extra half-box rows carry weight ``0`` and add exactly ``+0.0``.

        Head-room is an allocation decision, never a physics one. Padding rows
        contribute exactly ``+0.0`` per term, but the two grids sum different
        row counts (171 against 364 here), so the reduction order — and hence
        the last ulp — is platform-dependent (x86 bitwise, aarch64 one ulp
        apart); ``atol=1e-12`` is well below the feature scale under test.
        """
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        tight = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
            kspace_capacity_factor=1.0,
        )
        default = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
            kspace_capacity_factor=1.2,
        )
        out_tight = tight(feats, pos, batch, cell=cell, num_graphs=1)
        out_default = default(feats, pos, batch, cell=cell, num_graphs=1)
        assert out_tight.shape == (2, 8)
        assert tight.kgrid.capacity != default.kgrid.capacity  # not a self-comparison
        assert torch.allclose(out_tight, out_default, atol=1e-12, rtol=0.0)

    def test_capacity_factor_below_one_is_rejected(self) -> None:
        """The grid's own ``>= 1`` guard must not be bypassed by the keyword."""
        with pytest.raises(ValueError, match="capacity_factor must be >= 1"):
            GTOFieldFeatures(
                density_sigma=1.5,
                density_lmax=1,
                feature_lmax=1,
                feature_widths=[1.5, 3.0],
                kspace_capacity_factor=0.5,
            )

    def test_ctor_derived_buffers_leave_the_state_dict(self) -> None:
        """ac-016: five ``persistent=False`` buffers — one home per fact."""
        assert set(_dump_block().state_dict()) == set()

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, FEAT_PERIODIC10_NOSI), (True, FEAT_PERIODIC10_SI)],
    )
    def test_periodic_features_in_a_10_angstrom_cell_match_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        block = _dump_block(include_self_interaction=include_self_interaction)
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        out = block(feats, pos, batch, cell=cell, num_graphs=1)
        assert torch.allclose(
            out, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, FEAT_PERIODIC20_NOSI), (True, FEAT_PERIODIC20_SI)],
    )
    def test_periodic_features_in_a_20_angstrom_cell_match_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        block = _dump_block(include_self_interaction=include_self_interaction)
        pos = torch.tensor(PAIR_POS, dtype=torch.float64) + 7.0
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        out = block(feats, pos, batch, cell=cell, num_graphs=1)
        assert torch.allclose(
            out, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    @pytest.mark.parametrize(
        ("include_self_interaction", "expected"),
        [(False, FEAT_ISOLATED_NOSI), (True, FEAT_ISOLATED_SI)],
    )
    def test_isolated_features_match_literal(
        self, include_self_interaction: bool, expected: list[list[float]]
    ) -> None:
        """The reduction kernel swap is ``atol=1e-12``, not bitwise (ac-007)."""
        block = _dump_block(include_self_interaction=include_self_interaction)
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        out = block(feats, pos, batch, num_graphs=1)
        assert torch.allclose(
            out, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    def test_batch_cell_features_are_bitwise_equal(self) -> None:
        """ac-014: ``(1, 3, 3)`` differs from ``(3, 3)`` by a squeeze, nothing else."""
        block = _dump_block()
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        flat = block(feats, pos, batch, cell=cell, num_graphs=1)
        batched = block(feats, pos, batch, cell=cell.unsqueeze(0), num_graphs=1)
        assert torch.equal(flat, batched)

    def test_periodic_branch_compiles_to_one_graph(self) -> None:
        """ac-006: bound grid + explicit ``num_graphs`` → 1 graph, 0 breaks."""
        block = _dump_block()
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        block.bind_cell(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(block)(feats, pos, batch, cell=cell, num_graphs=1)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    def test_missing_num_graphs_keeps_the_documented_break(self) -> None:
        """``num_graphs=None`` is the eager-only fallback and must stay visible."""
        block = _dump_block()
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        block.bind_cell(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(block)(feats, pos, batch, cell=cell)
        assert explanation.graph_break_count > 0, (
            "num_graphs=None is the documented eager-only fallback: it keeps "
            "int(batch.max().item()) and must not silently become compile-clean."
        )

    def test_isolated_branch_still_breaks_by_design(self) -> None:
        """ac-007: the retained all-pairs enumeration is a *named* break."""
        block = _dump_block()
        pos = torch.tensor(PAIR_POS, dtype=torch.float64)
        feats = torch.tensor(PAIR_FEATS, dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(block)(feats, pos, batch, num_graphs=1)
        assert explanation.graph_break_count > 0, (
            "the isolated all-pairs enumeration is retained on purpose (D6); "
            "its break is owned by spec polar-compile-clean-04-pairs and must "
            "not become an accidental silent contract."
        )

    def test_kspace_call_chain_has_no_host_sync(self) -> None:
        """ac-005: no ``.item()``, no ``int()``/``float()``, no boolean-mask index."""
        path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "molpot"
            / "potentials"
            / "gto"
            / "features.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {
            node.name: node
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == "GTOFieldFeatures"
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in ("_kspace_features", "_kspace_potential"):
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
            / "features.py"
        )
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("from molix.F.scatter import")
        ]
        assert lines == ["from molix.F.scatter import scatter_sum_compile_safe"]

    def test_kspace_is_imported_by_public_name_only(self) -> None:
        """No private reach-through into :mod:`molpot.potentials.gto.kspace`.

        ``_as_cell`` / ``_FOURPI`` are that module's own internals; importing
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
            / "features.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "molpot.potentials.gto.kspace"
            for alias in node.names
        ]
        assert imported, "features.py must still import from the kspace module"
        assert [name for name in imported if name.startswith("_")] == []
