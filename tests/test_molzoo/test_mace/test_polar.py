"""Tests for molzoo.mace.polar — the Polar representation blocks, MacePolar and its key table.

The blocks are pinned against their closed forms and the official dump's
widths; the product only on what it adds over :class:`MacePotential`
(``test_potential.py``): the Hartree term, the charge-aware mask, the two
k-space grids. Everything builds at fp64 (``conftest.DTYPE``).
"""

from __future__ import annotations

import math

import pytest
import torch

import molrep.eq as meq
from molix.compile import Compiler
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY
from molzoo.checkpoint import CheckpointRemap
from molzoo.mace.checkpoint import MACE_CONVERSIONS, MACE_SKIP, polar_keys
from molzoo.mace.polar import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    EnvironmentDependentSpinSourceBlock,
    MacePolar,
    MultiLayerFeatureMixer,
    OneBodyMLPFieldReadout,
    SparseUvuTensorProduct,
)
from molzoo.mace.spec import MACEPolarSpec
from tests.conftest import make_graph_batch, rotate_irreps_features_simple
from tests.test_molzoo.test_mace.conftest import (
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    DTYPE,
    MAX_ATOMS_PER_GRAPH,
    SINGLE_GRAPH,
    TINY_POLAR_KWARGS,
    full_edge_index,
    official_layout,
)

polar_remap = CheckpointRemap(
    polar_keys, on_unexpected="return", skip=MACE_SKIP, conversions=MACE_CONVERSIONS
)

FORCES = ("atoms.pos",)


def _periodic_cluster(cell: torch.Tensor):
    """The cluster in ``cell`` as a one-graph batch carrying ``graphs.cell``."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs={"cell": cell.unsqueeze(0)},
    )


class TestMacePolar:
    """Polar's own ``energy``, the two ``bind_*`` steps and construction on the tiny backbone.

    The shared product surface (``energy_dtype`` branches, the readout stack,
    the mask) is asserted on MatPES in ``test_potential.py``; here only what
    Polar overrides: the geometry *and* charge/spin cast, the Hartree term on
    ``graphs.energy``, the ghosts' charges, the two pair capacities and the
    two k-space grids.
    """

    def test_official_runtime_cutoff_overrides_constructor_heuristic(self, tiny_polar_spec):
        model = MacePolar(
            tiny_polar_spec.model_copy(
                update={"kspace_cutoff": 1.234, "kspace_cutoff_factor": 1.5}
            ),
            dtype=DTYPE,
        )
        assert model.electric_potential_descriptor.k_cutoff == 1.234
        assert model.coulomb_energy.k_cutoff == 1.234

    @pytest.mark.parametrize("dtype", [None, torch.float32])
    def test_energy_precision_covers_hartree(self, tiny_polar_spec, omol_cluster, dtype):
        """Polar's ``energy`` lands the Hartree term at ``energy_dtype`` too (``None``: model's)."""
        model = MacePolar(tiny_polar_spec, energy_dtype=dtype, dtype=DTYPE)
        model.bind_capacity(MAX_ATOMS_PER_GRAPH)
        energy = model(omol_cluster)[ENERGY_KEY]
        assert energy.dtype == (dtype or torch.float64)

    def test_input_dtype_keeps_md_geometry_and_force_dtype(self, omol_cluster):
        """Polar's own ``energy`` casts the geometry (and charge / spin) too."""
        model = MacePolar(
            MACEPolarSpec(
                atomic_numbers=ATOMIC_NUMBERS,
                atomic_energies=ATOMIC_ENERGIES,
                **TINY_POLAR_KWARGS,
            ),
            wrt=FORCES,
            input_dtype=torch.float32,
            dtype=torch.float32,
        ).eval()
        model.bind_capacity(MAX_ATOMS_PER_GRAPH)
        pos = omol_cluster["atoms", "pos"]
        before = pos.clone()
        output = model(omol_cluster)
        assert output[FORCES_KEY].dtype == torch.float64
        assert torch.isfinite(output[FORCES_KEY]).all()
        torch.testing.assert_close(pos, before, atol=0, rtol=0)

    def test_polar_writes_atomic_energy_and_graph_extra(
        self, tiny_polar_spec: MACEPolarSpec, omol_cluster, monkeypatch: pytest.MonkeyPatch
    ):
        """``atoms.energy`` is the per-atom part; ``graphs.energy`` adds the Hartree term."""
        model = MacePolar(tiny_polar_spec, dtype=DTYPE).eval()
        model.bind_capacity(MAX_ATOMS_PER_GRAPH)
        hartree: list[torch.Tensor] = []
        coulomb = model.coulomb_energy.forward

        def record(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
            out = coulomb(*args, **kwargs)
            hartree.append(out.detach().clone())
            return out

        monkeypatch.setattr(model.coulomb_energy, "forward", record)
        out = model.energy(omol_cluster)
        site, energy = out[ATOMIC_ENERGY_KEY], out[ENERGY_KEY]
        assert site.shape == (len(CLUSTER_Z),) and energy.shape == (1,)
        assert torch.isfinite(energy).all()
        assert len(hartree) == 1 and float(hartree[0].abs()) > 1e-6
        torch.testing.assert_close(site.sum().reshape(1) + hartree[0], energy, atol=1e-10, rtol=0.0)

    def test_a_padded_batch_returns_the_unpadded_energy_and_forces(
        self, tiny_polar_spec: MACEPolarSpec, omol_cluster
    ):
        """``MoleculePadder`` ghosts are inert through the long-range terms too.

        Polar's per-atom energies are masked like MatPES's, and the ghost
        atoms' densities and Fukui sources are masked as well, so the charge
        equilibration, the field descriptor and the Hartree sum never see a
        ghost charge. The cluster is translated off the origin first: a padded
        ghost sits at ``(0, 0, 0)``, and a real atom coincident with it would
        exercise the kernels' small-``r`` clamp rather than the mask. Without
        ``max_graphs`` the padder files the ghosts under graph 0, so they
        count against that graph's pair capacity.
        """
        from molix.data import MoleculePadder

        torch.manual_seed(0)
        model = MacePolar(tiny_polar_spec, wrt=FORCES, dtype=DTYPE).eval()
        cluster = omol_cluster.clone()
        cluster["atoms", "pos"] = cluster["atoms", "pos"] + 2.5
        n_atoms = int(cluster["atoms"].batch_size[0])
        n_pairs = int(cluster["pairs"].batch_size[0])
        model.bind_capacity(n_atoms + 3)
        reference = model(cluster.clone())
        padded = MoleculePadder(
            max_atoms=n_atoms + 3, max_pairs=n_pairs + 5, cutoff=tiny_polar_spec.r_max
        )(cluster.clone())
        out = model(padded)
        assert torch.allclose(out[ENERGY_KEY], reference[ENERGY_KEY], atol=1e-10, rtol=0.0)
        assert torch.allclose(
            out[FORCES_KEY][:n_atoms], reference[FORCES_KEY], atol=1e-10, rtol=0.0
        )
        assert torch.allclose(
            out[FORCES_KEY][n_atoms:], torch.zeros(3, 3, dtype=torch.float64), atol=1e-12, rtol=0.0
        )

    def test_local_electron_energy_is_skipped_when_disabled(self, tiny_polar_spec, omol_cluster):
        """``add_local_electron_energy=False`` never calls the readout: no grad reaches it."""
        model = MacePolar(
            tiny_polar_spec.model_copy(update={"add_local_electron_energy": False}), dtype=DTYPE
        )
        model.bind_capacity(MAX_ATOMS_PER_GRAPH)
        model.energy(omol_cluster)[ENERGY_KEY].sum().backward()
        assert all(p.grad is None for p in model.local_electron_energy.parameters())
        assert all(p.grad is not None for p in model.layer_feature_mixer.parameters())

    def test_self_interaction_flags_reach_both_electrostatic_blocks(
        self, tiny_polar_spec: MACEPolarSpec
    ):
        """Hartree keeps the on-site term by default, field features drop it."""
        model = MacePolar(tiny_polar_spec, dtype=DTYPE)
        assert model.coulomb_energy.include_self_interaction is True
        assert model.electric_potential_descriptor.include_self_interaction is False
        flipped = MacePolar(
            tiny_polar_spec.model_copy(
                update={"electrostatic_self_interaction": False, "field_self_interaction": True}
            ),
            dtype=DTYPE,
        )
        assert flipped.coulomb_energy.include_self_interaction is False
        assert flipped.electric_potential_descriptor.include_self_interaction is True

    def test_energy_forwards_num_graphs_to_both_gto_blocks(
        self, tiny_polar_spec: MACEPolarSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One coulomb call and two descriptor calls per recursion step, each with ``B`` + cell."""
        model = MacePolar(tiny_polar_spec, dtype=DTYPE).eval()
        coulomb_calls: list[dict[str, object]] = []
        descriptor_calls: list[dict[str, object]] = []
        coulomb_forward = model.coulomb_energy.forward
        descriptor_forward = model.electric_potential_descriptor.forward

        def record_coulomb(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
            coulomb_calls.append(dict(kwargs))
            return coulomb_forward(*args, **kwargs)

        def record_descriptor(*args: torch.Tensor, **kwargs: object) -> torch.Tensor:
            descriptor_calls.append(dict(kwargs))
            return descriptor_forward(*args, **kwargs)

        monkeypatch.setattr(model.coulomb_energy, "forward", record_coulomb)
        monkeypatch.setattr(model.electric_potential_descriptor, "forward", record_descriptor)

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        with torch.no_grad():
            model.energy(_periodic_cluster(cell))

        assert len(coulomb_calls) == 1
        assert len(descriptor_calls) == 2 * model.num_recursion_steps
        for kwargs in coulomb_calls + descriptor_calls:
            assert kwargs.get("num_graphs") == 1
            assert kwargs.get("cell") is not None

    def test_bind_capacity_reaches_both_gto_blocks(self, tiny_polar_spec: MACEPolarSpec) -> None:
        """One call bounds both isolated pair lists — the two blocks are independent objects."""
        model = MacePolar(tiny_polar_spec, dtype=DTYPE)
        model.bind_capacity(7)
        assert model.electric_potential_descriptor.max_atoms_per_graph == 7
        assert model.coulomb_energy.max_atoms_per_graph == 7

    @pytest.mark.slow
    def test_isolated_forward_compiles_to_one_graph(
        self, tiny_polar_spec: MACEPolarSpec, pair_batch
    ) -> None:
        """Bound capacity + one eager forward → the isolated op is 1 graph, 0 breaks.

        Graphs of 4 and 3 atoms at the cluster's capacity of 5, so both GTO
        pair lists carry padding rows. Polar's isolated path (the per-graph
        pair lists with their capacity and contiguity checks) is its own
        compile seam, distinct from the periodic one below.
        """
        model = MacePolar(tiny_polar_spec, dtype=DTYPE).eval()
        model.bind_capacity(MAX_ATOMS_PER_GRAPH)
        model(pair_batch.clone())
        with torch.no_grad():
            explanation = Compiler.explain(model, pair_batch.clone())
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )

    def test_bind_cell_reaches_both_grids_and_refuses_an_over_capacity_cell(
        self, tiny_polar_spec: MACEPolarSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both GTO grids, the caller's storage, and a loud capacity overflow.

        The two grids are independent objects, so binding one and forgetting
        the other would leave half the reciprocal path unsized and fail only
        at trace time. The cell must arrive as the *same storage* the caller
        holds — MD copies barostat updates into it in place — and the
        ``(1, 3, 3)`` batched layout must take the identical route as ``(3, 3)``.
        An over-capacity cell fails closed and the message names the grid.
        """
        model = MacePolar(tiny_polar_spec, dtype=DTYPE).eval()
        field: list[int] = []
        hartree: list[int] = []
        monkeypatch.setattr(
            model.electric_potential_descriptor,
            "bind_cell",
            lambda cell: field.append(cell.data_ptr()),
        )
        monkeypatch.setattr(
            model.coulomb_energy, "bind_cell", lambda cell: hartree.append(cell.data_ptr())
        )

        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        model.bind_cell(cell.unsqueeze(0))
        assert field == [cell.data_ptr(), cell.data_ptr()]
        assert hartree == field

        monkeypatch.undo()
        model.bind_cell(cell)
        with pytest.raises(RuntimeError) as excinfo:
            model.bind_cell(cell * 2.0)
        assert "KSpaceGrid" in str(excinfo.value)
        assert "capacity_factor" in str(excinfo.value)

    @pytest.mark.slow
    def test_periodic_forward_compiles_to_one_graph(self, tiny_polar_spec: MACEPolarSpec) -> None:
        """Bound grids + one eager forward → the periodic op is 1 graph, 0 breaks.

        Polar's periodic path (``bind_cell`` + the two k-space grids inside
        ``forward``) is its own compile seam, distinct from MatPES's
        ``forward`` traced in ``test_potential.py``.
        """
        model = MacePolar(tiny_polar_spec, dtype=DTYPE).eval()
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.bind_cell(cell)
        model(_periodic_cluster(cell))
        with torch.no_grad():
            explanation = Compiler.explain(model, _periodic_cluster(cell))
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )


class TestPolarKeys:
    """The Polar key table: official-dialect keys land on MacePolar parameters."""

    def test_synthetic_state_fills_parameters(self, tiny_polar_spec: MACEPolarSpec):
        """A native-key dump plus one remapped nested weight loads without unfilled params."""
        model = MacePolar(tiny_polar_spec, dtype=DTYPE)
        native = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_tensor(v)  # ``_extra_state`` is metadata, not a checkpoint tensor
        }
        native = official_layout(model, native)
        official = {}
        for key, value in native.items():
            if key.endswith("readout.linear_mid.linear.weight"):
                official[key.replace(".linear.weight", ".weight")] = value
            elif key.endswith("readout.linear_2.linear.weight"):
                official[key.replace(".linear.weight", ".weight")] = value
            elif key.endswith("linear_up_q.linear.weight"):
                official["local_electron_energy.linear_up_q.weight"] = value
            elif key.endswith("linear_up_v.linear.weight"):
                official["local_electron_energy.linear_up_v.weight"] = value
            else:
                official[key] = value
        official["kspace_cutoff"] = torch.tensor(1.0)
        official["aux_head.weight"] = torch.ones(1)
        missing, unexpected = polar_remap.load(model, official)
        assert not any(k in dict(model.named_parameters()) for k in missing)
        assert "aux_head.weight" in unexpected


class TestSparseUvuTensorProduct:
    """Test the Polar sparse uvu product (l×l→0 and l×0→l)."""

    def test_mixed_precision_inputs_promote(self):
        """fp32 features against an fp64 weight must widen, not raise.

        The reassociated contraction uses ``F.linear`` / ``matmul``, which
        refuse mismatched dtypes where the original ``einsum`` promoted them.
        Polar runs an fp32 backbone with fp64 pieces, so the promotion is
        load-bearing, not a convenience.
        """
        tp = SparseUvuTensorProduct(
            meq.Irreps("O3", "2x0e+2x1o"),
            meq.Irreps("O3", "2x0e+2x1o"),
            meq.Irreps("O3", "2x0e+2x1o"),
            dtype=DTYPE,
        )
        x1 = torch.randn(3, tp.irreps_in1.dim, dtype=torch.float32)
        x2 = torch.randn(3, tp.irreps_in2.dim, dtype=torch.float32)
        out = tp(x1.double(), x2)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out, tp(x1.double(), x2.double()), atol=1e-12, rtol=0)

    def test_l0_contraction_is_product_times_weight(self):
        """W=1, x1=2, x2=3 → 6. path_weight=1 and √(2·0+1)=1."""
        tp = SparseUvuTensorProduct("1x0e", "1x0e", "1x0e", dtype=DTYPE)
        with torch.no_grad():
            tp.weight.fill_(1.0)
        out = tp(
            torch.tensor([[2.0]], dtype=torch.float64),
            torch.tensor([[3.0]], dtype=torch.float64),
        )
        assert out.shape == (1, 1)
        assert out.item() == pytest.approx(6.0)

    def test_two_independent_scalar_copies(self):
        """Identity W: channel u contracts only with channel u.

        The upstream element path weight is ``1/√mul_in2 = 1/√2``.
        """
        tp = SparseUvuTensorProduct("2x0e", "2x0e", "2x0e", dtype=DTYPE)
        with torch.no_grad():
            tp.weight.copy_(torch.eye(2, dtype=torch.float64).reshape(-1))
        out = tp(
            torch.tensor([[2.0, 5.0]], dtype=torch.float64),
            torch.tensor([[3.0, 7.0]], dtype=torch.float64),
        )
        scale = 1.0 / math.sqrt(2.0)
        assert torch.allclose(out, torch.tensor([[6.0 * scale, 35.0 * scale]], dtype=torch.float64))

    def test_vector_dot_is_z_rotation_invariant(self):
        """(1o × 1o → 0e) equals a·b / √3 and survives a z-rotation."""
        tp = SparseUvuTensorProduct("1x1o", "1x1o", "1x0e", dtype=DTYPE)
        with torch.no_grad():
            tp.weight.fill_(1.0)
        torch.manual_seed(0)
        x1 = torch.randn(5, 3, dtype=torch.float64)
        x2 = torch.randn(5, 3, dtype=torch.float64)
        out = tp(x1, x2)
        expect = (x1 * x2).sum(dim=-1, keepdim=True) / math.sqrt(3.0)
        assert torch.allclose(out, expect, atol=1e-12)
        theta = 0.7
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        out_rot = tp(
            rotate_irreps_features_simple(x1, rot, "1x1o"),
            rotate_irreps_features_simple(x2, rot, "1x1o"),
        )
        assert torch.allclose(out, out_rot, atol=1e-10)

    def test_reassociated_dot_matches_literal_pair_and_its_gradients(self):
        """GEMM-first uvu is the literal ``(N,u,v)`` contraction algebraically."""
        torch.manual_seed(4)
        tp = SparseUvuTensorProduct("3x0e+3x1o", "3x0e+3x1o", "3x0e", dtype=DTYPE)
        x1 = torch.randn(7, 12, dtype=torch.float64, requires_grad=True)
        x2 = torch.randn(7, 12, dtype=torch.float64, requires_grad=True)
        actual = tp(x1, x2)

        reference = actual.new_zeros(actual.shape)
        for path in tp._paths:
            s1, e1, s2, e2, so, eo, ws, we, mode, mul1, mul2, d1, path_weight = path
            assert mode == 0
            a = x1[:, s1:e1].reshape(7, d1, mul1).transpose(1, 2)
            b = x2[:, s2:e2].reshape(7, d1, mul2).transpose(1, 2)
            pair = torch.einsum("bud,bvd->buv", a, b)
            weight = tp.weight[ws:we].reshape(mul1, mul2)
            mixed = torch.einsum("buv,uv->bu", pair, weight)
            reference[:, so:eo] += mixed * (path_weight / math.sqrt(float(d1)))
        reference = reference * tp.output_mask

        assert torch.allclose(actual, reference, atol=1e-12, rtol=1e-12)
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (x1, x2, tp.weight), retain_graph=True
        )
        reference_grads = torch.autograd.grad(reference.square().sum(), (x1, x2, tp.weight))
        for observed, expected in zip(actual_grads, reference_grads):
            assert torch.allclose(observed, expected, atol=1e-11, rtol=1e-11)

    def test_dot_mixed_second_derivatives(self):
        """Force training differentiates through feature and parameter gradients."""
        tp = SparseUvuTensorProduct("2x0e+2x1o", "2x0e+2x1o", "2x0e", dtype=DTYPE)
        x1 = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        x2 = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)

        def with_weight(a, b, weight):
            return torch.func.functional_call(tp, {"weight": weight}, (a, b))

        assert torch.autograd.gradgradcheck(with_weight, (x1, x2, tp.weight))

    def test_scalar_modulation_scales_vector(self):
        """(1o × 0e → 1o): path_weight √3 cancels the √(2ℓ+1) kernel factor."""
        tp = SparseUvuTensorProduct("1x1o", "1x0e", "1x1o", dtype=DTYPE)
        with torch.no_grad():
            tp.weight.fill_(1.0)
        vec = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        scalar = torch.tensor([[4.0]], dtype=torch.float64)
        out = tp(vec, scalar)
        assert torch.allclose(out, vec * 4.0, atol=1e-12)

    def test_rejects_non_uvu_paths(self):
        """Only (ℓ×ℓ→0) and (ℓ×0→ℓ); a leftover vector path is refused."""
        with pytest.raises(NotImplementedError, match="uvu"):
            SparseUvuTensorProduct("1x1o", "1x1o", "1x1o", dtype=DTYPE)

    def test_rejects_non_2d_inputs(self):
        """Flattened (N, irreps.dim) only — a 3-D view would mix ir_mul axes."""
        tp = SparseUvuTensorProduct("1x0e", "1x0e", "1x0e", dtype=DTYPE)
        with pytest.raises(ValueError, match="2-D"):
            tp(torch.zeros(2, 1, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64))


class TestMultiLayerFeatureMixer:
    """Test the Polar per-layer feature mixer."""

    def test_forward_is_sum_of_per_layer_linears(self):
        """``(2, N, dim)`` mixes to ``(N, dim)`` as ``W0 h0 + W1 h1``."""
        torch.manual_seed(0)
        irreps = "4x0e+4x1o"
        dim = meq.Irreps("O3", irreps).dim
        mixer = MultiLayerFeatureMixer(irreps=irreps, num_layers=2, dtype=DTYPE)
        stacked = torch.randn(2, 3, dim, dtype=torch.float64)
        out = mixer(stacked)
        expect = mixer.linears[0](stacked[0]) + mixer.linears[1](stacked[1])
        assert out.shape == (3, dim)
        assert torch.allclose(out, expect)

    def test_stack_length_must_match_num_layers(self):
        """A silent truncate would mix the wrong Polar layers."""
        irreps = "4x0e"
        mixer = MultiLayerFeatureMixer(irreps=irreps, num_layers=2, dtype=DTYPE)
        stacked = torch.zeros(3, 1, meq.Irreps("O3", irreps).dim, dtype=torch.float64)
        with pytest.raises(ValueError, match="3 layers.*num_layers=2"):
            mixer(stacked)

    def test_rejects_zero_layers(self):
        """``num_layers >= 1`` — Polar-1-M dump is 2."""
        with pytest.raises(ValueError, match="num_layers"):
            MultiLayerFeatureMixer(irreps="4x0e", num_layers=0, dtype=DTYPE)


class TestEnvironmentDependentSpinSourceBlock:
    """Test the Polar spin-resolved multipole source."""

    def test_forward_view_is_two_y_copies(self):
        """``l_max=1``: flat 8 → ``(N, 2, 4)`` with monopoles at ``[:, :, 0]``."""
        torch.manual_seed(0)
        block = EnvironmentDependentSpinSourceBlock(irreps_in="4x0e+4x1o", l_max=1, dtype=DTYPE)
        n = 3
        x = torch.randn(n, meq.Irreps("O3", "4x0e+4x1o").dim, dtype=torch.float64)
        out = block(x)
        assert out.shape == (n, 2, 4)
        flat = block.linear(x)
        assert flat.shape == (n, 8)
        assert torch.equal(out[:, 0, 0], flat[:, 0])
        assert torch.equal(out[:, 1, 0], flat[:, 4])

    def test_zero_charges_zeros_monopoles_only(self):
        """Upstream ``zero_charges``: 0e slice exact 0, dipoles may stay."""
        torch.manual_seed(1)
        block = EnvironmentDependentSpinSourceBlock(
            irreps_in="4x0e+4x1o",
            l_max=1,
            zero_charges=True,
            dtype=DTYPE,
        )
        x = torch.randn(2, meq.Irreps("O3", "4x0e+4x1o").dim, dtype=torch.float64)
        out = block(x)
        assert out.shape == (2, 2, 4)
        assert torch.equal(out[..., 0], torch.zeros(2, 2, dtype=out.dtype))

    def test_y_dim_is_l_plus_one_squared(self):
        """``dim Y_l = (l+1)^2``; two copies → ``2*(l+1)^2`` flat width."""
        for l_max, y_dim, flat in ((0, 1, 2), (1, 4, 8), (2, 9, 18)):
            block = EnvironmentDependentSpinSourceBlock(irreps_in="2x0e", l_max=l_max, dtype=DTYPE)
            x = torch.zeros(1, meq.Irreps("O3", "2x0e").dim, dtype=torch.float64)
            out = block(x)
            assert out.shape[-1] == y_dim
            assert block.linear(x).shape[-1] == flat


_FEATS = "4x0e+4x1o"
_ATTRS = "2x0e"
_CHARGES = "2x0e+2x1o"
_POTENTIAL = "2x0e+2x1o"
_FEAT_DIM = 16  # 4*1 + 4*3
_CHARGE_DIM = 8  # 2*1 + 2*3
_OUT_DIM = 10  # charges ⊕ 2x0e


def _update():
    return AgnosticEmbeddedOneBodyVariableUpdate(
        node_attrs_irreps=_ATTRS,
        node_feats_irreps=_FEATS,
        potential_irreps=_POTENTIAL,
        charges_irreps=_CHARGES,
        dtype=DTYPE,
    )


def _readout():
    return OneBodyMLPFieldReadout(
        node_feats_irreps=_FEATS,
        potential_irreps=_POTENTIAL,
        charges_irreps=_CHARGES,
        dtype=DTYPE,
    )


class TestAgnosticEmbeddedOneBodyVariableUpdate:
    """Test the Polar non-SCF one-body field update."""

    def test_output_is_charges_plus_two_scalars(self):
        """Dump: charges 2x0e+2x1o (8) ⊕ 2x0e Fukui → 10."""
        torch.manual_seed(0)
        n = 3
        out = _update()(
            torch.randn(n, 2, dtype=torch.float64),
            torch.randn(n, _FEAT_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
        )
        assert out.shape == (n, _OUT_DIM)

    def test_radial_mlp_is_official_width(self):
        """RadialMLP([2 n_0e, 64, 64, 64, n_0e]); tiny n_0e=4."""
        assert _update().nonlinearity.hs == [8, 64, 64, 64, 4]

    def test_readout_gate_width_is_32_times_sh(self):
        """MLP = 32 × Y_{lmax=1} → 32x0e+32x1o (128); gate input 160."""
        assert _update().readout.linear_1.irreps_out.dim == 160

    def test_even_scalars_are_z_rotation_invariant(self):
        """0e slices of charges⊕Fukui stay put under a joint z-rotation."""
        torch.manual_seed(1)
        block = _update()
        n = 4
        node_attrs = torch.randn(n, 2, dtype=torch.float64)
        node_feats = torch.randn(n, _FEAT_DIM, dtype=torch.float64)
        potential = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        charges = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        out = block(node_attrs, node_feats, potential, charges)
        theta = 0.5
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        out_rot = block(
            node_attrs,
            rotate_irreps_features_simple(node_feats, rot, _FEATS),
            rotate_irreps_features_simple(potential, rot, _POTENTIAL),
            rotate_irreps_features_simple(charges, rot, _CHARGES),
        )
        # unsimplified 2x0e+2x1o+2x0e: 0e at [0:2] and [8:10]
        assert torch.allclose(out[:, :2], out_rot[:, :2], atol=1e-8)
        assert torch.allclose(out[:, 8:], out_rot[:, 8:], atol=1e-8)


class TestOneBodyMLPFieldReadout:
    """Test the Polar post-induction field energy readout."""

    def test_output_is_per_atom_scalar(self):
        """Squeeze of RadialMLP(..., 1) → (N,) eV/atom."""
        torch.manual_seed(0)
        n = 3
        out = _readout()(
            torch.randn(n, _FEAT_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
        )
        assert out.shape == (n,)

    def test_uses_sum_of_induced_and_reference_charges(self):
        """q_in = charges_induced + charges_0; swapping them must not change E."""
        torch.manual_seed(2)
        block = _readout()
        n = 2
        feats = torch.randn(n, _FEAT_DIM, dtype=torch.float64)
        field = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        q0 = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        qi = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        a = block(feats, field, q0, qi)
        b = block(feats, field, qi, q0)
        assert torch.allclose(a, b, atol=1e-12)

    def test_mlp_is_official_width(self):
        """RadialMLP([2 n_0e, 128, 128, 128, 1]); tiny n_0e=4."""
        assert _readout().mlp.hs == [8, 128, 128, 128, 1]
