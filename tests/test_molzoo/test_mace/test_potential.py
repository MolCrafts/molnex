"""Tests for molzoo.mace.potential — the MACE products on the potential contract.

The two-method contract itself — ``forward`` is the
:class:`molix.autograd.Derivative` of ``energy`` built from ``wrt``, an empty
``wrt`` writes the energy alone, the frame comes back by handle — is asserted
once on a stub in ``tests/test_molix/test_autograd/test_potential.py``. The
cases here test each product's own physics on that surface — what is read,
what is written, what is derived — once each; the encoder primitives (element
table, invariance of the features, conditioning, the block stack) are
asserted in ``test_encoder.py``, the specs' rules in ``test_spec.py`` and the
key tables in ``test_checkpoint.py``.

Every model is tiny (2 layers, 16 channels, ``l_max=1``), fp64 (autouse
``_build_at_fp64`` in ``conftest.py``), CPU, seeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix import config
from molix.autograd import Derivative
from molix.compile import Compiler
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY, VIRIAL_KEY
from molzoo.mace.potential import MaceLes, MaceMatpes, MaceOff, MaceOmol
from molzoo.mace.spec import MACELesSpec, MACEMatpesSpec, MACEOffSpec
from tests.conftest import make_graph_batch, rotate_graph, rotation_matrix_x
from tests.test_molzoo.test_mace.conftest import (
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    PAIR_Z,
    SINGLE_GRAPH,
    TINY_MATPES_KWARGS,
    full_edge_index,
    official_layout,
)

#: fp64 agreement of one energy evaluated two ways (reduction order only).
ENERGY_ATOL = 1e-10  # eV
FORCE_ATOL = 1e-10  # eV/Å

#: Newton's third law on an isolated cluster (measured 1.4e-17 on this model).
NET_FORCE_ATOL = 1e-10  # eV/Å

#: Fourth-order central difference of the energy: step and tolerances.
#:
#: The truncation term ``h⁴ E⁽⁵⁾ / 30`` is set by the parameter-free ZBL
#: repulsion at the 0.95 Å O–H pair, not by the weights: the second-order
#: stencil at ``h = 1e-4`` sat at 2.8e-7 eV/Å for every seed tried, i.e. on
#: the old 1e-6 gate. With ``E ≈ -3.1e3`` eV the round-off floor is
#: ``ε |E| / h ≈ 7e-10`` eV/Å at this step; the measured worst error over
#: three seeds is 9e-10. ``FD_RTOL`` keeps the gate meaningful should the
#: forces ever be large.
FD_STEP = 1e-3  # Å
FD_ATOL = 1e-8  # eV/Å
FD_RTOL = 1e-7

#: Atom/axis pairs probed by the finite-difference test.
FD_PROBES = ((0, 0), (2, 1), (4, 2))

#: Rotation angle (rad) of the force-equivariance test, about ``x``.
ROTATION_X = -0.4

#: Cubic cell edge (Å) and neighbour cutoff (Å) of the dead-edge padding case.
PBC_CELL = 7.0
PBC_CUTOFF = 3.4

#: A four-atom water-like cluster inside :data:`PBC_CELL`, one atom wrapped
#: across the boundary so the minimum-image path is exercised.
PBC_POS = [
    [0.5, 0.5, 0.5],
    [1.6, 0.6, 0.4],
    [0.4, 1.7, 0.6],
    [6.8, 6.9, 0.2],
]
PBC_Z = [8, 1, 1, 1]

#: Conditioning of the ``charged_omol_cluster``: deliberately **not** the
#: ``(0, 1)`` neutral singlet ``energy`` falls back to.
CHARGED_TOTAL_CHARGE = 1
CHARGED_TOTAL_SPIN = 0

#: Per-graph conditioning of the batched OMOL case: a neutral and an anionic
#: molecule with different spin rows.
PAIR_TOTAL_CHARGE = [0, -1]
PAIR_TOTAL_SPIN = [0, 1]

#: Parameter perturbation of the force-loss case: a fresh MACE readout is
#: zero-initialised, so an unperturbed model has identically zero forces.
PERTURBATION = 0.05
PERTURBATION_SEED = 7

#: An isolated-atom reference large enough that float32 would round the
#: interaction energy away (``2^24`` is the last exactly representable step).
LARGE_E0 = 2.0**24  # eV

FORCES = ("atoms.pos",)
FORCES_AND_VIRIAL = ("atoms.pos", "graphs.strain")


def _frame(
    pos: torch.Tensor,
    Z: torch.Tensor,
    pair_index: torch.Tensor,
    batch: torch.Tensor,
    *,
    edge_diff: torch.Tensor | None = None,
    graphs: dict[str, torch.Tensor] | None = None,
) -> TensorDict:
    """A post-collate batch; ``edge_diff`` overrides the displacement derived from ``pos``."""
    frame = make_graph_batch(pos, Z, pair_index, batch, graphs=graphs)
    if edge_diff is not None:
        frame["pairs", "pair_diff"] = edge_diff
        frame["pairs", "pair_dist"] = edge_diff.norm(dim=-1)
    return frame


def _perturb(model: torch.nn.Module) -> None:
    """Move every parameter off its initialisation, deterministically."""
    torch.manual_seed(PERTURBATION_SEED)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * PERTURBATION)


def _fp32_matpes(**kwargs: object) -> MaceMatpes:
    """A tiny MatPES product built under fp32, the process precision restored after."""
    config.set_precision("fp32")
    try:
        torch.manual_seed(0)
        spec = MACEMatpesSpec(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            **TINY_MATPES_KWARGS,
        )
        return MaceMatpes(spec, wrt=FORCES, **kwargs).eval()
    finally:
        config.set_precision("fp64")


def _write_checkpoint(directory: Path, spec: MACEMatpesSpec, model: MaceMatpes) -> Path:
    """Write a MatPES config json and the weights beside it; return the json path.

    The official → molnex key translation is ``test_checkpoint.py``'s fact,
    so the weights carry the model's own names (the remap is the identity on
    them), in the official layout (``official_layout``), with the element
    table under its official key — the one entry ``load`` reads itself.
    """
    official: dict[str, object] = {
        "atomic_numbers": spec.atomic_numbers,
        "atomic_energies": spec.atomic_energies,
        "r_max": spec.r_max,
        "num_bessel": spec.num_bessel,
        "num_polynomial_cutoff": spec.num_polynomial_cutoff,
        "max_ell": spec.l_max,
        "correlation": spec.correlation,
        "num_interactions": spec.num_interactions,
        "hidden_irreps": f"{spec.num_features}x0e+{spec.num_features}x1o",
        "MLP_irreps": f"{spec.mlp_dim}x0e",
        "radial_MLP": spec.radial_mlp,
        "atomic_inter_scale": spec.scale,
        "atomic_inter_shift": spec.shift,
    }
    config_path = directory / "matpes_config.json"
    config_path.write_text(json.dumps(official))
    state = official_layout(
        model, {k: v for k, v in model.state_dict().items() if torch.is_tensor(v)}
    )
    state["atomic_numbers"] = state.pop("z_table")
    torch.save(state, directory / "matpes_cueq_state.pt")
    return config_path


@pytest.fixture
def charged_omol_cluster() -> TensorDict:
    """The cluster conditioned on a non-default charge/spin pair."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs={
            "total_charge": torch.tensor([CHARGED_TOTAL_CHARGE], dtype=torch.long),
            "total_spin": torch.tensor([CHARGED_TOTAL_SPIN], dtype=torch.long),
        },
    )


class TestMacePotential:
    """The shared product surface, on the MatPES product."""

    def test_invalid_energy_dtype_is_rejected(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        with pytest.raises(ValueError, match="energy_dtype"):
            MaceMatpes(tiny_matpes_spec, energy_dtype=torch.float16)

    def test_invalid_input_dtype_is_rejected(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        with pytest.raises(ValueError, match="input_dtype"):
            MaceMatpes(tiny_matpes_spec, input_dtype=torch.float16)

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
    def test_energy_dtype_selects_the_accumulation(
        self, tiny_matpes_spec: MACEMatpesSpec, cluster: TensorDict, dtype
    ) -> None:
        """Both energy columns land at ``energy_dtype`` (``None``: the model's)."""
        out = MaceMatpes(tiny_matpes_spec, energy_dtype=dtype).eval()(cluster)
        assert out[ENERGY_KEY].dtype == (dtype or torch.float64)
        assert out[ATOMIC_ENERGY_KEY].dtype == (dtype or torch.float64)

    def test_fp32_model_keeps_the_interaction_below_a_large_reference_energy(
        self, cluster: TensorDict
    ) -> None:
        """Each term is cast to ``energy_dtype`` *before* it is added.

        An fp32 model with a ``2^24`` eV isolated-atom reference: the default
        float64 accumulation keeps the learned energy (to fp32 precision) and
        the forces bit-for-bit, where an fp32 sum would have rounded the
        interaction away.
        """
        model = _fp32_matpes()
        frame = cluster.clone()
        frame["atoms", "pos"] = frame["atoms", "pos"].float()
        frame["pairs", "pair_diff"] = frame["pairs", "pair_diff"].float()
        with torch.no_grad():
            model.atomic_energies.atomic_energies.zero_()
        base = model(frame.clone())
        with torch.no_grad():
            model.atomic_energies.atomic_energies.fill_(LARGE_E0)
        shifted = model(frame.clone())
        torch.testing.assert_close(
            shifted[ENERGY_KEY] - len(CLUSTER_Z) * LARGE_E0, base[ENERGY_KEY], rtol=0.0, atol=1e-7
        )
        torch.testing.assert_close(shifted[FORCES_KEY], base[FORCES_KEY], rtol=0.0, atol=0.0)

    def test_forward_applies_the_input_dtype(self, cluster: TensorDict) -> None:
        """fp64 geometry drives an fp32 model built with ``input_dtype=float32``.

        ``graphs.energy`` equals the fp32-geometry evaluation of the same
        model (the cast happens inside ``energy``, so the op's geometry
        rebuild sees it too), and the forces come back at the caller's fp64
        — the cast is differentiable, to fp32 precision.
        """
        model = _fp32_matpes(input_dtype=torch.float32)
        single = make_graph_batch(
            pos=torch.tensor(CLUSTER_POS, dtype=torch.float32),
            Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
            pair_index=full_edge_index(SINGLE_GRAPH),
            batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        )
        expected = model(single)
        actual = model(cluster)
        assert actual[FORCES_KEY].dtype == torch.float64
        torch.testing.assert_close(actual[ENERGY_KEY], expected[ENERGY_KEY], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            actual[FORCES_KEY], expected[FORCES_KEY].double(), atol=1e-6, rtol=1e-6
        )

    def test_atomic_energies_partition_the_graph_energy(
        self, matpes_model: MaceMatpes, pair_batch: TensorDict
    ) -> None:
        """``atoms.energy`` ``(N,)`` scatters onto ``graphs.energy`` ``(B,)`` by membership."""
        out = matpes_model.energy(pair_batch.clone())
        site, energy = out[ATOMIC_ENERGY_KEY], out[ENERGY_KEY]
        assert site.shape == (len(PAIR_Z),)
        assert energy.shape == (2,)
        torch.testing.assert_close(
            torch.zeros_like(energy).index_add_(0, pair_batch["atoms", "batch"], site),
            energy,
            atol=ENERGY_ATOL,
            rtol=0.0,
        )

    def test_energy_requires_the_graphs_namespace(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """``B`` is ``graphs.batch_size[0]``, never a host-syncing fallback."""
        del cluster["graphs"]
        with pytest.raises(KeyError):
            matpes_model.energy(cluster)

    def test_readout_stack_is_linear_then_one_non_linear_last(
        self, tiny_off_spec: MACEOffSpec
    ) -> None:
        """Upstream MACE: readout 0 is linear; the non-linear one is the last of *several*.

        On the OFF spec because MatPES refuses a single interaction.
        """
        two = MaceOff(tiny_off_spec)
        assert [type(r).__name__ for r in two.readouts] == ["LinearReadout", "NonLinearReadout"]
        one = MaceOff(tiny_off_spec.model_copy(update={"num_interactions": 1}))
        assert [type(r).__name__ for r in one.readouts] == ["LinearReadout"]

    def test_a_padded_batch_returns_the_unpadded_energy_and_forces(
        self, matpes_model: MaceMatpes, tiny_matpes_spec: MACEMatpesSpec, cluster: TensorDict
    ) -> None:
        """``MoleculePadder`` ghosts add nothing: every per-atom term is multiplied by the mask."""
        from molix.data import MoleculePadder

        n_atoms = int(cluster["atoms"].batch_size[0])
        n_pairs = int(cluster["pairs"].batch_size[0])
        reference = matpes_model(cluster.clone())
        padded = MoleculePadder(
            max_atoms=n_atoms + 3, max_pairs=n_pairs + 7, cutoff=tiny_matpes_spec.r_max
        )(cluster.clone())
        out = matpes_model(padded)
        torch.testing.assert_close(
            out[ENERGY_KEY], reference[ENERGY_KEY], atol=ENERGY_ATOL, rtol=0.0
        )
        torch.testing.assert_close(
            out[FORCES_KEY][:n_atoms], reference[FORCES_KEY], atol=FORCE_ATOL, rtol=0.0
        )
        assert torch.equal(out[ATOMIC_ENERGY_KEY][n_atoms:], torch.zeros(3, dtype=torch.float64))
        assert torch.equal(out[FORCES_KEY][n_atoms:], torch.zeros(3, 3, dtype=torch.float64))

    def test_energy_is_extensive_over_separated_graphs(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """Two non-interacting copies in one batch carry one copy's energy each."""
        one = matpes_model(cluster.clone())[ENERGY_KEY]
        n_atoms = len(CLUSTER_Z)
        pair = _frame(
            torch.cat([cluster["atoms", "pos"]] * 2),
            torch.cat([cluster["atoms", "Z"]] * 2),
            torch.cat([cluster["pairs", "pair_index"], cluster["pairs", "pair_index"] + n_atoms]),
            torch.cat(
                [torch.zeros(n_atoms, dtype=torch.long), torch.ones(n_atoms, dtype=torch.long)]
            ),
        )
        torch.testing.assert_close(
            matpes_model(pair)[ENERGY_KEY], one.repeat(2), atol=ENERGY_ATOL, rtol=0.0
        )

    # -- physics --------------------------------------------------------------

    def test_net_force_vanishes(self, matpes_model: MaceMatpes, cluster: TensorDict) -> None:
        """Newton's third law: the energy depends on ``pos`` only through ``r_ij``."""
        forces = matpes_model(cluster)[FORCES_KEY]
        assert float(forces.sum(0).abs().max()) <= NET_FORCE_ATOL

    def test_forces_rotate_with_the_system(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """Forces are equivariant: ``F(Rx) = R F(x)``."""
        rotation = rotation_matrix_x(ROTATION_X, torch.float64)
        base = matpes_model(cluster.clone())[FORCES_KEY]
        turned = matpes_model(rotate_graph(cluster, rotation))[FORCES_KEY]
        torch.testing.assert_close(turned, base @ rotation.T, atol=FORCE_ATOL, rtol=0.0)

    def test_forces_match_central_differences(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """``F = -dE/dx`` against a fourth-order central difference of ``energy``.

        See :data:`FD_STEP` for why the stencil is fourth-order: the
        second-order one's truncation error on the ZBL term sat on the gate.
        """
        forces = matpes_model(cluster.clone())[FORCES_KEY]
        pos = cluster["atoms", "pos"]
        Z, pair_index, batch = (
            cluster["atoms", "Z"],
            cluster["pairs", "pair_index"],
            cluster["atoms", "batch"],
        )

        def total(atom: int, axis: int, steps: float) -> float:
            shifted = pos.clone()
            shifted[atom, axis] += steps * FD_STEP
            with torch.no_grad():
                return float(
                    matpes_model.energy(_frame(shifted, Z, pair_index, batch))[ENERGY_KEY].sum()
                )

        for atom, axis in FD_PROBES:
            numerical = -(
                -total(atom, axis, 2.0)
                + 8.0 * total(atom, axis, 1.0)
                - 8.0 * total(atom, axis, -1.0)
                + total(atom, axis, -2.0)
            ) / (12.0 * FD_STEP)
            assert float(forces[atom, axis]) == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)

    def test_virial_matches_central_differences_of_the_strained_energy(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """``W = ∂E/∂ε`` against a finite difference of this model's own energy.

        The reference strains the same tensors the derivation strains,
        ``v @ (I + ε_sym)ᵀ``, and differences the energy centrally in fp64.
        Cauchy convention: no doubled shear, so ``σ = W / V``.
        """
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        cluster["graphs", "cell"] = cell.unsqueeze(0)
        npt = Derivative(matpes_model.energy, wrt=FORCES_AND_VIRIAL)
        with torch.no_grad():
            virial = npt(cluster.clone())[VIRIAL_KEY][0]

        pos = cluster["atoms", "pos"].detach()
        edge_diff = cluster["pairs", "pair_diff"].detach()
        eps = 2.0e-4

        def strained_energy(delta: torch.Tensor) -> float:
            frame = cluster.clone()
            transform = (torch.eye(3, dtype=torch.float64) + 0.5 * (delta + delta.T)).T
            frame["atoms", "pos"] = pos @ transform
            frame["graphs", "cell"] = (cell @ transform).unsqueeze(0)
            # The energy reads the imaged edge displacements, so the reference
            # strains exactly what the derivation strains.
            frame["pairs", "pair_diff"] = edge_diff @ transform
            with torch.no_grad():
                return float(matpes_model.energy(frame)[ENERGY_KEY].sum())

        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2)):
            delta = torch.zeros(3, 3, dtype=torch.float64)
            delta[i, j] = eps
            fd = (strained_energy(delta) - strained_energy(-delta)) / (2 * eps)
            assert virial[i, j].item() == pytest.approx(fd, abs=2e-4, rel=1e-3)

    def test_energy_and_forces_ignore_dead_edges(self, matpes_model: MaceMatpes) -> None:
        """Same system, 0% vs heavy neighbour-list padding: identical energy and forces.

        The load-bearing property of :class:`molix.md.NeighborList`: its
        fixed-capacity buffers pad with self-loops on atom 0 displaced beyond
        the cutoff, and the whole rebuild-under-CUDA-graphs design assumes
        such edges are invisible to the model. Random weights make the check
        stronger — the zeros must come from the cutoff structure.
        """
        from molix.md import NeighborList

        cell = torch.eye(3, dtype=torch.float64) * PBC_CELL
        pos = torch.tensor(PBC_POS, dtype=torch.float64)
        Z = torch.tensor(PBC_Z, dtype=torch.long)
        batch = torch.zeros(len(PBC_Z), dtype=torch.long)
        outs = []
        for factor in (1.0, 3.0):
            nl = NeighborList(cell=cell, cutoff=PBC_CUTOFF, positions=pos, capacity_factor=factor)
            edge_diff = pos[nl.pair_index[:, 1]] - pos[nl.pair_index[:, 0]] + nl.shifts
            out = matpes_model(_frame(pos, Z, nl.pair_index, batch, edge_diff=edge_diff))
            outs.append((nl, out[ENERGY_KEY].clone(), out[FORCES_KEY].clone()))
        (nl_a, energy_a, forces_a), (nl_b, energy_b, forces_b) = outs
        assert nl_b.capacity > nl_a.capacity  # the padded arm really is padded
        assert torch.equal(energy_a, energy_b)
        assert torch.equal(forces_a, forces_b)

    # -- compile --------------------------------------------------------------

    @pytest.mark.slow
    def test_forward_compiles_to_one_graph_in_inference(
        self, matpes_model: MaceMatpes, cluster: TensorDict
    ) -> None:
        """The op is the compile boundary; one eager forward spends the element gate first."""
        matpes_model(cluster)
        with torch.no_grad():
            explanation = Compiler.explain(matpes_model, cluster)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )


class TestMaceMatpes:
    """``MaceMatpes.load`` — json + weights in one call, through the typed config."""

    def test_load_builds_from_config_and_weights(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """``load`` reads the archive, builds the spec, constructs with the kwargs, remaps in."""
        torch.manual_seed(0)
        source = MaceMatpes(tiny_matpes_spec)
        config_path = _write_checkpoint(tmp_path, tiny_matpes_spec, source)
        torch.manual_seed(1)  # a different init, so the equality below is the checkpoint's doing
        built = MaceMatpes.load(config_path, wrt=FORCES, energy_dtype=None)
        assert built.energy_dtype is None
        assert built.derivative.wrt == (("atoms", "pos"),)
        for (name, want), (_, got) in zip(
            source.named_parameters(), built.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_load_rejects_a_checkpoint_whose_element_table_disagrees(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """The config's table builds ``E0``; the checkpoint's would misalign it."""
        torch.manual_seed(0)
        config_path = _write_checkpoint(tmp_path, tiny_matpes_spec, MaceMatpes(tiny_matpes_spec))
        official = json.loads(config_path.read_text())
        official["atomic_numbers"] = [1, 6, 7]
        config_path.write_text(json.dumps(official))
        with pytest.raises(ValueError, match="atomic_numbers"):
            MaceMatpes.load(config_path)


class TestMaceOmol:
    def test_the_total_charge_moves_the_energy(
        self, omol_model: MaceOmol, omol_cluster: TensorDict, charged_omol_cluster: TensorDict
    ) -> None:
        """``energy`` reads ``graphs.{total_charge,total_spin}`` into the joint embedding."""
        neutral = omol_model(omol_cluster)[ENERGY_KEY]
        charged = omol_model(charged_omol_cluster)[ENERGY_KEY]
        assert float((charged - neutral).abs().max()) > 1e-6

    def test_per_graph_conditioning_routes_to_each_graph(
        self, omol_model: MaceOmol, pair_batch: TensorDict
    ) -> None:
        """Two molecules with their own charge/spin price as they would alone."""
        pair_batch["graphs", "total_charge"] = torch.tensor(PAIR_TOTAL_CHARGE, dtype=torch.long)
        pair_batch["graphs", "total_spin"] = torch.tensor(PAIR_TOTAL_SPIN, dtype=torch.long)
        batched = omol_model(pair_batch.clone())[ENERGY_KEY]
        membership = pair_batch["atoms", "batch"]
        for graph in (0, 1):
            atoms = membership == graph
            pair_index = pair_batch["pairs", "pair_index"]
            edges = atoms[pair_index[:, 0]]
            offset = int(atoms.nonzero()[0])
            alone = _frame(
                pair_batch["atoms", "pos"][atoms],
                pair_batch["atoms", "Z"][atoms],
                pair_index[edges] - offset,
                torch.zeros(int(atoms.sum()), dtype=torch.long),
                graphs={
                    "total_charge": torch.tensor([PAIR_TOTAL_CHARGE[graph]], dtype=torch.long),
                    "total_spin": torch.tensor([PAIR_TOTAL_SPIN[graph]], dtype=torch.long),
                },
            )
            torch.testing.assert_close(
                batched[graph : graph + 1],
                omol_model(alone)[ENERGY_KEY],
                atol=ENERGY_ATOL,
                rtol=0.0,
            )

    def test_force_loss_through_forward_reaches_the_parameters(
        self, omol_model: MaceOmol, charged_omol_cluster: TensorDict
    ) -> None:
        """Force-supervised training backprops through the op's own ``autograd.grad``.

        Regression for the pre-cutover model gating ``create_graph`` on
        ``self.training``: in eval mode the force was detached from the
        parameter graph (spec ``cuet-force-doublebackward``). The readout
        is perturbed first so the fresh model's forces are not identically
        zero.
        """
        assert not omol_model.training
        _perturb(omol_model)
        forces = omol_model(charged_omol_cluster)[FORCES_KEY]
        assert forces.abs().max() > 0.0
        omol_model.zero_grad(set_to_none=True)
        (forces**2).mean().backward()
        parameters = list(omol_model.parameters())
        with_grad = sum(
            int(p.grad is not None and float(p.grad.abs().sum()) > 0.0) for p in parameters
        )
        assert with_grad > len(parameters) // 2, (
            f"force loss reached only {with_grad}/{len(parameters)} parameters"
        )


class TestMaceOff:
    def test_off_carries_no_pair_repulsion(self, tiny_off_spec: MACEOffSpec) -> None:
        """Standard MACE: the ZBL term is MatPES's; the OFF spec switches it off."""
        assert MaceOff(tiny_off_spec).pair_repulsion is None


class TestMaceLes:
    def test_les_long_range_lands_on_graphs_energy(
        self, tiny_les_spec: MACELesSpec, cluster: TensorDict
    ) -> None:
        """``atoms.energy`` is the MACE-OFF site energy; ``graphs.energy`` adds the Ewald sum."""
        torch.manual_seed(0)
        les = MaceLes(tiny_les_spec).eval()
        off = MaceOff(tiny_les_spec).eval()
        off.load_state_dict(
            {k: v for k, v in les.state_dict().items() if not k.startswith("charge_readouts.")}
        )
        assert [type(r).__name__ for r in les.charge_readouts] == [
            "LinearReadout",
            "NonLinearReadout",
        ]
        out = les.energy(cluster.clone())
        short_range = off.energy(cluster.clone())
        torch.testing.assert_close(
            out[ATOMIC_ENERGY_KEY], short_range[ATOMIC_ENERGY_KEY], atol=ENERGY_ATOL, rtol=0.0
        )
        long_range = out[ENERGY_KEY] - short_range[ENERGY_KEY]
        assert float(long_range.abs()) > 1e-6
        torch.testing.assert_close(
            out[ATOMIC_ENERGY_KEY].sum().reshape(1) + long_range,
            out[ENERGY_KEY],
            atol=ENERGY_ATOL,
            rtol=0.0,
        )
