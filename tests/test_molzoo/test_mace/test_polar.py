"""Tests for MacePolar and polar_remap."""

from __future__ import annotations

import pytest
import torch

from molix import config
from molix.compile import Compiler
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar
from molzoo.mace.spec import MACEPolarSpec
from tests.test_molzoo.test_mace.conftest import (
    CLUSTER_POS,
    CLUSTER_Z,
    SINGLE_GRAPH,
    full_edge_index,
    polar_kwargs,
)

#: Seeded tiny-Polar ``energy_core`` energies (eV) — hard-coded fp64 literals
#: captured from *this* tree before the fixed-capacity k-grid rewrite (molnex @
#: ``feat/oplsaa-parameterizer``, ``PYTHONPATH=src python
#: capture_kgrid_goldens.py``, 2026-09-03). Construction is
#: ``torch.manual_seed(0)`` → ``MacePolar(**polar_kwargs(tiny_polar_spec)).eval()``
#: on :data:`CLUSTER_POS` / :data:`CLUSTER_Z`.
#: No live oracle.
TINY_POLAR_PERIODIC10_ENERGY_SEED0 = -2860.720879072809
TINY_POLAR_ISOLATED_ENERGY_SEED0 = -2829.8252847964013


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


polar_remap = CheckpointRemap(polar_keys, on_unexpected="return")


class TestMacePolar:
    """Isolated-molecule Polar energy on the tiny Polar backbone."""

    def test_official_runtime_cutoff_overrides_constructor_heuristic(self, tiny_polar_spec):
        model = MacePolar(
            **polar_kwargs(tiny_polar_spec, kspace_cutoff=1.234, kspace_cutoff_factor=1.5)
        )
        assert model.electric_potential_descriptor.k_cutoff == 1.234
        assert model.coulomb_energy.k_cutoff == 1.234

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
    def test_energy_precision_covers_hartree(self, tiny_polar_spec, omol_cluster, dtype):
        model = MacePolar(**polar_kwargs(tiny_polar_spec, energy_dtype=dtype))
        energy = model(omol_cluster)["graphs", "energy"]
        assert energy.dtype == (dtype or torch.float64)

    def test_split_precision_keeps_md_geometry_and_force_dtype(self, tiny_polar_spec, omol_cluster):
        model = MacePolar(**polar_kwargs(tiny_polar_spec, wrt=("atoms.pos",))).float().eval()
        model.configure_input_dtype(torch.float32)
        pos = omol_cluster["atoms", "pos"]
        before = pos.clone()
        output = model(omol_cluster)
        assert output["atoms", "forces"].dtype == torch.float64
        assert torch.isfinite(output["atoms", "forces"]).all()
        torch.testing.assert_close(pos, before, atol=0, rtol=0)

    def test_official_submodule_names(self, tiny_polar_spec: MACEPolarSpec):
        """Keys polar_remap copies by name."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec))
        names = set(model._modules)
        assert {
            "lr_source_maps",
            "layer_feature_mixer",
            "fukui_source_map",
            "field_dependent_charges_maps",
            "local_electron_energy",
            "electric_potential_descriptor",
            "coulomb_energy",
        } <= names
        assert len(model.lr_source_maps) == tiny_polar_spec.num_interactions
        assert len(model.field_dependent_charges_maps) == tiny_polar_spec.num_recursion_steps

    def test_energy_is_finite_and_per_graph(self, tiny_polar_spec: MACEPolarSpec, omol_cluster):
        """Neutral singlet cluster: one finite eV."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec))
        out = model(omol_cluster)
        energy = out["graphs", "energy"]
        assert energy.shape == (1,)
        assert torch.isfinite(energy).all()

    def test_energy_is_translation_invariant(self, tiny_polar_spec: MACEPolarSpec, omol_cluster):
        """Isolated electrostatics + local MACE: E(r+Δ)=E(r)."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec))
        e0 = model(omol_cluster.clone())["graphs", "energy"].clone()
        shifted = omol_cluster.clone()
        shifted["atoms", "pos"] = shifted["atoms", "pos"] + 2.5
        e1 = model(shifted)["graphs", "energy"]
        assert torch.allclose(e0, e1, atol=1e-8)

    def test_a_padded_batch_returns_the_unpadded_energy_and_forces(
        self, tiny_polar_spec: MACEPolarSpec, omol_cluster
    ):
        """``MoleculePadder`` ghosts are inert through the long-range terms too.

        Polar's per-atom energies (``e0``, the readouts, the local electron
        energy) are masked like MatPES's, and the ghost atoms' densities and
        Fukui sources are masked as well, so the charge equilibration, the
        field descriptor and the Hartree sum never see a ghost charge. The
        cluster is translated off the origin first: a padded ghost sits at
        ``(0, 0, 0)``, and a real atom coincident with it would exercise the
        kernels' small-``r`` clamp rather than the mask.
        """
        from molix.data import MoleculePadder
        from molix.schema import ENERGY_KEY, FORCES_KEY

        torch.manual_seed(0)
        model = MacePolar(**polar_kwargs(tiny_polar_spec, wrt=("atoms.pos",))).eval()
        cluster = omol_cluster.clone()
        cluster["atoms", "pos"] = cluster["atoms", "pos"] + 2.5
        n_atoms = int(cluster["atoms"].batch_size[0])
        n_pairs = int(cluster["pairs"].batch_size[0])
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

    def test_fp32_energy_keeps_interaction_below_large_atomic_baseline(
        self, tiny_polar_spec: MACEPolarSpec
    ):
        """Adding a large constant E0 cannot round away the learned energy."""
        config.set_precision("fp32")
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float32, requires_grad=True)
        z = torch.tensor(CLUSTER_Z, dtype=torch.long)
        edges = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(CLUSTER_Z), dtype=torch.long)
        with torch.no_grad():
            model.atomic_energies.atomic_energies.zero_()
        base = model.energy_core(pos, z, edges, batch, num_graphs=1)
        force0 = torch.autograd.grad(base.sum(), pos)[0]
        with torch.no_grad():
            model.atomic_energies.atomic_energies.fill_(2**24)
        shifted = model.energy_core(pos, z, edges, batch, num_graphs=1)
        force1 = torch.autograd.grad(shifted.sum(), pos)[0]
        torch.testing.assert_close(shifted - len(CLUSTER_Z) * 2**24, base, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(force1, force0, rtol=0.0, atol=0.0)

    def test_self_interaction_flags_reach_both_electrostatic_blocks(
        self, tiny_polar_spec: MACEPolarSpec
    ):
        """Hartree keeps the on-site term by default, field features drop it.

        The two blocks answer different questions (energy vs per-atom
        potential) and the dump sets them independently, so one flag each —
        wired straight through :class:`~molzoo.mace.spec.MACEPolarSpec`.
        """
        model = MacePolar(**polar_kwargs(tiny_polar_spec))
        assert model.coulomb_energy.include_self_interaction is True
        assert model.electric_potential_descriptor.include_self_interaction is False
        flipped = MacePolar(
            **polar_kwargs(
                tiny_polar_spec,
                electrostatic_self_interaction=False,
                field_self_interaction=True,
            )
        )
        assert flipped.coulomb_energy.include_self_interaction is False
        assert flipped.electric_potential_descriptor.include_self_interaction is True


class TestPOLARRemap:
    """Official-dialect keys land on MacePolar parameters."""

    def test_synthetic_state_fills_parameters(self, tiny_polar_spec: MACEPolarSpec):
        """A native-key dump plus one remapped nested weight loads without unfilled params."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec))
        native = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_tensor(v)  # ``_extra_state`` is metadata, not a checkpoint tensor
        }
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


class TestMacePolarKspaceCompile:
    """``energy_core`` threads ``num_graphs`` / ``cell`` into both GTO blocks."""

    def test_energy_core_forwards_num_graphs_to_both_gto_blocks(
        self, tiny_polar_spec: MACEPolarSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One coulomb call and two descriptor calls per recursion step.

        No call may fall back to ``int(batch.max().item())`` — the pipeline is
        broken today at ``polar.py:241-246`` / ``:266``, which pass ``cell``
        only.
        """
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
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

        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        atomic_numbers = torch.tensor(CLUSTER_Z, dtype=torch.long)
        edge_index = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(CLUSTER_Z), dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        with torch.no_grad():
            model.energy_core(pos, atomic_numbers, edge_index, batch, 1, cell=cell)

        assert len(coulomb_calls) == 1
        assert len(descriptor_calls) == 2 * tiny_polar_spec.num_recursion_steps
        for kwargs in coulomb_calls + descriptor_calls:
            assert kwargs.get("num_graphs") == 1
            assert kwargs.get("cell") is not None

    @pytest.mark.slow
    def test_periodic_energy_core_compiles_to_one_graph(
        self, tiny_polar_spec: MACEPolarSpec
    ) -> None:
        """Bound grids + explicit ``num_graphs`` → 1 graph, 0 breaks."""
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        atomic_numbers = torch.tensor(CLUSTER_Z, dtype=torch.long)
        edge_index = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(CLUSTER_Z), dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        model.coulomb_energy.bind_cell(cell)
        model.electric_potential_descriptor.bind_cell(cell)
        explanation = Compiler.explain(
            model.energy_core, pos, atomic_numbers, edge_index, batch, 1, cell=cell
        )
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0

    def test_periodic_energy_matches_pre_change_literal(
        self, tiny_polar_spec: MACEPolarSpec
    ) -> None:
        """fp64 tiny-Polar energy in a 10 Å cubic cell is unchanged."""
        torch.manual_seed(0)
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        atomic_numbers = torch.tensor(CLUSTER_Z, dtype=torch.long)
        edge_index = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(CLUSTER_Z), dtype=torch.long)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        with torch.no_grad():
            energy = model.energy_core(pos, atomic_numbers, edge_index, batch, 1, cell=cell)
        assert energy.item() == pytest.approx(TINY_POLAR_PERIODIC10_ENERGY_SEED0, abs=1e-10)

    def test_isolated_energy_matches_pre_change_literal(
        self, tiny_polar_spec: MACEPolarSpec
    ) -> None:
        """fp64 tiny-Polar isolated energy is unchanged by the k-grid work."""
        torch.manual_seed(0)
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        atomic_numbers = torch.tensor(CLUSTER_Z, dtype=torch.long)
        edge_index = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(CLUSTER_Z), dtype=torch.long)
        with torch.no_grad():
            energy = model.energy_core(pos, atomic_numbers, edge_index, batch, 1)
        assert energy.item() == pytest.approx(TINY_POLAR_ISOLATED_ENERGY_SEED0, abs=1e-10)


class TestMacePolarBindCell:
    """The override that gives the base class's no-op something to do."""

    def test_bind_cell_reaches_both_grids_and_refuses_an_over_capacity_cell(
        self, tiny_polar_spec: MACEPolarSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both GTO grids, the caller's storage, and a loud capacity overflow.

        The two grids are independent objects (``polar.py`` builds the field
        descriptor and the Hartree energy separately), so binding one and
        forgetting the other would leave half the reciprocal path unsized and
        fail only at trace time. The cell must arrive as the *same storage* the
        caller holds — MD copies barostat updates into it in place — and the
        ``(1, 3, 3)`` batched layout must take the identical route as ``(3, 3)``.
        An over-capacity cell fails closed and the message names the grid, so
        the k-space overflow is never mistaken for the neighbour list's.
        """
        model = MacePolar(**polar_kwargs(tiny_polar_spec)).eval()
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
