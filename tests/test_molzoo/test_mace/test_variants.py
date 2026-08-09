"""Tests for molzoo.mace.variants — the two named foundation models.

:class:`~molzoo.mace.variants.MACEMatpes` and
:class:`~molzoo.mace.variants.MACEOMol` are **existing public API**: they are
constructed by keyword in ``scripts/matpes_port/run_nve.py:148-164`` and
``benchmarks/bench_mace_matpes.py:41-54``, which the
``mace-subpackage-restructure-06-wire`` cutover must not touch. After the
cutover they are thin adapters — old keyword signature in, 04/05's spec presets
and :class:`~molzoo.mace.potential.MACEPotential` out — so what this file pins
is the surface those callers bind, plus everything that is genuinely the
variants' own: :meth:`MACEMatpes.energy_forces` /
:meth:`MACEOMol.energy_forces` exist **only** here (``MACEPotential`` has
``forward`` and ``energy_core``), so the raw-tensor physics belongs in this
module and not in ``test_potential.py``.

The load-bearing oddity is ``_compute_energy``: a **private** method consumed
across packages (``run_nve.py:118`` binds it as the compiled energy core,
``bench_mace_matpes.py``'s compiled arm compiles it). Re-pointing that consumer at the
public :meth:`~molzoo.mace.potential.MACEPotential.energy_core` belongs to
``mace-subpackage-restructure-07-cleanup``; until then the positional signature
``(pos, Z, edge_index, batch, num_graphs, shifts)`` is a contract, and the case
below is what keeps the NVE script alive on cutover day.

Migration note (``mace-subpackage-restructure-06-wire``): the cases below the
``-- migrated from tests/test_molzoo/test_mace_{matpes,omol}.py --`` markers
come from the two pre-cutover flat test files, deleted by that step. Their
assertion criteria and tolerances are preserved verbatim; what changed is the
import (``molzoo.mace.variants`` instead of the flat modules), the geometry
(the package's fixed :data:`~tests.test_molzoo.test_mace.conftest.CLUSTER_POS`
instead of a seeded ``randn`` cloud) and, for OMOL, the precision — the flat
OMOL file built at fp32 and called ``.double()``, which leaves cuEquivariance
contracting in fp32; the package's autouse ``fp64`` fixture builds at fp64, so
the same tolerances are strictly harder to meet.

Every model here is tiny (2 layers, 16 channels, ``l_max=1``), fp64 (autouse
``fp64`` fixture in ``conftest.py``), CPU and seeded.
"""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from molzoo.mace.potential import MACEPotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec
from molzoo.mace.variants import (
    MACEMatpes,
    MACEOMol,
    load_matpes_state_dict,
    load_omol_state_dict,
)
from tests.conftest import make_graph_batch
from tests.test_molzoo.test_mace.conftest import (
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    PAIR_Z,
    SINGLE_GRAPH,
    TINY_MATPES_KWARGS,
    TINY_OMOL_KWARGS,
    full_edge_index,
    raw_tensors,
)

#: Iron (26) is outside the ``[1, 6, 8]`` table — ``searchsorted`` would snap it
#: onto a neighbouring row and return a plausible, wrong energy.
OFF_TABLE_Z = [26]

#: The same probe inside a whole five-atom system, for the ``forward`` gate.
OFF_TABLE_CLUSTER_Z = [1, 6, 8, 1, 26]

#: ``scale`` / ``shift`` are the two ``run_nve.py`` kwargs the tiny preset omits;
#: pinned to non-default values so the construction case really passes all 15.
SCALE = 1.5
SHIFT = -0.25

#: OMOL's charge/spin conditioning sizes, the four kwargs outside the tiny
#: preset. Small on purpose: the embedding tables are ``num_classes``-sized.
CHARGE_CLASSES = 7
CHARGE_OFFSET = 3
SPIN_CLASSES = 5
SPIN_OFFSET = 0

#: A checkpoint key that exists in no MACE dialect — the loaders must refuse it
#: rather than quietly leaving the model at its initialisation.
ALIEN_CHECKPOINT: dict[str, torch.Tensor] = {"not.a.mace.key": torch.zeros(1)}

#: Same class, same weights, same operator order: the kwargs form and the spec
#: form are one code path, so their energies must agree *bitwise*, not closely.
EXACT = 0.0

#: ``forward(batch)`` (batch schema, shared force kernel) against
#: ``energy_forces(...)`` (raw tensors, own position leaf) — two code paths on
#: one energy. Criteria carried over verbatim from the flat test files.
FORWARD_ENERGY_ATOL = 1e-9  # eV
FORWARD_FORCE_ATOL = 1e-8  # eV/Å

#: Newton's third law on an isolated system, as the two flat files asserted it.
MATPES_NET_FORCE_ATOL = 1e-8  # eV/Å
OMOL_NET_FORCE_ATOL = 1e-7  # eV/Å

#: Rotation angles (rad) of the two rigid test rotations: ``Rz`` for the energy
#: invariance, ``Rx`` for the force equivariance (flat file's own choices).
ROTATION_Z = 0.7
ROTATION_X = -0.4

#: Central-difference step and tolerance of the finite-difference force check.
FD_STEP = 1e-6  # Å
FD_ATOL = 1e-5  # eV/Å

#: Atom/axis pairs probed by the finite-difference case.
FD_PROBES = ((0, 0), (2, 1), (4, 2))

#: Cubic cell edge (Å) and neighbour cutoff (Å) of the dead-edge padding cases.
PBC_CELL = 7.0
PBC_CUTOFF = 3.4

#: Capacity factors of the un-padded and heavily padded neighbour lists.
PBC_CAPACITY_FACTORS = (1.0, 3.0)

#: A four-atom water-like cluster inside :data:`PBC_CELL`, one atom wrapped
#: across the boundary so the minimum-image path is exercised.
PBC_POS = [
    [0.5, 0.5, 0.5],
    [1.6, 0.6, 0.4],
    [0.4, 1.7, 0.6],
    [6.8, 6.9, 0.2],
]
PBC_Z = [8, 1, 1, 1]

#: The rebuild case drops the wrapped atom — three atoms, no periodic image.
REBUILD_CAPACITY_FACTOR = 2.0

#: Per-graph conditioning of the batched OMOL case: a neutral and an anionic
#: molecule with different spin rows.
PAIR_TOTAL_CHARGE = [0, -1]
PAIR_TOTAL_SPIN = [0, 1]

#: Conditioning of the ``charged_omol_cluster``: deliberately **not** the
#: ``(0, 1)`` neutral singlet ``forward`` falls back to, so a ``forward`` that
#: ignored ``graphs`` could not pass the consistency cases.
CHARGED_TOTAL_CHARGE = 1
CHARGED_TOTAL_SPIN = 0

#: Parameter perturbation of the force-loss cases: a fresh MACE readout is
#: zero-initialised, so an unperturbed model has identically zero forces and no
#: force loss can reach anything (a vacuous pass).
PERTURBATION = 0.05
PERTURBATION_SEED = 7


def _rotation_z(angle: float) -> torch.Tensor:
    """Right-handed rotation about ``z`` by ``angle`` rad, fp64."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _rotation_x(angle: float) -> torch.Tensor:
    """Right-handed rotation about ``x`` by ``angle`` rad, fp64."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


def _perturb(model: torch.nn.Module) -> None:
    """Move every parameter off its initialisation, deterministically."""
    torch.manual_seed(PERTURBATION_SEED)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * PERTURBATION)


def _fraction_with_gradient(model: torch.nn.Module) -> tuple[int, int]:
    """``(parameters carrying a non-zero grad, total parameters)``."""
    parameters = list(model.parameters())
    with_grad = sum(int(p.grad is not None and float(p.grad.abs().sum()) > 0.0) for p in parameters)
    return with_grad, len(parameters)


@pytest.fixture
def charged_omol_cluster() -> TensorDict:
    """The cluster conditioned on a non-default charge/spin pair."""
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs={
            "total_charge": torch.tensor([CHARGED_TOTAL_CHARGE], dtype=torch.long),
            "total_spin": torch.tensor([CHARGED_TOTAL_SPIN], dtype=torch.long),
        },
    )


@pytest.fixture
def omol_pair_batch(pair_batch: TensorDict) -> TensorDict:
    """Two molecules in one batch, each with its own charge and spin."""
    pair_batch["graphs", "total_charge"] = torch.tensor(PAIR_TOTAL_CHARGE, dtype=torch.long)
    pair_batch["graphs", "total_spin"] = torch.tensor(PAIR_TOTAL_SPIN, dtype=torch.long)
    return pair_batch


class TestMACEMatpes:
    """The keyword surface ``run_nve.py`` / ``bench_mace_matpes.py`` bind."""

    def test_constructs_from_the_nve_script_keywords(self) -> None:
        """All 15 ``run_nve.py:148-164`` keywords, ``scale`` / ``shift`` included."""
        torch.manual_seed(0)
        model = MACEMatpes(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=torch.tensor(ATOMIC_ENERGIES),
            scale=SCALE,
            shift=SHIFT,
            **TINY_MATPES_KWARGS,
        )
        assert isinstance(model, torch.nn.Module)

    def test_exposes_the_consumed_method_surface(self, matpes_variant: MACEMatpes) -> None:
        """``forward`` / ``energy_forces`` / ``validate_elements`` all survive."""
        assert all(
            callable(getattr(matpes_variant, name, None))
            for name in ("forward", "energy_forces", "validate_elements")
        )

    def test_registers_the_element_table_as_a_buffer(self, matpes_variant: MACEMatpes) -> None:
        """``z_table`` is a persistent buffer, so it moves with ``.to(device)``."""
        buffers = dict(matpes_variant.named_buffers())
        assert torch.equal(buffers["z_table"], torch.tensor(ATOMIC_NUMBERS, dtype=torch.long))

    def test_compute_energy_takes_six_positional_arguments(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """``run_nve.py:118`` binds this private core and calls it positionally."""
        edge_index = cluster["edges", "edge_index"]
        shifts = torch.zeros(edge_index.shape[0], 3, dtype=torch.float64)
        with torch.no_grad():
            energy = matpes_variant._compute_energy(
                cluster["atoms", "pos"],
                cluster["atoms", "Z"],
                edge_index,
                cluster["atoms", "batch"],
                1,
                shifts,
            )
        assert energy.shape == (1,)

    def test_compute_energy_agrees_with_the_forward_energy(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """The compiled MD core and the batch path are one energy, not two."""
        with torch.no_grad():
            direct = matpes_variant._compute_energy(
                cluster["atoms", "pos"],
                cluster["atoms", "Z"],
                cluster["edges", "edge_index"],
                cluster["atoms", "batch"],
                1,
                None,
            )
        energy = matpes_variant(cluster)["graphs", "energy"]
        assert float((energy - direct).abs().max()) == EXACT

    def test_energy_forces_returns_energy_and_forces(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """``bench_mace_matpes.py``'s eager arm calls it with ``num_graphs=`` / ``shifts=``."""
        out = matpes_variant.energy_forces(
            cluster["atoms", "pos"],
            cluster["atoms", "Z"],
            cluster["edges", "edge_index"],
            cluster["atoms", "batch"],
            num_graphs=1,
            shifts=None,
        )
        assert out["energy"].shape == (1,) and out["forces"].shape == (len(CLUSTER_Z), 3)

    def test_forward_writes_the_energy_and_forces_on_the_batch(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """The molnex pipeline entry point: ``graphs.energy`` + ``atoms.forces``."""
        out = matpes_variant(cluster)
        assert out["graphs", "energy"].shape == (1,)
        assert out["atoms", "forces"].shape == (len(CLUSTER_Z), 3)

    def test_kwargs_form_matches_the_spec_form_exactly(
        self, matpes_variant: MACEMatpes, tiny_matpes_spec: MACEMatpesSpec, cluster: TensorDict
    ) -> None:
        """The alias must *be* the spec form, not merely approximate it."""
        spec_form = MACEPotential(tiny_matpes_spec).eval()
        spec_form.load_state_dict(matpes_variant.state_dict(), strict=True)
        alias_energy = matpes_variant(cluster.clone())["graphs", "energy"]
        spec_energy = spec_form(cluster.clone())["graphs", "energy"]
        assert float((alias_energy - spec_energy).abs().max()) == EXACT

    def test_rejects_atomic_numbers_outside_the_table(self, matpes_variant: MACEMatpes) -> None:
        """An off-table element would be snapped onto a neighbouring row."""
        with pytest.raises(ValueError):
            matpes_variant.validate_elements(torch.tensor(OFF_TABLE_Z, dtype=torch.long))

    def test_load_matpes_state_dict_refuses_an_alien_checkpoint(
        self, matpes_variant: MACEMatpes
    ) -> None:
        """The alias forwards to the strict ``CheckpointRemap``, not to ``load_state_dict``."""
        with pytest.raises(RuntimeError):
            load_matpes_state_dict(matpes_variant, ALIEN_CHECKPOINT)

    # -- migrated from tests/test_molzoo/test_mace_matpes.py::TestMACEMatpes by
    #    mace-subpackage-restructure-06-wire (criteria verbatim) --------------

    def test_forward_matches_energy_forces(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """forward(td) energy/forces == raw energy_forces() to machine precision."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        reference = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)
        out = matpes_variant.forward(cluster)
        assert torch.allclose(
            out["graphs", "energy"], reference["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0
        )
        assert torch.allclose(
            out["atoms", "forces"], reference["forces"], atol=FORWARD_FORCE_ATOL, rtol=0
        )

    def test_forward_returns_the_same_batch_with_new_keys(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """forward mutates in place and returns the same TensorDict object."""
        out = matpes_variant.forward(cluster)
        assert out is cluster
        nested = out.keys(include_nested=True)
        assert ("graphs", "energy") in nested
        assert ("atoms", "forces") in nested

    def test_net_force_vanishes_on_an_isolated_cluster(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """Net force through the variant's own raw-tensor seam is ~zero.

        ``forward``-path net force is covered by
        ``test_potential.py::TestMACEPotential::test_net_force_vanishes``; this
        case drives ``energy_forces`` so the variant-only leaf path carries its
        own Newton's-third-law lock.
        """
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        result = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)
        assert float(result["forces"].detach().sum(0).abs().max()) < MATPES_NET_FORCE_ATOL

    def test_energy_is_rotation_invariant(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """A rigid rotation leaves the energy unchanged (O(3) invariance)."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        rotation = _rotation_z(ROTATION_Z)
        base = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)
        turned = matpes_variant.energy_forces(pos @ rotation.T, Z, edge_index, batch, num_graphs=1)
        assert torch.allclose(base["energy"], turned["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0)

    def test_forces_rotate_with_the_system(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """Forces are equivariant: F(Rx) = R F(x)."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        rotation = _rotation_x(ROTATION_X)
        base = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)["forces"]
        turned = matpes_variant.energy_forces(pos @ rotation.T, Z, edge_index, batch, num_graphs=1)[
            "forces"
        ]
        assert torch.allclose(turned, base @ rotation.T, atol=FORWARD_FORCE_ATOL, rtol=0)

    def test_forces_match_finite_differences(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """``F = -dE/dpos`` — the autograd force must match a numeric gradient."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        forces = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)["forces"]

        for atom, axis in FD_PROBES:
            shifted = pos.clone()
            shifted[atom, axis] += FD_STEP
            plus = float(
                matpes_variant.energy_forces(shifted, Z, edge_index, batch, num_graphs=1)[
                    "energy"
                ].sum()
            )
            shifted[atom, axis] -= 2 * FD_STEP
            minus = float(
                matpes_variant.energy_forces(shifted, Z, edge_index, batch, num_graphs=1)[
                    "energy"
                ].sum()
            )
            assert float(forces[atom, axis]) == pytest.approx(
                -(plus - minus) / (2 * FD_STEP), abs=FD_ATOL
            )

    def test_batched_graphs(self, matpes_variant: MACEMatpes, pair_batch: TensorDict) -> None:
        """Two clusters in one batch: per-graph energies, correct atom routing."""
        pos, Z, edge_index, batch, num_graphs = raw_tensors(pair_batch)
        reference = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=num_graphs)
        out = matpes_variant.forward(pair_batch)
        assert out["graphs", "energy"].shape == (2,)
        assert torch.allclose(
            out["graphs", "energy"], reference["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0
        )

    def test_energy_is_extensive_over_separated_graphs(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """Two non-interacting copies carry twice one copy's energy."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        n_atoms = len(CLUSTER_Z)
        one = matpes_variant.energy_forces(pos, Z, edge_index, batch, num_graphs=1)["energy"]
        pair = matpes_variant.energy_forces(
            torch.cat([pos, pos]),
            torch.cat([Z, Z]),
            torch.cat([edge_index, edge_index + n_atoms], dim=0),
            torch.cat([batch, batch + 1]),
            num_graphs=2,
        )["energy"]
        assert torch.allclose(pair, one.repeat(2), atol=FORWARD_ENERGY_ATOL, rtol=0)

    def test_forward_rejects_an_element_outside_the_table(
        self, matpes_variant: MACEMatpes, cluster: TensorDict
    ) -> None:
        """An out-of-table Z would be snapped onto a neighbour and silently wrong."""
        cluster["atoms", "Z"] = torch.tensor(OFF_TABLE_CLUSTER_Z, dtype=torch.long)
        with pytest.raises(ValueError, match="outside this model"):
            matpes_variant.forward(cluster)

    def test_rejects_single_interaction(self) -> None:
        """The readout schedule needs a first and a last layer.

        The spec owns the rule (``test_spec.py``); this pins that the keyword
        constructor propagates it instead of silently building a one-layer model.
        """
        with pytest.raises(ValueError, match="num_interactions"):
            MACEMatpes(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=torch.tensor(ATOMIC_ENERGIES),
                num_interactions=1,
            )

    def test_first_product_has_no_skip_connection(self, matpes_variant: MACEMatpes) -> None:
        """MACE only carries a residual from the second layer on."""
        assert matpes_variant.products[0].use_sc is False
        assert matpes_variant.products[1].use_sc is True

    def test_last_layer_keeps_scalars_only(self, matpes_variant: MACEMatpes) -> None:
        """The final hidden state is scalar; earlier layers keep l>0."""
        assert matpes_variant.readouts[0].__class__.__name__ == "LinearReadout"
        assert matpes_variant.readouts[1].__class__.__name__ == "NonLinearReadout"

    # -- migrated from tests/test_molzoo/test_mace_matpes.py::TestDeadEdgePadding
    #    (same step; the subject is the variant, so the two cases fold into this
    #    class rather than keeping a second class for one production unit) ----
    #
    # A padded (dead) edge must contribute exactly nothing. This is the
    # load-bearing property of :class:`molix.md.PeriodicNeighborList`: its
    # fixed-capacity buffers pad with self-loops on atom 0 displaced beyond the
    # cutoff, and the whole rebuild-under-CUDA-graphs design assumes such edges
    # are invisible to the model. Random weights make the check stronger — the
    # zeros must come from the cutoff structure, not from trained smallness.

    def test_energy_and_forces_ignore_dead_edges(self, matpes_variant: MACEMatpes) -> None:
        """Same system, 0% vs heavy padding: identical energy and forces."""
        from molix.md import PeriodicNeighborList

        cell = torch.eye(3, dtype=torch.float64) * PBC_CELL
        pos = torch.tensor(PBC_POS, dtype=torch.float64)
        Z = torch.tensor(PBC_Z, dtype=torch.long)
        batch = torch.zeros(len(PBC_Z), dtype=torch.long)

        outs = []
        for factor in PBC_CAPACITY_FACTORS:
            nl = PeriodicNeighborList(
                cell=cell, cutoff=PBC_CUTOFF, positions=pos, capacity_factor=factor
            )
            out = matpes_variant.energy_forces(
                pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts
            )
            outs.append((nl, out))

        (nl_a, a), (nl_b, b) = outs
        assert nl_b.capacity > nl_a.capacity  # the padded arm really is padded
        assert nl_a.num_edges == nl_b.num_edges
        assert torch.equal(a["energy"], b["energy"])
        assert torch.equal(a["forces"], b["forces"])

    def test_rebuild_preserves_energy_at_identical_positions(
        self, matpes_variant: MACEMatpes
    ) -> None:
        """rebuild() at the same positions must not change the physics."""
        from molix.md import PeriodicNeighborList

        cell = torch.eye(3, dtype=torch.float64) * PBC_CELL
        pos = torch.tensor(PBC_POS[:3], dtype=torch.float64)
        Z = torch.tensor(PBC_Z[:3], dtype=torch.long)
        batch = torch.zeros(3, dtype=torch.long)
        nl = PeriodicNeighborList(
            cell=cell,
            cutoff=PBC_CUTOFF,
            positions=pos,
            capacity_factor=REBUILD_CAPACITY_FACTOR,
        )
        before = matpes_variant.energy_forces(
            pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts
        )["energy"].clone()
        nl.rebuild(pos)
        after = matpes_variant.energy_forces(
            pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts
        )["energy"]
        assert torch.equal(before, after)


class TestMACEOMol:
    """The keyword surface of the OMOL foundation variant."""

    def test_constructs_from_the_seventeen_keywords(self) -> None:
        """The full consumed signature, charge/spin conditioning sizes included."""
        torch.manual_seed(0)
        model = MACEOMol(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=torch.tensor(ATOMIC_ENERGIES),
            charge_classes=CHARGE_CLASSES,
            charge_offset=CHARGE_OFFSET,
            spin_classes=SPIN_CLASSES,
            spin_offset=SPIN_OFFSET,
            scale=SCALE,
            shift=SHIFT,
            **TINY_OMOL_KWARGS,
        )
        assert isinstance(model, torch.nn.Module)

    def test_exposes_the_consumed_method_surface(self, omol_variant: MACEOMol) -> None:
        """``forward`` / ``energy_forces`` / ``validate_elements`` all survive."""
        assert all(
            callable(getattr(omol_variant, name, None))
            for name in ("forward", "energy_forces", "validate_elements")
        )

    def test_registers_the_element_table_as_a_buffer(self, omol_variant: MACEOMol) -> None:
        """``z_table`` is a persistent buffer, so it moves with ``.to(device)``."""
        buffers = dict(omol_variant.named_buffers())
        assert torch.equal(buffers["z_table"], torch.tensor(ATOMIC_NUMBERS, dtype=torch.long))

    def test_energy_forces_returns_energy_and_forces(
        self, omol_variant: MACEOMol, omol_cluster: TensorDict
    ) -> None:
        """OMOL's own raw-tensor entry point takes the two conditioning tensors."""
        out = omol_variant.energy_forces(
            omol_cluster["atoms", "pos"],
            omol_cluster["atoms", "Z"],
            omol_cluster["edges", "edge_index"],
            omol_cluster["atoms", "batch"],
            omol_cluster["graphs", "total_charge"],
            omol_cluster["graphs", "total_spin"],
        )
        assert out["energy"].shape == (1,) and out["forces"].shape == (len(CLUSTER_Z), 3)

    def test_forward_writes_the_energy_and_forces_on_the_batch(
        self, omol_variant: MACEOMol, omol_cluster: TensorDict
    ) -> None:
        """The molnex pipeline entry point: ``graphs.energy`` + ``atoms.forces``."""
        out = omol_variant(omol_cluster)
        assert out["graphs", "energy"].shape == (1,)
        assert out["atoms", "forces"].shape == (len(CLUSTER_Z), 3)

    def test_kwargs_form_matches_the_spec_form_exactly(
        self, omol_variant: MACEOMol, tiny_omol_spec: MACEOMolSpec, omol_cluster: TensorDict
    ) -> None:
        """Non-vacuous by construction — see ``conftest.wake_zero_init_readout``."""
        spec_form = MACEPotential(tiny_omol_spec).eval()
        spec_form.load_state_dict(omol_variant.state_dict(), strict=True)
        alias_energy = omol_variant(omol_cluster.clone())["graphs", "energy"]
        spec_energy = spec_form(omol_cluster.clone())["graphs", "energy"]
        assert float((alias_energy - spec_energy).abs().max()) == EXACT

    def test_rejects_atomic_numbers_outside_the_table(self, omol_variant: MACEOMol) -> None:
        """The pre-cutover flat ``MACEOMol`` had no such gate; the shared encoder brings one."""
        with pytest.raises(ValueError):
            omol_variant.validate_elements(torch.tensor(OFF_TABLE_Z, dtype=torch.long))

    def test_load_omol_state_dict_refuses_an_alien_checkpoint(self, omol_variant: MACEOMol) -> None:
        """OMOL returns unhoused keys, but still refuses to leave parameters unfilled."""
        with pytest.raises(RuntimeError):
            load_omol_state_dict(omol_variant, ALIEN_CHECKPOINT)

    # -- migrated from tests/test_molzoo/test_mace_omol.py (module-level cases)
    #    by mace-subpackage-restructure-06-wire (criteria verbatim) -----------

    def test_forward_matches_energy_forces(
        self, omol_variant: MACEOMol, charged_omol_cluster: TensorDict
    ) -> None:
        """forward(td) energy/forces == raw energy_forces() to machine precision.

        Conditioned on a *non-default* charge/spin pair, so a ``forward`` that
        ignored ``graphs`` and fell back to the neutral singlet would fail here.
        """
        pos, Z, edge_index, batch, _ = raw_tensors(charged_omol_cluster)
        total_charge = charged_omol_cluster["graphs", "total_charge"]
        total_spin = charged_omol_cluster["graphs", "total_spin"]
        reference = omol_variant.energy_forces(pos, Z, edge_index, batch, total_charge, total_spin)
        out = omol_variant.forward(charged_omol_cluster)
        assert torch.allclose(
            out["graphs", "energy"], reference["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0
        )
        assert torch.allclose(
            out["atoms", "forces"], reference["forces"], atol=FORWARD_FORCE_ATOL, rtol=0
        )

    def test_forward_returns_the_same_batch_with_new_keys(
        self, omol_variant: MACEOMol, charged_omol_cluster: TensorDict
    ) -> None:
        """forward mutates in place and returns the same TensorDict object."""
        out = omol_variant.forward(charged_omol_cluster)
        assert out is charged_omol_cluster
        assert ("graphs", "energy") in out.keys(include_nested=True)
        assert ("atoms", "forces") in out.keys(include_nested=True)

    def test_net_force_vanishes_on_an_isolated_molecule(
        self, omol_variant: MACEOMol, charged_omol_cluster: TensorDict
    ) -> None:
        """Net force on an isolated molecule is ~zero (translation invariance)."""
        out = omol_variant.forward(charged_omol_cluster)
        assert float(out["atoms", "forces"].detach().sum(0).abs().max()) < OMOL_NET_FORCE_ATOL

    def test_missing_charge_spin_defaults_to_neutral(
        self, omol_variant: MACEOMol, cluster: TensorDict
    ) -> None:
        """Absent graphs.total_charge/total_spin → neutral singlet, still runs."""
        pos, Z, edge_index, batch, _ = raw_tensors(cluster)
        # forward defaults to the OMOL neutral closed-shell singlet: charge=0,
        # spin=1 (spin index 1 is a trained row; spin 0 hits an untrained
        # embedding).
        charge = torch.zeros(1, dtype=torch.long)
        spin = torch.ones(1, dtype=torch.long)
        reference = omol_variant.energy_forces(pos, Z, edge_index, batch, charge, spin)
        out = omol_variant.forward(cluster)
        assert torch.allclose(
            out["graphs", "energy"], reference["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0
        )

    def test_force_loss_reaches_parameters_in_eval_mode(
        self, omol_variant: MACEOMol, charged_omol_cluster: TensorDict
    ) -> None:
        """Force-supervised training must backprop to the parameters.

        Regression for the pre-cutover OMOL model gating ``create_graph`` on
        ``self.training``: with the model in its default eval mode the autograd
        force was detached from the parameter graph, so a force loss produced
        zero gradient for every parameter (and ``loss.backward()`` raised). The
        robust form keeps the force in the graph whenever grad is enabled, so a
        strict majority of parameters receive a gradient. See spec
        ``cuet-force-doublebackward`` Findings (run 2).

        The readout's output layers are zero-initialised (MACE starts at the E0
        baseline), which makes a fresh model's energy position-independent and
        its forces identically zero — a degenerate state in which no force loss
        can reach any parameter. Perturb the parameters first so the model
        produces real forces, then assert the loss reaches them.
        """
        pos, Z, edge_index, batch, _ = raw_tensors(charged_omol_cluster)
        total_charge = charged_omol_cluster["graphs", "total_charge"]
        total_spin = charged_omol_cluster["graphs", "total_spin"]
        assert not omol_variant.training  # default eval mode — the regression condition

        _perturb(omol_variant)
        omol_variant.zero_grad(set_to_none=True)
        forces = omol_variant.energy_forces(pos, Z, edge_index, batch, total_charge, total_spin)[
            "forces"
        ]
        assert forces.abs().max() > 0.0, "perturbed model still produces zero forces"

        (forces**2).mean().backward()

        with_grad, total = _fraction_with_gradient(omol_variant)
        assert with_grad > total // 2, (
            f"force loss reached only {with_grad}/{total} parameters; "
            "the force is detached from the parameter graph"
        )

    def test_force_loss_through_forward_reaches_parameters(
        self, omol_variant: MACEOMol, charged_omol_cluster: TensorDict
    ) -> None:
        """The molnex-pipeline ``forward`` also unlocks force-supervised training.

        ``forward`` derives forces through the shared batch-level force kernel
        (``molpot.derivation.kernels``) rather than the variant's own
        ``energy_forces`` leaf, so it is a second path that must stay connected
        to the parameters. Before the port it raised ``setup_context`` because
        ``torch.func.grad`` cannot trace cuEquivariance's fused custom ops; see
        spec ``cuet-force-doublebackward`` Resolution (run 3).
        """
        _perturb(omol_variant)

        out = omol_variant.forward(charged_omol_cluster)
        forces = out["atoms", "forces"]
        assert forces.abs().max() > 0.0

        omol_variant.zero_grad(set_to_none=True)
        (forces**2).mean().backward()

        with_grad, total = _fraction_with_gradient(omol_variant)
        assert with_grad > total // 2, (
            f"force loss through forward reached only {with_grad}/{total} parameters"
        )

    def test_batched_graphs(self, omol_variant: MACEOMol, omol_pair_batch: TensorDict) -> None:
        """Two molecules in one batch: per-graph energies, correct atom routing."""
        pos, Z, edge_index, batch, _ = raw_tensors(omol_pair_batch)
        reference = omol_variant.energy_forces(
            pos,
            Z,
            edge_index,
            batch,
            omol_pair_batch["graphs", "total_charge"],
            omol_pair_batch["graphs", "total_spin"],
        )
        out = omol_variant.forward(omol_pair_batch)
        assert out["graphs", "energy"].shape == (2,)
        assert torch.allclose(
            out["graphs", "energy"], reference["energy"], atol=FORWARD_ENERGY_ATOL, rtol=0
        )
        assert torch.allclose(
            out["atoms", "forces"], reference["forces"], atol=FORWARD_FORCE_ATOL, rtol=0
        )
        assert out["atoms", "forces"].shape == (len(PAIR_Z), 3)
