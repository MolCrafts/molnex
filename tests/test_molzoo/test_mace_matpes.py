"""Tests for the MACEMatpes model and its official-checkpoint loader.

Structural / contract tests on a small instance. Numerical parity against the
official ``MACE-matpes-r2scan-0`` weights is verified out of tree (the oracle
imports ``mace-torch``); the numbers are recorded in
``src/molzoo/specs/mace_matpes.md`` §7.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix import config
from molzoo.mace_matpes import MACEMatpes, load_matpes_state_dict

ATOMIC_NUMBERS = [1, 6, 8]
ATOMIC_ENERGIES = torch.tensor([-13.6, -1029.0, -2041.0])


@pytest.fixture(autouse=True)
def fp64():
    """Build every model in this module at fp64.

    cuEquivariance freezes its working precision at construction, so a model
    built under the default fp32 and then ``.double()``-d still contracts in
    float32 — ~1e-8 eV of noise, which is fine for inference but swamps the
    invariance and finite-difference assertions below. Foundation-weight use is
    fp64 anyway, so the tests exercise that path.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


@pytest.fixture
def model():
    """Small fp64 MACEMatpes (l_max=1) — non-MatPES dims, fast on CPU."""
    torch.manual_seed(0)
    return MACEMatpes(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        l_max=1,
        num_features=16,
        max_hidden_l=1,
        num_interactions=2,
        correlation=2,
        mlp_dim=8,
        radial_mlp=[8],
        use_fallback=True,  # CPU tests: fused kernels need a GPU + ops wheel
    ).eval()


@pytest.fixture
def single_graph():
    """A 5-atom cluster with all intra-graph ordered pairs as edges."""
    torch.manual_seed(1)
    pos = torch.randn(5, 3, dtype=torch.float64) * 1.5
    Z = torch.tensor([1, 6, 8, 1, 1])
    batch = torch.zeros(5, dtype=torch.long)
    src, dst = [], []
    for i in range(5):
        for j in range(5):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst]).t().contiguous()  # (E, 2)
    return pos, Z, batch, edge_index


def _make_batch(pos, Z, batch, edge_index) -> TensorDict:
    """Assemble a post-collate-style TensorDict (atoms / edges)."""
    e2 = edge_index
    return TensorDict(
        {
            "atoms": TensorDict({"Z": Z, "pos": pos, "batch": batch}, batch_size=[Z.shape[0]]),
            "edges": TensorDict({"edge_index": e2}, batch_size=[e2.shape[0]]),
        },
        batch_size=[],
    )


class TestMACEMatpes:
    """Test the MACE-MatPES energy/force model."""

    def test_forward_matches_energy_forces(self, model, single_graph):
        """forward(td) energy/forces == raw energy_forces() to machine precision."""
        pos, Z, batch, edge_index = single_graph
        ref = model.energy_forces(pos, Z, edge_index, batch, num_graphs=1)
        out = model.forward(_make_batch(pos.clone(), Z, batch, edge_index))
        assert torch.allclose(out["graphs", "energy"], ref["energy"], atol=1e-9, rtol=0)
        assert torch.allclose(out["atoms", "forces"], ref["forces"], atol=1e-8, rtol=0)

    def test_forward_returns_same_td_with_new_keys(self, model, single_graph):
        """forward mutates in place and returns the same TensorDict object."""
        td = _make_batch(*single_graph[:3], single_graph[3])
        out = model.forward(td)
        assert out is td
        nested = out.keys(include_nested=True)
        assert ("graphs", "energy") in nested
        assert ("atoms", "forces") in nested

    def test_forces_translation_invariant(self, model, single_graph):
        """Net force on an isolated cluster is ~zero."""
        out = model.forward(_make_batch(*single_graph[:3], single_graph[3]))
        assert float(out["atoms", "forces"].sum(0).abs().max()) < 1e-8

    def test_energy_rotation_invariant(self, model, single_graph):
        """A rigid rotation leaves the energy unchanged (O(3) invariance)."""
        pos, Z, batch, edge_index = single_graph
        angle = torch.tensor(0.7, dtype=torch.float64)
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        base = model.energy_forces(pos, Z, edge_index, batch, num_graphs=1, compute_forces=False)
        turned = model.energy_forces(
            pos @ rotation.T, Z, edge_index, batch, num_graphs=1, compute_forces=False
        )
        assert torch.allclose(base["energy"], turned["energy"], atol=1e-9, rtol=0)

    def test_forces_rotate_with_the_system(self, model, single_graph):
        """Forces are equivariant: F(Rx) = R F(x)."""
        pos, Z, batch, edge_index = single_graph
        angle = torch.tensor(-0.4, dtype=torch.float64)
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
        base = model.energy_forces(pos, Z, edge_index, batch, num_graphs=1)["forces"]
        turned = model.energy_forces(pos @ rotation.T, Z, edge_index, batch, num_graphs=1)["forces"]
        assert torch.allclose(turned, base @ rotation.T, atol=1e-8, rtol=0)

    def test_forces_match_finite_differences(self, model, single_graph):
        """``F = -dE/dpos`` — the autograd force must match a numeric gradient."""
        pos, Z, batch, edge_index = single_graph
        forces = model.energy_forces(pos, Z, edge_index, batch, num_graphs=1)["forces"]

        eps = 1e-6
        for atom, axis in ((0, 0), (2, 1), (4, 2)):
            shifted = pos.clone()
            shifted[atom, axis] += eps
            plus = float(
                model.energy_forces(
                    shifted, Z, edge_index, batch, num_graphs=1, compute_forces=False
                )["energy"].sum()
            )
            shifted[atom, axis] -= 2 * eps
            minus = float(
                model.energy_forces(
                    shifted, Z, edge_index, batch, num_graphs=1, compute_forces=False
                )["energy"].sum()
            )
            assert float(forces[atom, axis]) == pytest.approx(-(plus - minus) / (2 * eps), abs=1e-5)

    def test_batched_graphs(self, model):
        """Two clusters in one batch: per-graph energies, correct atom routing."""
        torch.manual_seed(2)
        pos = torch.randn(7, 3, dtype=torch.float64) * 1.5
        Z = torch.tensor([1, 6, 8, 1, 8, 6, 1])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        src, dst = [], []
        for i in range(7):
            for j in range(7):
                if i != j and batch[i] == batch[j]:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst]).t().contiguous()  # (E, 2)
        out = model.forward(_make_batch(pos, Z, batch, edge_index))
        ref = model.energy_forces(pos, Z, edge_index, batch, num_graphs=2)
        assert out["graphs", "energy"].shape == (2,)
        assert torch.allclose(out["graphs", "energy"], ref["energy"], atol=1e-9, rtol=0)

    def test_energy_is_extensive_over_separated_graphs(self, model):
        """Two non-interacting copies carry twice one copy's energy."""
        torch.manual_seed(3)
        pos = torch.randn(3, 3, dtype=torch.float64)
        Z = torch.tensor([1, 8, 1])
        edge = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]).t().contiguous()  # (E, 2)
        one = model.energy_forces(
            pos, Z, edge, torch.zeros(3, dtype=torch.long), num_graphs=1, compute_forces=False
        )["energy"]
        pair = model.energy_forces(
            torch.cat([pos, pos]),
            torch.cat([Z, Z]),
            torch.cat([edge, edge + 3], dim=0),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            num_graphs=2,
            compute_forces=False,
        )["energy"]
        assert torch.allclose(pair, one.repeat(2), atol=1e-9, rtol=0)

    def test_rejects_element_outside_the_table(self, model, single_graph):
        """An out-of-table Z would be snapped onto a neighbour and silently wrong."""
        pos, _, batch, edge_index = single_graph
        Z = torch.tensor([1, 6, 8, 1, 26])  # Fe is not in this model's table
        with pytest.raises(ValueError, match="outside this model"):
            model.forward(_make_batch(pos, Z, batch, edge_index))

    def test_rejects_single_interaction(self):
        """The readout schedule needs a first and a last layer."""
        with pytest.raises(ValueError, match="num_interactions"):
            MACEMatpes(
                atomic_numbers=ATOMIC_NUMBERS,
                atomic_energies=ATOMIC_ENERGIES,
                num_interactions=1,
            )

    def test_first_product_has_no_skip_connection(self, model):
        """MACE only carries a residual from the second layer on."""
        assert model.products[0].use_sc is False
        assert model.products[1].use_sc is True

    def test_last_layer_keeps_scalars_only(self, model):
        """The final hidden state is scalar; earlier layers keep l>0."""
        assert model.readouts[0].__class__.__name__ == "LinearReadout"
        assert model.readouts[1].__class__.__name__ == "NonLinearReadout"


class TestLoadMatpesStateDict:
    """Test the strict official-checkpoint loader."""

    def _roundtrip_state(self, model: MACEMatpes) -> dict:
        """Rename a model's own state_dict back into official cueq key names."""
        rename = {
            "node_embedding.": "node_embedding.linear.",
            "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
            "distance_transform.": "radial_embedding.distance_transform.",
            "pair_repulsion.": "pair_repulsion_fn.",
            "z_table": "atomic_numbers",
        }
        out = {}
        for key, value in model.state_dict().items():
            for prefix, official in sorted(rename.items(), key=lambda kv: -len(kv[0])):
                if key == prefix or key.startswith(prefix):
                    key = official + (key[len(prefix) :] if key != prefix else "")
                    break
            out[key] = value
        return out

    def test_roundtrip_restores_every_parameter(self, model):
        """A checkpoint written in official names loads back bit-for-bit."""
        state = self._roundtrip_state(model)
        fresh = MACEMatpes(
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
            r_max=5.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            l_max=1,
            num_features=16,
            max_hidden_l=1,
            num_interactions=2,
            correlation=2,
            mlp_dim=8,
            radial_mlp=[8],
        )
        load_matpes_state_dict(fresh, state)
        for (name, want), (_, got) in zip(
            model.named_parameters(), fresh.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_rejects_unknown_checkpoint_key(self, model):
        """A key with no home means the mapping is stale — do not load silently."""
        state = self._roundtrip_state(model)
        state["interactions.0.mystery_layer.weight"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="no home"):
            load_matpes_state_dict(model, state)

    def test_rejects_missing_parameter(self, model):
        """A parameter the checkpoint never fills would keep its random init."""
        state = self._roundtrip_state(model)
        del state["interactions.0.linear_up.weight"]
        with pytest.raises(RuntimeError, match="not covered"):
            load_matpes_state_dict(model, state)

    def test_rejects_shape_mismatch(self, model):
        """Wrong shapes mean the model was built with the wrong config."""
        state = self._roundtrip_state(model)
        state["interactions.0.linear_up.weight"] = torch.zeros(1, 7, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            load_matpes_state_dict(model, state)

    def test_accepts_rank_difference_on_frozen_scalars(self, model):
        """MACE stores scale/shift as ``(1,)``; molnex holds a 0-d buffer."""
        state = self._roundtrip_state(model)
        state["scale_shift.scale"] = torch.tensor([0.75], dtype=torch.float64)
        load_matpes_state_dict(model, state)
        assert float(model.scale_shift.scale) == pytest.approx(0.75)


class TestDeadEdgePadding:
    """A padded (dead) edge must contribute exactly nothing.

    This is the load-bearing property of :class:`molix.md.PeriodicNeighborList`:
    its fixed-capacity buffers pad with self-loops on atom 0 displaced beyond
    the cutoff, and the whole rebuild-under-CUDA-graphs design assumes such
    edges are invisible to the model. Random weights make the check stronger —
    the zeros must come from the cutoff structure, not from trained smallness.
    """

    def test_energy_and_forces_ignore_dead_edges(self, model):
        """Same system, 0% vs heavy padding: identical energy and forces."""
        from molix.md import PeriodicNeighborList

        torch.manual_seed(4)
        cell = torch.eye(3, dtype=torch.float64) * 7.0
        pos = torch.tensor(
            [[0.5, 0.5, 0.5], [1.6, 0.6, 0.4], [0.4, 1.7, 0.6], [6.8, 6.9, 0.2]],
            dtype=torch.float64,
        )
        Z = torch.tensor([8, 1, 1, 1])
        batch = torch.zeros(4, dtype=torch.long)

        outs = []
        for factor in (1.0, 3.0):
            nl = PeriodicNeighborList(cell=cell, cutoff=3.4, positions=pos, capacity_factor=factor)
            out = model.energy_forces(pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts)
            outs.append((nl, out))

        (nl_a, a), (nl_b, b) = outs
        assert nl_b.capacity > nl_a.capacity  # the padded arm really is padded
        assert nl_a.num_edges == nl_b.num_edges
        assert torch.equal(a["energy"], b["energy"])
        assert torch.equal(a["forces"], b["forces"])

    def test_rebuild_preserves_energy_at_identical_positions(self, model):
        """rebuild() at the same positions must not change the physics."""
        from molix.md import PeriodicNeighborList

        cell = torch.eye(3, dtype=torch.float64) * 7.0
        pos = torch.tensor([[0.5, 0.5, 0.5], [1.6, 0.6, 0.4], [0.4, 1.7, 0.6]], dtype=torch.float64)
        Z = torch.tensor([8, 1, 1])
        batch = torch.zeros(3, dtype=torch.long)
        nl = PeriodicNeighborList(cell=cell, cutoff=3.4, positions=pos, capacity_factor=2.0)
        before = model.energy_forces(
            pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts, compute_forces=False
        )["energy"].clone()
        nl.rebuild(pos)
        after = model.energy_forces(
            pos, Z, nl.edge_index, batch, num_graphs=1, shifts=nl.shifts, compute_forces=False
        )["energy"]
        assert torch.equal(before, after)
