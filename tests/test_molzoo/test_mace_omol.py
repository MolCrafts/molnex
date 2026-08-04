"""Integration tests for the MACEOMol molnex-pipeline adapter.

Covers ``mace-omol-port-02-pipeline-integration`` ac-001: ``MACEOMol.forward``
consumes the post-collate ``atoms / edges / graphs`` TensorDict, routes forces
through ``molpot.derivation.ForceDerivation``, and agrees with the raw-tensor
``MACEOMol.energy_forces`` entry point to machine precision.

The full-OMOL energy/force faithfulness vs the official model is covered by
``mace-omol-port-01`` (scripts/omol_port/verify_e2e.py); here we only assert the
two molnex entry points are consistent, on a small l_max=1 instance.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molzoo.mace_omol import MACEOMol


def _full_edges(batch: torch.Tensor) -> torch.Tensor:
    """All intra-graph ordered pairs as ``(E, 2)`` ``[source, target]`` edges."""
    src, dst = [], []
    n = batch.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and batch[i] == batch[j]:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long).t().contiguous()


def _make_batch(pos, Z, batch, edge_index_e2, total_charge, total_spin) -> TensorDict:
    """Assemble a post-collate-style TensorDict (atoms / edges / graphs)."""
    src = edge_index_e2[:, 0]
    dst = edge_index_e2[:, 1]
    edge_diff = pos[dst] - pos[src]
    edge_dist = torch.linalg.norm(edge_diff, dim=-1)
    n = Z.shape[0]
    e = edge_index_e2.shape[0]
    b = total_charge.shape[0]
    return TensorDict(
        {
            "atoms": TensorDict({"Z": Z, "pos": pos, "batch": batch}, batch_size=[n]),
            "edges": TensorDict(
                {
                    "edge_index": edge_index_e2,
                    "edge_diff": edge_diff,
                    "edge_dist": edge_dist,
                },
                batch_size=[e],
            ),
            "graphs": TensorDict(
                {"total_charge": total_charge, "total_spin": total_spin},
                batch_size=[b],
            ),
        },
        batch_size=[],
    )


@pytest.fixture
def model():
    """Small fp64 MACEOMol (l_max=1) — non-OMOL dims, fast on CPU."""
    torch.manual_seed(0)
    ae = torch.tensor([-13.6, -1029.0, -2041.0])
    m = MACEOMol(
        atomic_numbers=[1, 6, 8],
        atomic_energies=ae,
        r_max=5.0,
        num_bessel=8,
        l_max=1,
        num_features=64,
        num_interactions=2,
        correlation=2,
    ).double()
    return m.eval()


@pytest.fixture
def single_graph():
    torch.manual_seed(1)
    pos = torch.randn(5, 3, dtype=torch.float64) * 1.5
    Z = torch.tensor([1, 6, 8, 1, 1])
    batch = torch.zeros(5, dtype=torch.long)
    total_charge = torch.tensor([1], dtype=torch.long)  # charged molecule
    total_spin = torch.tensor([0], dtype=torch.long)
    return pos, Z, batch, total_charge, total_spin


def test_forward_matches_energy_forces(model, single_graph):
    """forward(td) energy/forces == raw energy_forces() to machine precision."""
    pos, Z, batch, tc, ts = single_graph
    edge_e2 = _full_edges(batch)

    ref = model.energy_forces(pos, Z, edge_e2.t().contiguous(), batch, tc, ts)

    td = _make_batch(pos.clone(), Z, batch, edge_e2, tc, ts)
    out = model.forward(td)

    assert torch.allclose(out["graphs", "energy"], ref["energy"], atol=1e-9, rtol=0)
    assert torch.allclose(out["atoms", "forces"], ref["forces"], atol=1e-8, rtol=0)


def test_forward_returns_same_td_with_new_keys(model, single_graph):
    """forward mutates in place and returns the same TensorDict object."""
    pos, Z, batch, tc, ts = single_graph
    td = _make_batch(pos, Z, batch, _full_edges(batch), tc, ts)
    out = model.forward(td)
    assert out is td
    assert ("graphs", "energy") in out.keys(include_nested=True)
    assert ("atoms", "forces") in out.keys(include_nested=True)


def test_forces_translation_invariant(model, single_graph):
    """Net force on an isolated molecule is ~zero (translation invariance)."""
    pos, Z, batch, tc, ts = single_graph
    td = _make_batch(pos, Z, batch, _full_edges(batch), tc, ts)
    out = model.forward(td)
    net = out["atoms", "forces"].sum(0)
    assert net.abs().max() < 1e-7


def test_missing_charge_spin_defaults_to_neutral(model, single_graph):
    """Absent graphs.total_charge/total_spin → neutral singlet, still runs."""
    pos, Z, batch, _, _ = single_graph
    edge_e2 = _full_edges(batch)
    td = TensorDict(
        {
            "atoms": TensorDict({"Z": Z, "pos": pos, "batch": batch}, batch_size=[5]),
            "edges": TensorDict({"edge_index": edge_e2}, batch_size=[edge_e2.shape[0]]),
        },
        batch_size=[],
    )
    out = model.forward(td)
    # forward defaults to the OMOL neutral closed-shell singlet: charge=0, spin=1
    # (spin index 1 is a trained row; spin 0 hits an untrained embedding).
    charge = torch.zeros(1, dtype=torch.long)
    spin = torch.ones(1, dtype=torch.long)
    ref = model.energy_forces(pos, Z, edge_e2.t().contiguous(), batch, charge, spin)
    assert torch.allclose(out["graphs", "energy"], ref["energy"], atol=1e-9, rtol=0)


def test_force_loss_reaches_parameters_in_eval_mode(model, single_graph):
    """Force-supervised training must backprop to the parameters.

    Regression for ``mace_omol.py`` ``energy_forces`` gating ``create_graph`` on
    ``self.training``: with the model in its default eval mode the autograd
    force was detached from the parameter graph, so a force loss produced zero
    gradient for every parameter (and ``loss.backward()`` raised). The robust
    form keeps the force in the graph whenever grad is enabled, so a strict
    majority of parameters receive a gradient. See spec
    ``cuet-force-doublebackward`` Findings (run 2).

    The readout's output layers are zero-initialised (MACE starts at the E0
    baseline), which makes a fresh model's energy position-independent and its
    forces identically zero — a degenerate state in which no force loss can
    reach any parameter. Perturb the parameters first so the model produces real
    forces, then assert the loss reaches them.
    """
    pos, Z, batch, tc, ts = single_graph
    edge_index = _full_edges(batch).t().contiguous()
    assert not model.training  # default eval mode — the regression condition

    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    model.zero_grad(set_to_none=True)
    out = model.energy_forces(pos, Z, edge_index, batch, tc, ts, compute_forces=True)
    forces = out["forces"]
    assert forces.abs().max() > 0.0, "perturbed model still produces zero forces"

    loss = (forces**2).mean()
    loss.backward()

    params = list(model.parameters())
    n_with_grad = sum(int(p.grad is not None and float(p.grad.abs().sum()) > 0.0) for p in params)
    assert n_with_grad > len(params) // 2, (
        f"force loss reached only {n_with_grad}/{len(params)} parameters; "
        "the force is detached from the parameter graph"
    )


def test_functorch_force_loss_trains_via_forward(model, single_graph):
    """The molnex-pipeline ``forward`` (functorch ``ForceDerivation``) unlocks
    force-supervised training.

    ``forward`` derives forces with ``torch.func.grad`` (compile-friendly, single
    backward). That transform cannot trace cuEquivariance's fused custom ops, so
    before the functorch port it raised ``setup_context``; now the spherical
    harmonics are pure-torch and the equivariant tensor products run on their
    ``use_fallback`` path, so a force loss backprops to the parameters. See spec
    ``cuet-force-doublebackward`` Resolution (run 3, functorch port).
    """
    pos, Z, batch, tc, ts = single_graph

    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    td = _make_batch(pos, Z, batch, _full_edges(batch), tc, ts)
    out = model.forward(td)
    forces = out["atoms", "forces"]
    assert forces.abs().max() > 0.0

    model.zero_grad(set_to_none=True)
    (forces**2).mean().backward()

    params = list(model.parameters())
    n_with_grad = sum(int(p.grad is not None and float(p.grad.abs().sum()) > 0.0) for p in params)
    assert n_with_grad > len(params) // 2, (
        f"functorch force loss reached only {n_with_grad}/{len(params)} parameters"
    )


def test_batched_graphs(model):
    """Two molecules in one batch: per-graph energies, correct atom routing."""
    torch.manual_seed(2)
    pos = torch.randn(7, 3, dtype=torch.float64) * 1.5
    Z = torch.tensor([1, 6, 8, 1, 8, 6, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    tc = torch.tensor([0, -1], dtype=torch.long)
    ts = torch.tensor([0, 1], dtype=torch.long)
    edge_e2 = _full_edges(batch)
    td = _make_batch(pos, Z, batch, edge_e2, tc, ts)
    out = model.forward(td)
    ref = model.energy_forces(pos, Z, edge_e2.t().contiguous(), batch, tc, ts)
    assert out["graphs", "energy"].shape == (2,)
    assert torch.allclose(out["graphs", "energy"], ref["energy"], atol=1e-9, rtol=0)
    assert torch.allclose(out["atoms", "forces"], ref["forces"], atol=1e-8, rtol=0)
