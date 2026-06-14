"""Tests for :class:`molpot.heads.BondChargeHead`.

The contract under test:

1. **Architectural charge conservation.** For a clean bidirectional
   edge list, ``\\sum_i q_i = 0`` to ``float32``/``float64`` precision
   *without* relying on the projection step. This is the central
   value-add of the antisymmetric construction.
2. **Pair-wise antisymmetry.** ``q_{ij} = -q_{ji}`` per edge — proves
   that the MLP is invoked correctly with the swapped input.
3. **Half-list path.** With ``full_neighbor_list=False`` and a half
   edge list, the scatter to source/target with opposite signs
   reproduces the same per-atom charges as the full-list path.
4. **Charge projection.** When the caller supplies a non-zero
   ``total_charge``, the per-graph sum matches it exactly after
   projection. The pre-projection diagnostic remains the architectural
   sum.
5. **SO(3) invariance.** Per-atom charges are scalars; a global
   rotation of positions/edge-vectors must not change ``q_i``.
6. **End-to-end into the long-range kernel.** Output ``q_i`` is
   consumed by
   :class:`molpot.potentials.elec.kernels.MultipoleEwaldKernel`
   directly — the kernel is a pure tensor function, and the head
   *calls* it. The shape/dtype contract holds and the resulting energy
   is finite and differentiable through the head MLP.
"""

from __future__ import annotations

import torch

from molpot.heads import BondChargeHead
from molpot.potentials.elec.kernels import MultipoleEwaldKernel


def _bidirectional_edges(n_atoms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full bidirectional edge index for a complete graph on n atoms.

    Returns ``(edge_index, half_index)`` where ``half_index`` selects
    the canonical-direction half of ``edge_index`` (used to validate
    pair-wise antisymmetry).
    """
    pairs = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long)
    half_index = torch.tensor(
        [k for k, (i, j) in enumerate(pairs) if i < j],
        dtype=torch.long,
    )
    return edge_index, half_index


def _make_inputs(n_atoms: int = 5, node_dim: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    pos = torch.randn(n_atoms, 3, dtype=torch.float64)
    node_features = torch.randn(n_atoms, node_dim, dtype=torch.float64)
    edge_index, half_index = _bidirectional_edges(n_atoms)
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1)
    atom_batch = torch.zeros(n_atoms, dtype=torch.long)
    return {
        "pos": pos,
        "node_features": node_features,
        "edge_index": edge_index,
        "half_index": half_index,
        "bond_diff": bond_diff,
        "bond_dist": bond_dist,
        "atom_batch": atom_batch,
    }


def _build_head(node_dim: int = 8, **kwargs) -> BondChargeHead:
    head = BondChargeHead(node_dim=node_dim, hidden_dim=32, **kwargs)
    return head.to(dtype=torch.float64)


def test_neutral_sum_is_zero_without_projection():
    """Architectural antisymmetry forces ``\\sum_i q_i = 0`` to fp precision."""
    head = _build_head(charge_projection=False)
    inp = _make_inputs()
    out = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )
    assert out["atomic_charges"].sum().abs().item() < 1e-12


def test_pair_antisymmetry():
    """``q_{ij} = -q_{ji}`` per edge."""
    head = _build_head(charge_projection=False)
    inp = _make_inputs()
    out = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )
    q_ij = out["bond_charges"]
    edge_index = inp["edge_index"]
    pair_to_idx = {(int(s), int(t)): k for k, (s, t) in enumerate(edge_index.tolist())}
    for (s, t), k in pair_to_idx.items():
        k_rev = pair_to_idx[(t, s)]
        assert torch.allclose(q_ij[k], -q_ij[k_rev], rtol=0.0, atol=1e-12), (
            f"q_{{{s},{t}}} = {q_ij[k]} not equal to -q_{{{t},{s}}} = {-q_ij[k_rev]}"
        )


def test_half_list_matches_full_list():
    """Half-list scatter (+q to src, -q to tgt) must reproduce the full-list q_i."""
    inp = _make_inputs()

    full = _build_head(charge_projection=False, full_neighbor_list=True)
    out_full = full(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )

    half = _build_head(charge_projection=False, full_neighbor_list=False)
    half.load_state_dict(full.state_dict())
    half_edge_index = inp["edge_index"][inp["half_index"]]
    half_bond_dist = inp["bond_dist"][inp["half_index"]]
    out_half = half(
        node_features=inp["node_features"],
        edge_index=half_edge_index,
        bond_dist=half_bond_dist,
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )

    torch.testing.assert_close(
        out_full["atomic_charges"], out_half["atomic_charges"], rtol=1e-10, atol=1e-12
    )
    assert out_half["atomic_charges"].sum().abs().item() < 1e-12


def test_projection_to_nonzero_total_charge():
    """``\\sum_i q_i`` matches a non-zero ``total_charge`` after projection."""
    head = _build_head(charge_projection=True)
    inp = _make_inputs()

    for q_net in (-2.0, -1.0, 0.0, 1.0, 3.0):
        out = head(
            node_features=inp["node_features"],
            edge_index=inp["edge_index"],
            bond_dist=inp["bond_dist"],
            atom_batch=inp["atom_batch"],
            num_graphs=1,
            total_charge=torch.tensor([q_net], dtype=torch.float64),
        )
        torch.testing.assert_close(
            out["atomic_charges"].sum(),
            torch.tensor(q_net, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )
        torch.testing.assert_close(
            out["charge_sum_post_proj"],
            torch.tensor([q_net], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )


def test_diagnostics_record_pre_and_post_sums():
    """``charge_sum_pre_proj`` reports the architectural sum."""
    head = _build_head(charge_projection=True)
    inp = _make_inputs()
    out = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
        total_charge=torch.tensor([1.0], dtype=torch.float64),
    )
    assert out["charge_sum_pre_proj"].shape == (1,)
    assert out["charge_sum_post_proj"].shape == (1,)
    assert out["charge_sum_pre_proj"].abs().item() < 1e-12


def test_rotation_invariance_of_atomic_charges():
    """Per-atom charges are SO(3) scalars — invariant under global rotation."""
    head = _build_head(charge_projection=False)
    inp = _make_inputs()
    out_ref = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )

    theta = torch.tensor(0.7, dtype=torch.float64)
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    bond_diff_rot = inp["bond_diff"] @ R.T
    bond_dist_rot = bond_diff_rot.norm(dim=-1)
    out_rot = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=bond_dist_rot,
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )

    torch.testing.assert_close(
        out_rot["atomic_charges"], out_ref["atomic_charges"], rtol=1e-12, atol=1e-13
    )


def test_batched_two_graphs_each_neutral():
    """Two graphs in one batch — each graph's sum must be 0 independently."""
    head = _build_head(charge_projection=False)
    inp_a = _make_inputs(n_atoms=4, seed=0)
    inp_b = _make_inputs(n_atoms=3, seed=1)

    n_a = 4
    n_b = 3
    node_features = torch.cat([inp_a["node_features"], inp_b["node_features"]], dim=0)
    edge_b_shifted = inp_b["edge_index"] + n_a
    edge_index = torch.cat([inp_a["edge_index"], edge_b_shifted], dim=0)
    bond_dist = torch.cat([inp_a["bond_dist"], inp_b["bond_dist"]], dim=0)
    atom_batch = torch.cat(
        [torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)],
        dim=0,
    )

    out = head(
        node_features=node_features,
        edge_index=edge_index,
        bond_dist=bond_dist,
        atom_batch=atom_batch,
        num_graphs=2,
    )
    sum_per_graph = out["charge_sum_pre_proj"]
    assert sum_per_graph.shape == (2,)
    assert sum_per_graph.abs().max().item() < 1e-12


def test_head_calls_kernel_directly():
    """``BondChargeHead`` output ``q_i`` is consumed by the pure kernel.

    Demonstrates the head → kernel separation: the head emits charges
    (no electrostatic logic), and the standalone
    :class:`MultipoleEwaldKernel` consumes them with no head dependency.
    Validates the q-q + q-μ + μ-μ kernels are exercised end-to-end and
    that gradients flow through the head MLP via the kernel.
    """
    head = _build_head(charge_projection=False).double()
    kernel = MultipoleEwaldKernel(sigma=1.0, remove_self_interaction=True).double()

    inp = _make_inputs(n_atoms=5)
    inp["node_features"].requires_grad_(True)
    out = head(
        node_features=inp["node_features"],
        edge_index=inp["edge_index"],
        bond_dist=inp["bond_dist"],
        atom_batch=inp["atom_batch"],
        num_graphs=1,
    )
    q = out["atomic_charges"]

    torch.manual_seed(7)
    mu = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    pos = inp["pos"].clone().requires_grad_(True)

    result = kernel.compute_realspace(pos=pos, q=q, mu=mu)
    pot = result["pot"]

    assert torch.isfinite(pot).all()
    grad_pos, grad_mu, grad_feat = torch.autograd.grad(pot, [pos, mu, inp["node_features"]])
    assert torch.isfinite(grad_pos).all()
    assert torch.isfinite(grad_mu).all()
    assert torch.isfinite(grad_feat).all()
    assert grad_feat.abs().max() > 0.0
