"""Charge-projection tests for :class:`molpot.heads.PermMultipoleHead`.

Covers the fail-fast contract on ``("graphs", "total_charge")`` and the
projection behavior on charged systems (``q_target ≠ 0``).
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molpot.heads import PermMultipoleHead


def _stub_batch(
    *,
    n_atoms: int = 4,
    n_edges: int = 8,
    feat_dim: int = 8,
    total_charge: float | None = 0.0,
) -> TensorDict:
    """Single-graph dummy batch wired through the keys PermMultipoleHead reads.

    Pass ``total_charge=None`` to omit the ``("graphs", "total_charge")``
    key entirely (used for the fail-fast test).
    """
    torch.manual_seed(0)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [0, 3], [3, 0]],
        dtype=torch.long,
    )[:n_edges]
    atoms = TensorDict(
        Z=torch.tensor([1, 6, 8, 1][:n_atoms]),
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        batch_size=[n_atoms],
    )
    edges = TensorDict(
        edge_index=edge_index,
        bond_diff=torch.randn(n_edges, 3),
        bond_dist=torch.rand(n_edges) + 0.5,
        batch_size=[n_edges],
    )
    edges["edge_features"] = torch.randn(n_edges, feat_dim)
    graphs = TensorDict(num_atoms=torch.tensor([n_atoms]), batch_size=[1])
    if total_charge is not None:
        graphs["total_charge"] = torch.tensor([total_charge])
    return TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])


class TestProjectionToTarget:
    def test_neutral_target(self):
        """Σq_proj must equal q_target=0 (within numerical precision)."""
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0)
        out = head(_stub_batch(total_charge=0.0))
        sum_q = out["atomic_charges"].sum()
        assert sum_q.abs().item() < 1e-5

    def test_anion_target_minus_one(self):
        """Σq_proj must equal q_target=-1."""
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0)
        out = head(_stub_batch(total_charge=-1.0))
        sum_q = out["atomic_charges"].sum()
        torch.testing.assert_close(sum_q, torch.tensor(-1.0), rtol=1e-5, atol=1e-5)

    def test_cation_target_plus_two(self):
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0)
        out = head(_stub_batch(total_charge=2.0))
        sum_q = out["atomic_charges"].sum()
        torch.testing.assert_close(sum_q, torch.tensor(2.0), rtol=1e-5, atol=1e-5)

    def test_diagnostics_record_pre_and_post_sums(self):
        """``charge_sum_pre_proj`` is the raw sum, ``post`` is the projected one."""
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0)
        out = head(_stub_batch(total_charge=-1.0))
        assert "charge_sum_pre_proj" in out
        assert "charge_sum_post_proj" in out
        # Post-projection sum must match the target.
        torch.testing.assert_close(
            out["charge_sum_post_proj"],
            torch.tensor([-1.0]),
            rtol=1e-5,
            atol=1e-5,
        )


class TestFailFastOnMissingTotalCharge:
    def test_missing_total_charge_raises(self):
        """Without ``ConstantLabel`` the dataset omits this key → KeyError."""
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0)
        with pytest.raises(KeyError):
            head(_stub_batch(total_charge=None))

    def test_constrain_off_does_not_require_total_charge(self):
        """When projection is disabled, the field isn't read at all."""
        head = PermMultipoleHead(input_dim=8, avg_num_neighbors=4.0, constrain_total_charge=False)
        # Should not raise even though ``total_charge`` is absent.
        out = head(_stub_batch(total_charge=None))
        assert "atomic_charges" in out
        # No projection diagnostics emitted in this mode.
        assert "charge_sum_pre_proj" not in out
        assert "charge_sum_post_proj" not in out
