"""Tests for :class:`molpot.heads.dipole.DipoleHead`.

Scope: the construction-time dtype contract. ``atomic_dipole_gate`` and
``bond_vector_mlp`` are built at the constructor dtype; the scalar MLPs
(``charge_mlp`` / ``bond_scalar_mlp``) must be too, or the head comes out
mixed-precision at fp64.
"""

from __future__ import annotations

import pytest
import torch

from molpot.heads.dipole import DipoleHead

#: Floating dtype every module under test is built at.
DTYPE = torch.float64


def _make_head(variant: str) -> DipoleHead:
    """A minimal head with every optional dim supplied, so ``variant`` alone
    decides which submodules exist."""
    return DipoleHead(
        node_scalar_dim=4,
        node_vector_dim=2,
        edge_scalar_dim=3,
        edge_vector_dim=2,
        hidden_dim=8,
        variant=variant,
        dtype=DTYPE,
    )


class TestDipoleHead:
    @pytest.mark.parametrize(
        "variant",
        ["ac", "ad", "bc", "ac_ad", "ac_bc"],
        ids=["ac", "ad", "bc", "ac_ad", "ac_bc"],
    )
    def test_all_parameters_are_built_at_the_given_dtype(self, variant):
        """Every parameter is fp64 when the head is built at fp64.

        A layer that ignored the constructor dtype would leave the head
        mixed-precision, and its first forward would die on a dtype-mismatched
        matmul. The
        variants select which term MLPs are constructed, so the sweep
        covers each submodule the head can own.
        """
        head = _make_head(variant)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_at_the_given_dtype(self):
        """A head built at fp64 assembles an fp64 dipole from fp64 features."""
        head = _make_head("ac_bc")

        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        pair_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = head(
            pos=pos,
            atom_batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            node_scalars=torch.ones(2, 4, dtype=torch.float64),
            edge_scalars=torch.ones(2, 3, dtype=torch.float64),
            edge_vectors=torch.ones(2, 3, 2, dtype=torch.float64),
            pair_index=pair_index,
            pair_diff=pos[pair_index[:, 1]] - pos[pair_index[:, 0]],
        )

        assert out["dipole"].shape == (1, 3)
        assert out["dipole"].dtype == torch.float64
        assert out["atomic_charges"].dtype == torch.float64
