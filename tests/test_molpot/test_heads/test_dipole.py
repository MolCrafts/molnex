"""Tests for :class:`molpot.heads.dipole.DipoleHead`.

Scope: the construction-time dtype contract. ``atomic_dipole_gate`` and
``bond_vector_mlp`` already read ``config["ftype"]``; the scalar MLPs
(``charge_mlp`` / ``bond_scalar_mlp``) must do the same or the head comes
out mixed-precision under ``config.set_precision("fp64")``.
"""

from __future__ import annotations

import pytest
import torch

from molpot.heads.dipole import DipoleHead


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
    )


class TestDipoleHead:
    @pytest.mark.parametrize(
        "variant",
        ["ac", "ad", "bc", "ac_ad", "ac_bc"],
        ids=["ac", "ad", "bc", "ac_ad", "ac_bc"],
    )
    def test_all_parameters_honour_the_fp64_precision(self, fp64, variant):
        """Every parameter is fp64 when the head is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the head mixed-precision
        and its first forward dies on a dtype-mismatched matmul. The
        variants select which term MLPs are constructed, so the sweep
        covers each submodule the head can own.
        """
        head = _make_head(variant)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64):
        """A head built at fp64 assembles an fp64 dipole from fp64 features."""
        head = _make_head("ac_bc")

        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = head(
            pos=pos,
            atom_batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            node_scalars=torch.ones(2, 4, dtype=torch.float64),
            edge_scalars=torch.ones(2, 3, dtype=torch.float64),
            edge_vectors=torch.ones(2, 3, 2, dtype=torch.float64),
            edge_index=edge_index,
            edge_diff=pos[edge_index[:, 1]] - pos[edge_index[:, 0]],
        )

        assert out["dipole"].shape == (1, 3)
        assert out["dipole"].dtype == torch.float64
        assert out["atomic_charges"].dtype == torch.float64
