"""Tests for :class:`molpot.heads.charge_response.ChargeResponseHead`.

Scope: the construction-time dtype contract. ``sigma_raw`` and
``edge_vector_mlp`` are built at the constructor dtype; the scalar MLPs
(``atom_diag_mlp`` / ``edge_scalar_mlp`` / ``iso_mlp``) must be too, or the
head comes out mixed-precision at fp64.
"""

from __future__ import annotations

import pytest
import torch

from molpot.heads.charge_response import ChargeResponseHead

#: Floating dtype every module under test is built at.
DTYPE = torch.float64


def _make_head(*, variant: str = "localchi", iso: bool = False) -> ChargeResponseHead:
    """A minimal head — the dtype contract is size-independent."""
    return ChargeResponseHead(
        node_scalar_dim=4,
        edge_scalar_dim=3,
        edge_vector_dim=2,
        hidden_dim=8,
        variant=variant,
        iso=iso,
        dtype=DTYPE,
    )


class TestChargeResponseHead:
    @pytest.mark.parametrize(
        "variant, iso",
        [("localchi", False), ("localchi", True), ("eem", True)],
        ids=["localchi", "localchi-iso", "eem-iso"],
    )
    def test_all_parameters_are_built_at_the_given_dtype(self, variant, iso):
        """Every parameter is fp64 when the head is built at fp64.

        A layer that ignored the constructor dtype would leave the head
        mixed-precision, and its first forward would die on a dtype-mismatched
        matmul. The
        ``iso`` cases additionally cover the optional ``iso_mlp`` branch.
        """
        head = _make_head(variant=variant, iso=iso)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_at_the_given_dtype(self):
        """A head built at fp64 turns fp64 features into an fp64 polarizability."""
        head = _make_head()

        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        pair_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = head(
            pos=pos,
            Z=torch.tensor([1, 8], dtype=torch.long),
            atom_batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            pair_index=pair_index,
            pair_diff=pos[pair_index[:, 1]] - pos[pair_index[:, 0]],
            node_scalars=torch.ones(2, 4, dtype=torch.float64),
            edge_scalars=torch.ones(2, 3, dtype=torch.float64),
        )

        assert out["alpha"].shape == (1, 3, 3)
        assert out["alpha"].dtype == torch.float64
        assert out["atom_diag"].dtype == torch.float64
