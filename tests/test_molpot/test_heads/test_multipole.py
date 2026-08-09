"""Tests for :class:`molpot.heads.multipole.PermMultipoleHead`.

Scope: the construction-time dtype contract. ``q_head`` / ``mu_proj`` /
``theta_proj`` already read ``config["ftype"]``; the two ``cuet.Linear``
moment-collapse layers (``mu_collapse`` / ``theta_collapse``) must do the
same or the head comes out mixed-precision under
``config.set_precision("fp64")`` and its first forward dies on the l=1 / l=2
collapse with ``expected scalar type Float but found Double``.
"""

from __future__ import annotations

import cuequivariance as cue
import pytest
import torch
from tensordict import TensorDict

from molpot.heads import PermMultipoleHead

#: Encoder tensor-track irreps with the uniform multiplicity the head
#: requires, carrying both the ``1o`` block the dipole readout slices and
#: the ``2e`` block the quadrupole readout slices.
TENSOR_IRREPS = cue.Irreps(cue.O3, [(4, "0e"), (4, "1o"), (4, "2e")])

#: Which ``(charge, dipole, quadrupole)`` combinations to sweep. ``dipole``
#: and ``quadrupole`` are the flags that construct a ``cuet.Linear``.
MOMENT_CASES: list[tuple[str, ...]] = [
    ("charge",),
    ("charge", "dipole"),
    ("charge", "quadrupole"),
    ("charge", "dipole", "quadrupole"),
    ("dipole", "quadrupole"),
]


def _make_head(moments: tuple[str, ...]) -> PermMultipoleHead:
    """A minimal head with only ``moments`` enabled."""
    return PermMultipoleHead(
        input_dim=8,
        hidden_dim=8,
        avg_num_neighbors=4.0,
        charge="charge" in moments,
        dipole="dipole" in moments,
        quadrupole="quadrupole" in moments,
        tensor_irreps=TENSOR_IRREPS,
    )


def _stub_batch(dtype: torch.dtype) -> TensorDict:
    """Single-graph batch wired through every key the head reads.

    Three atoms, four (bidirectional) edges, with both the scalar and the
    tensor edge tracks present so the q / μ / Θ readouts all run.
    """
    n_atoms, n_edges = 3, 4
    atoms = TensorDict(
        Z=torch.tensor([1, 6, 8]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
        ),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        batch_size=[n_atoms],
    )
    edges = TensorDict(
        edge_index=torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long),
        batch_size=[n_edges],
    )
    edges["edge_features"] = torch.ones(n_edges, 8, dtype=dtype)
    edges["edge_tensor_features"] = torch.ones(n_edges, TENSOR_IRREPS.dim, dtype=dtype)
    graphs = TensorDict(
        num_atoms=torch.tensor([n_atoms]),
        total_charge=torch.zeros(1, dtype=dtype),
        batch_size=[1],
    )
    return TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])


class TestPermMultipoleHead:
    @pytest.mark.parametrize(
        "moments",
        MOMENT_CASES,
        ids=["q", "q_mu", "q_theta", "q_mu_theta", "mu_theta"],
    )
    def test_all_parameters_honour_the_fp64_precision(
        self, fp64: None, moments: tuple[str, ...]
    ) -> None:
        """Every parameter is fp64 when the head is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the head mixed-precision
        and its first forward dies on a dtype-mismatched matmul. The sweep
        covers each moment flag, so both ``cuet.Linear`` collapse weights
        (``mu_collapse`` for l=1, ``theta_collapse`` for l=2) are exercised.
        """
        head = _make_head(moments)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64: None) -> None:
        """A head built at fp64 reads fp64 features and emits fp64 moments."""
        head = _make_head(("charge", "dipole", "quadrupole"))

        out = head(_stub_batch(torch.float64))

        assert out["atomic_charges"].shape == (3,)
        assert out["atomic_charges"].dtype == torch.float64
        assert out["atomic_dipoles"].shape == (3, 3)
        assert out["atomic_dipoles"].dtype == torch.float64
        assert out["atomic_quadrupoles"].shape == (3, 5)
        assert out["atomic_quadrupoles"].dtype == torch.float64
        assert out["molecular_dipole"].dtype == torch.float64
