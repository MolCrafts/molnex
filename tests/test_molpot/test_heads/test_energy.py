"""Tests for the per-atom energy heads in :mod:`molpot.heads.energy`.

``TestEnergyHead.test_forward_energy`` was moved from the stale
``tests/test_molpot/test_readout/`` mirror — the production module is
``src/molpot/heads/energy.py``, so the mirror is
``tests/test_molpot/test_heads/test_energy.py``.
"""

from __future__ import annotations

import torch

from molpot.heads.energy import AtomicEnergyMLP, AtomicReferenceEnergy, EnergyHead


class TestAtomicEnergyMLP:
    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the head mixed-precision
        and its first forward dies on a dtype-mismatched matmul.
        """
        head = AtomicEnergyMLP(hidden_dim=4)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


class TestEnergyHead:
    def test_forward_energy(self):
        head = EnergyHead(hidden_dim=4)
        h = torch.ones(3, 4)
        batch = torch.tensor([0, 0, 1])
        out = head(h, batch)
        assert out.shape == torch.Size([2])

    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64."""
        head = EnergyHead(hidden_dim=4)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64):
        """A head built at fp64 pools fp64 atomic energies into fp64 totals."""
        head = EnergyHead(hidden_dim=4)

        out = head(torch.ones(3, 4, dtype=torch.float64), torch.tensor([0, 0, 1]))

        assert out.shape == torch.Size([2])
        assert out.dtype == torch.float64


class TestAtomicReferenceEnergy:
    def test_lookup_buffer_honours_the_fp64_precision(self, fp64):
        """The Z-indexed ``E0`` buffer follows ``config["ftype"]``.

        Already honoured by the production code (the buffer is built with
        ``dtype=config.ftype``); pinned here so the dtype contract of the
        module is covered end to end.
        """
        head = AtomicReferenceEnergy(
            atomic_energies=[-13.6, -1000.0],
            atomic_numbers=[1, 8],
        )

        assert head.atomic_energies.dtype == torch.float64
