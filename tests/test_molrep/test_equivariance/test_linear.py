"""The GEMM-per-irrep-block linear map against cuEquivariance's descriptor."""

from __future__ import annotations

import pytest
import torch

import molrep.eq as meq
import molrep.eq.torch as cuet
from molrep.embedding.angular import SphericalHarmonics
from molrep.equivariance import Linear

IRREPS = meq.Irreps("O3", "4x0e+4x1o")


class TestLinear:
    def test_matches_cuequivariance_with_the_same_weight(self):
        """One GEMM per irrep block reproduces cue's linear descriptor — the
        ``1/√u`` path coefficients included — and an output irrep with no
        input partner (``3o`` here) comes out zero."""
        irreps_in = meq.Irreps("O3", "4x0e+4x1o+4x2e")
        irreps_out = meq.Irreps("O3", "4x0e+4x1o+4x3o")
        torch.manual_seed(0)
        ours = Linear(irreps_in, irreps_out, layout=meq.ir_mul, dtype=torch.float64)
        theirs = cuet.Linear(
            irreps_in, irreps_out, layout=meq.ir_mul, dtype=torch.float64, method="naive"
        )
        with torch.no_grad():
            theirs.weight.copy_(ours.weight)
        x = torch.randn(7, irreps_in.dim, dtype=torch.float64)
        expected = theirs(x)
        got = ours(x)
        assert got.shape == (7, irreps_out.dim)
        assert torch.allclose(got, expected, atol=1e-12, rtol=0.0)
        assert torch.count_nonzero(got[:, -4 * 7 :]) == 0

    def test_state_dict_is_the_flat_cuequivariance_weight(self):
        """Checkpoints written for ``cuet.Linear`` load unchanged: one ``weight``
        of cue's ``(1, weight_numel)`` shape and nothing else."""
        ours = Linear(IRREPS, IRREPS, layout=meq.ir_mul, dtype=torch.float32)
        theirs = cuet.Linear(IRREPS, IRREPS, layout=meq.ir_mul, dtype=torch.float32)
        assert set(ours.state_dict()) == {"weight"}
        assert ours.weight.shape == theirs.weight.shape == (1, theirs.weight_numel)
        assert ours.weight_numel == theirs.weight_numel

    def test_rejects_a_layout_other_than_ir_mul(self):
        with pytest.raises(ValueError, match="ir_mul"):
            Linear(IRREPS, IRREPS, layout=meq.mul_ir, dtype=torch.float32)

    def test_force_loss_reaches_the_weights(self):
        """``g = ∂E/∂x`` with ``create_graph``; backprop of ``g²`` must reach the ``l>0`` weights.

        The energy must depend on ``|r|`` (``normalize=False`` on the spherical
        harmonics): a single normalised direction has constant rotation
        invariants, so ``∂E/∂x`` would vanish identically.
        """
        torch.manual_seed(0)
        sh = SphericalHarmonics(l_max=2, normalize=False, dtype=torch.float64)
        linear = Linear(
            meq.Irreps("O3", "1x0e+1x1o+1x2e"),
            meq.Irreps("O3", "4x0e+4x1o+4x2e"),
            layout=meq.ir_mul,
            dtype=torch.float64,
        )
        pos = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        energy = linear(sh(pos)).pow(2).sum()
        force = torch.autograd.grad(energy, pos, create_graph=True)[0]
        assert force.detach().abs().max() > 0.0
        (force**2).mean().backward()
        assert linear.weight.grad is not None and linear.weight.grad.abs().sum() > 0.0
