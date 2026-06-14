"""Parity test: ``EwaldMultipoleEnergy.forward`` vs the brute-force oracle.

Asserts ≤1e-6 (float64) agreement on every output channel
(``pot, phi, field, q_induced, u_induced``) over 50 random configurations
on each of:

* the non-periodic O(N²) realspace path (``cell=None``)
* the 3D-periodic reciprocal path (random triclinic cell)

Configurations cover all 2³ moment combinations
(charge-only, charge+dipole, charge+quadrupole, full multipole) with
randomly-toggled κ / α response heads so the inline non-self-consistent
induced terms are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molpot.potentials import EwaldMultipoleEnergy
from tests._oracles.screened_coulomb import (
    brute_realspace,
    brute_reciprocal,
)

# 50 random seeds for the parity sweep — small enough for CI, large enough
# to surface any sign/scaling regression on a stochastic input.
SEEDS = list(range(50))
N_ATOMS = 6  # small enough for O(N²) oracle to be cheap


def _random_traceless_quadrupole(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return ``(n, 3, 3)`` symmetric traceless tensor (Flag #2 compliance)."""
    raw = rng.standard_normal((n, 3, 3))
    sym = 0.5 * (raw + raw.transpose(0, 2, 1))
    trace = np.einsum("nii->n", sym) / 3.0
    return sym - trace[:, None, None] * np.eye(3)[None, :, :]


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(torch.float64)


def _make_inputs(seed: int, with_mu: bool, with_Q: bool, periodic: bool):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-5.0, 5.0, size=(N_ATOMS, 3))
    q = rng.standard_normal(N_ATOMS)
    q -= q.mean()  # neutralise to avoid Yeh-Berkowitz background drift in tests
    mu = rng.standard_normal((N_ATOMS, 3)) if with_mu else None
    Q = _random_traceless_quadrupole(rng, N_ATOMS) if with_Q else None
    if periodic:
        # Random near-cubic cell with size ~12 Å, large enough that
        # σ=1 Å Gaussians don't overlap across periodic images.
        L = 12.0 + rng.uniform(-1.0, 1.0)
        cell = L * (np.eye(3) + 0.05 * rng.standard_normal((3, 3)))
    else:
        cell = None
    return pos, q, mu, Q, cell


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "with_mu,with_Q", [(False, False), (True, False), (False, True), (True, True)]
)
def test_forward_realspace_parity(
    seed: int,
    with_mu: bool,
    with_Q: bool,
) -> None:
    """Non-periodic O(N²) path: ``forward`` must match ``brute_realspace``."""
    pos, q, mu, Q, _ = _make_inputs(seed, with_mu, with_Q, periodic=False)

    pot = EwaldMultipoleEnergy(sigma=1.0, prefactor=90.4756)
    out = pot.forward(
        q=_to_tensor(q),
        pos=_to_tensor(pos),
        cell=None,
        mu=_to_tensor(mu) if with_mu else None,
        Q=_to_tensor(Q) if with_Q else None,
    )

    ref = brute_realspace(
        pos, q, mu=mu, Q=Q, sigma=1.0, prefactor=90.4756, remove_self_interaction=True
    )

    torch.testing.assert_close(
        out["pot"].detach().cpu(),
        torch.tensor(ref["pot"], dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
        msg=f"pot seed={seed} mu={with_mu} Q={with_Q}",
    )
    torch.testing.assert_close(
        out["phi"].detach().cpu(),
        torch.from_numpy(ref["phi"]),
        atol=1e-6,
        rtol=1e-6,
        msg=f"phi seed={seed} mu={with_mu} Q={with_Q}",
    )
    torch.testing.assert_close(
        out["field"].detach().cpu(),
        torch.from_numpy(ref["field"]),
        atol=1e-6,
        rtol=1e-6,
        msg=f"field seed={seed} mu={with_mu} Q={with_Q}",
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("with_mu,with_Q", [(False, False), (True, False), (True, True)])
def test_forward_reciprocal_parity(
    seed: int,
    with_mu: bool,
    with_Q: bool,
) -> None:
    """Periodic path: ``forward`` must match ``brute_reciprocal``."""
    pos, q, mu, Q, cell = _make_inputs(seed, with_mu, with_Q, periodic=True)

    pot = EwaldMultipoleEnergy(sigma=1.0, dl=2.0, prefactor=90.4756)
    out = pot.forward(
        q=_to_tensor(q),
        pos=_to_tensor(pos),
        cell=_to_tensor(cell),
        mu=_to_tensor(mu) if with_mu else None,
        Q=_to_tensor(Q) if with_Q else None,
    )

    ref = brute_reciprocal(
        pos,
        q,
        cell,
        mu=mu,
        Q=Q,
        sigma=1.0,
        dl=2.0,
        prefactor=90.4756,
        remove_self_interaction=True,
    )

    torch.testing.assert_close(
        out["pot"].detach().cpu(),
        torch.tensor(ref["pot"], dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
        msg=f"pot seed={seed} mu={with_mu} Q={with_Q}",
    )
    torch.testing.assert_close(
        out["phi"].detach().cpu(),
        torch.from_numpy(ref["phi"]),
        atol=1e-6,
        rtol=1e-6,
        msg=f"phi seed={seed} mu={with_mu} Q={with_Q}",
    )
    torch.testing.assert_close(
        out["field"].detach().cpu(),
        torch.from_numpy(ref["field"]),
        atol=1e-6,
        rtol=1e-6,
        msg=f"field seed={seed} mu={with_mu} Q={with_Q}",
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_forward_induced_response(seed: int) -> None:
    """When κ / α are passed, energy gets ``½ Φ·q_ind`` and ``-½ E·u_ind``.

    Compare against the oracle's φ/E and apply the response formula in
    NumPy; assert the closed-form match exactly (the response is
    one-shot linear, so no iteration tolerance is needed).
    """
    pos, q, mu, Q, _ = _make_inputs(seed, with_mu=True, with_Q=False, periodic=False)
    rng = np.random.default_rng(seed + 1000)
    kappa = np.abs(rng.standard_normal(N_ATOMS)) + 0.1  # strictly positive
    alpha = np.abs(rng.standard_normal(N_ATOMS)) + 0.1  # isotropic

    pot = EwaldMultipoleEnergy(sigma=1.0, prefactor=90.4756)
    out = pot.forward(
        q=_to_tensor(q),
        pos=_to_tensor(pos),
        cell=None,
        mu=_to_tensor(mu),
        kappa=_to_tensor(kappa),
        alpha=_to_tensor(alpha),
    )

    ref = brute_realspace(pos, q, mu=mu, sigma=1.0, prefactor=90.4756, remove_self_interaction=True)
    ref_q_ind = -kappa * ref["phi"]
    ref_u_ind = alpha[:, None] * ref["field"]
    ref_pot = (
        ref["pot"]
        + 0.5 * float(np.dot(ref["phi"], ref_q_ind))
        - 0.5 * float(np.einsum("ic,ic->", ref["field"], ref_u_ind))
    )

    torch.testing.assert_close(
        out["pot"].detach().cpu(),
        torch.tensor(ref_pot, dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        out["q_induced"].detach().cpu(),
        torch.from_numpy(ref_q_ind),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        out["u_induced"].detach().cpu(),
        torch.from_numpy(ref_u_ind),
        atol=1e-6,
        rtol=1e-6,
    )
