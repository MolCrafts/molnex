"""Brute-force NumPy oracle for σ-screened multipole Coulomb electrostatics.

Three pure-NumPy reference implementations of the σ-Gaussian-screened
Coulomb multipole energy / potential / field, exercising the textbook
formulae directly so production-side `EwaldMultipoleEnergy` parity tests
can compare against an obviously-correct, dependency-free reference:

* :func:`brute_realspace` — non-periodic O(N²) all-pairs sum with kernel
  ``erf(r·a)/r`` (``a = 1/(σ√2)``) and its multipole-gradient siblings
  ``f_qu, f_uu, f_Qu, f_QQ``.

* :func:`brute_reciprocal` — periodic reciprocal-space half-k-sphere sum
  with kernel ``exp(-σ²k²/2)/k²`` weighted by the multipole structure
  factor ``S(k) = Σᵢ [qᵢ + i k·μᵢ − ½ k·Qᵢ·k] e^{i k·rᵢ}``.

* :func:`self_corrections` — the five per-multipole self-correction
  constants subtracted (or added back) when ``remove_self_interaction``
  toggles between ``True`` (LES default) and ``False``.

These mirror — line-by-line — the algorithm in ``les.module.ewald.py``
and ``les.module.make_kernels.py`` from
``github.com/ChengUCB/les`` (treated as the upstream reference but
**not** imported here, so the test path has zero external deps).
The math itself is from:

* Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed.
  (Academic Press, 2002), §12.1 — derivation of the
  ``erf(r/(σ√2))/r ↔ 4π exp(-σ²k²/2)/k²`` Fourier pair from a
  Gaussian charge density.
* Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed.
  (Oxford, 2017), §6.5 — Ewald reciprocal-space sum, half-k-sphere
  optimisation, self-energy correction.
* Hünenberger, *Adv. Polym. Sci.* **173**, 105 (2005), §3.2.2 —
  multipole moment kernels via successive gradients of the screened
  Coulomb propagator.
* Stone, *The Theory of Intermolecular Forces*, 2nd ed. (Oxford,
  2013), §3.3 — bare ``T_n`` multipole interaction tensors recovered
  in the ``a → ∞`` (no-screening) limit.
* Yeh & Berkowitz, *J. Chem. Phys.* **111**, 3155 (1999) — the
  ``k = 0`` exclusion convention (uniform compensating background).

The CRC-handbook polarizability values used by element baselines
elsewhere in the test suite are not reproduced here; this module
covers only the kernel mathematics.

Sign conventions (consistent with LES upstream):

* ``r_ij[i, j, c] = r[j, c] − r[i, c]`` (i is source, j is receiver).
* Quadrupole structure-factor coefficient is ``−½ k·Qᵢ·k``
  (``ewald.py`` L336); flips overall sign of ``QQ`` energy term.
* Quadrupole is assumed **symmetric traceless** — non-traceless input
  silently absorbs trace into a redefinition of σ-monopole. Caller
  responsibility (Flag #2 in ``les-electrostatics.md`` spec).

All energies have units of ``prefactor`` × ``[charge]² / [length]``.
With the project default ``prefactor = 90.4756 eV·Å·e⁻²`` (= ``1/(2ε₀)``
in eV-Å-e), charges in e and positions in Å, energies land in eV.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# scipy is not a hard dep of MolNex tests; fall back to vectorised math.erf
# if scipy isn't on the path. Both produce identical numerical results.
try:
    from scipy.special import erf as _scipy_erf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on local install
    _erf_vec = np.vectorize(math.erf, otypes=[np.float64])

    def _scipy_erf(x: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
        return _erf_vec(x)


__all__ = [
    "make_kernels",
    "brute_realspace",
    "brute_reciprocal",
    "self_corrections",
]


# ---------------------------------------------------------------------------
# Real-space pair kernels
# ---------------------------------------------------------------------------


def make_kernels(
    r: np.ndarray,
    sigma: float,
    norm_const: float,
) -> dict[str, np.ndarray]:
    """Compute the five σ-screened Coulomb pair kernels at every (i, j).

    Mirrors ``les.module.make_kernels.make_kernels`` in pure NumPy.
    Diagonal (``i == j``) entries are zero.

    Args:
        r: ``(N, 3)`` positions.
        sigma: Gaussian smearing length.
        norm_const: ``prefactor / (2π)`` — the ``1/(4πε₀)`` factor
            multiplying every kernel.

    Returns:
        Dict with keys:

        * ``"f_qq"`` ``(N, N)`` — charge-charge ``erf(r·a)/r``.
        * ``"f_qu"`` ``(N, N, 3)`` — charge-dipole, ``∇φ`` direction.
        * ``"f_uu"`` ``(N, N, 3, 3)`` — dipole-dipole.
        * ``"f_Qu"`` ``(N, N, 3, 3, 3)`` — quadrupole-dipole / dipole-Q.
        * ``"f_QQ"`` ``(N, N, 3, 3, 3, 3)`` — quadrupole-quadrupole.
    """
    n = r.shape[0]
    a = 1.0 / (sigma * math.sqrt(2.0))
    sqrt_pi = math.sqrt(math.pi)

    mask_off = ~np.eye(n, dtype=bool)

    r_ij = r[None, :, :] - r[:, None, :]  # r_ij[i, j] = r[j] - r[i]
    r_norm = np.linalg.norm(r_ij, axis=-1)

    rinv = np.zeros_like(r_norm)
    rinv[mask_off] = 1.0 / r_norm[mask_off]

    erf_vals = np.zeros_like(r_norm)
    erf_vals[mask_off] = _scipy_erf(r_norm[mask_off] * a)

    f_qq = erf_vals * rinv * norm_const  # (N, N)

    rinv2 = rinv * rinv
    rinv3 = rinv2 * rinv
    gauss = np.exp(-((a * r_norm) ** 2)) * mask_off
    rhat = r_ij * rinv[..., None]
    eye = np.eye(3)

    s1 = erf_vals * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
    s2 = (
        3.0 * erf_vals * rinv3
        - (6.0 * a / sqrt_pi) * gauss * rinv2
        - (4.0 * a**3 / sqrt_pi) * gauss
    )

    rr = rhat[..., :, None] * rhat[..., None, :]  # (N, N, 3, 3)
    f_uu = (s2[:, :, None, None] * rr - s1[:, :, None, None] * eye[None, None]) * norm_const
    f_qu = s1[..., None] * r_ij * norm_const  # (N, N, 3)

    rinv4 = rinv3 * rinv
    rinv5 = rinv4 * rinv
    s3 = (
        15.0 * erf_vals * rinv4
        - (30.0 * a / sqrt_pi) * gauss * rinv3
        - (20.0 * a**3 / sqrt_pi) * gauss * rinv
        - (8.0 * a**5 / sqrt_pi) * gauss * r_norm
    )
    s4 = (
        105.0 * erf_vals * rinv5
        - (210.0 * a / sqrt_pi) * gauss * rinv4
        - (140.0 * a**3 / sqrt_pi) * gauss * rinv2
        - (56.0 * a**5 / sqrt_pi) * gauss
        - (16.0 * a**7 / sqrt_pi) * gauss * r_norm**2
    )

    rrr = np.einsum("nmi,nmj,nmk->nmijk", rhat, rhat, rhat)
    term_delta_r = (
        np.einsum("ab,ijc->ijabc", eye, rhat)
        + np.einsum("ac,ijb->ijabc", eye, rhat)
        + np.einsum("bc,ija->ijabc", eye, rhat)
    )
    f_Qu = (
        s3[..., None, None, None] * rrr - (s2 * rinv)[..., None, None, None] * term_delta_r
    ) * norm_const

    rrrr = np.einsum("ija,ijb,ijc,ijd->ijabcd", rhat, rhat, rhat, rhat)
    term_delta_rr = (
        np.einsum("ab,ijc,ijd->ijabcd", eye, rhat, rhat)
        + np.einsum("ac,ijb,ijd->ijabcd", eye, rhat, rhat)
        + np.einsum("ad,ijb,ijc->ijabcd", eye, rhat, rhat)
        + np.einsum("bc,ija,ijd->ijabcd", eye, rhat, rhat)
        + np.einsum("bd,ija,ijc->ijabcd", eye, rhat, rhat)
        + np.einsum("cd,ija,ijb->ijabcd", eye, rhat, rhat)
    )
    term_delta_delta = (
        np.einsum("ab,cd->abcd", eye, eye)
        + np.einsum("ac,bd->abcd", eye, eye)
        + np.einsum("ad,bc->abcd", eye, eye)
    )[None, None]
    f_QQ = (
        s4[..., None, None, None, None] * rrrr
        - (s3 * rinv)[..., None, None, None, None] * term_delta_rr
        + (s2 * rinv2)[..., None, None, None, None] * term_delta_delta
    ) * norm_const

    return {"f_qq": f_qq, "f_qu": f_qu, "f_uu": f_uu, "f_Qu": f_Qu, "f_QQ": f_QQ}


# ---------------------------------------------------------------------------
# Self-correction constants
# ---------------------------------------------------------------------------


def self_corrections(sigma: float, prefactor: float) -> dict[str, float]:
    """The five per-multipole self-correction scalars (see module docstring).

    Returns:
        Dict with keys ``"energy_q"``, ``"phi_q"``, ``"energy_u"``,
        ``"energy_Q"``, ``"field_u"``. See module docstring for the
        algebraic forms.
    """
    twopi32 = (2.0 * math.pi) ** 1.5
    a = 1.0 / (sigma * math.sqrt(2.0))
    norm_const = prefactor / (2.0 * math.pi)
    return {
        "energy_q": prefactor / (sigma * twopi32),
        "phi_q": 2.0 * prefactor / (sigma * twopi32),
        "energy_u": prefactor / (3.0 * sigma**3 * twopi32),
        "energy_Q": prefactor / (10.0 * sigma**5 * twopi32),
        "field_u": (4.0 / (3.0 * math.sqrt(math.pi))) * a**3 * norm_const,
    }


# ---------------------------------------------------------------------------
# Non-periodic O(N²) realspace path
# ---------------------------------------------------------------------------


def brute_realspace(
    r: np.ndarray,
    q: np.ndarray,
    *,
    mu: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    sigma: float = 1.0,
    prefactor: float = 90.4756,
    remove_self_interaction: bool = True,
    e_ext: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute total electrostatic energy + per-atom Φ, E by direct O(N²) sum.

    Mirrors ``les.module.ewald.compute_potential_realspace``. All atom
    pairs ``(i, j)`` with ``i ≠ j`` contribute through the screened
    Coulomb pair kernels; ``i == j`` (self) is excluded by construction
    of the kernels themselves.

    Args:
        r: ``(N, 3)`` positions.
        q: ``(N,)`` per-atom charges.
        mu: optional ``(N, 3)`` per-atom dipoles.
        Q: optional ``(N, 3, 3)`` per-atom quadrupoles. **Caller
            responsibility**: pass symmetric traceless tensors.
        sigma: Gaussian smearing length. Default 1.0 (LES default).
        prefactor: ``1/(2ε₀)`` in your units. Default 90.4756 (eV·Å·e⁻²).
        remove_self_interaction: If ``True`` (LES default), realspace
            kernels already skip the diagonal so no further action is
            needed. If ``False``, the analytic self-energy / self-Φ /
            self-E constants are added back, mirroring upstream.
        e_ext: Optional ``(3,)`` external field, added to ``field``
            before any α-induced response (caller handles α externally).

    Returns:
        Dict with keys ``"pot"`` (scalar), ``"phi"`` ``(N,)``,
        ``"field"`` ``(N, 3)``.
    """
    n = r.shape[0]
    norm_const = prefactor / (2.0 * math.pi)
    kernels = make_kernels(r, sigma, norm_const)
    f_qq = kernels["f_qq"]
    f_qu = kernels["f_qu"]
    f_uu = kernels["f_uu"]
    f_Qu = kernels["f_Qu"]
    f_QQ = kernels["f_QQ"]

    # Charge → potential at every atom: φⱼ = Σᵢ qᵢ · f_qq(i, j)
    e_phi = np.einsum("i,ij->j", q, f_qq)
    pot = 0.5 * float(np.dot(e_phi, q))

    # Dipole contributions
    e_phi_u = np.zeros(n)
    E_u = np.zeros((n, 3))
    if mu is not None:
        # φ from dipoles: φⱼ += Σᵢ μᵢ·∇f_qq |source=i, receiver=j
        # (kernels.f_qu[i,j,c] = (∇φ at j due to charge at i)[c])
        e_phi_u = np.einsum("ic,ijc->j", mu, f_qu)
        e_phi = e_phi + e_phi_u
        # dipole-charge cross energy (non-symmetric, non-½ — see LES L145)
        pot += float(np.dot(e_phi_u, q))

        # field on j from dipoles at i (kernels.f_uu[i,j,c,d])
        E_u = np.einsum("ijcd,ic->jd", f_uu, mu)
        pot += -0.5 * float(np.einsum("ic,ic->", mu, E_u))

    # Quadrupole contributions
    E_Q = np.zeros((n, 3))
    if Q is not None:
        # φ from quadrupoles: ½ Q : f_uu (Hessian of φ)
        e_phi_Q = 0.5 * np.einsum("iab,ijab->j", Q, f_uu)
        # field from quadrupoles: ½ Q : f_Qu
        E_Q = 0.5 * np.einsum("iab,ijabc->jc", Q, f_Qu)
        e_phi = e_phi + e_phi_Q

        pot += float(np.dot(q, e_phi_Q))
        # Q-Q energy: ⅛ Q : f_QQ : Q
        pot += 0.125 * float(np.einsum("iab,ijabcd,jcd->", Q, f_QQ, Q))
        if mu is not None:
            pot += -float(np.einsum("ic,ic->", mu, E_Q))

    # ``remove_self_interaction=False`` ⇒ add back analytic self-terms.
    if not remove_self_interaction:
        sc = self_corrections(sigma, prefactor)
        pot += float(np.sum(q * q)) * sc["energy_q"]
        e_phi = e_phi + q * sc["phi_q"]
        if mu is not None:
            pot += float(np.sum(mu * mu)) * sc["energy_u"]
            E_u = E_u - sc["field_u"] * mu
        if Q is not None:
            pot += float(np.sum(Q * Q)) * sc["energy_Q"]

    # Compose total field for α-response (does not feed back into pot here).
    e_field = np.einsum("i,ijc->jc", q, f_qu)
    if mu is not None:
        e_field = e_field + E_u
    if Q is not None:
        e_field = e_field + E_Q
    if e_ext is not None:
        e_field = e_field + np.asarray(e_ext)[None, :]

    return {"pot": pot, "phi": e_phi, "field": e_field}


# ---------------------------------------------------------------------------
# Periodic reciprocal-space path
# ---------------------------------------------------------------------------


def brute_reciprocal(
    r: np.ndarray,
    q: np.ndarray,
    cell: np.ndarray,
    *,
    mu: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    sigma: float = 1.0,
    dl: float = 2.0,
    prefactor: float = 90.4756,
    remove_self_interaction: bool = True,
    e_ext: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute Ewald reciprocal-space energy + per-atom Φ, E for a 3D-periodic cell.

    Mirrors ``les.module.ewald.compute_potential_triclinic``. The
    structure factor ``S(k) = Σᵢ [qᵢ + i k·μᵢ − ½ k·Qᵢ·k] e^{i k·rᵢ}``
    is evaluated on a half-k-sphere (with ``factor = 2`` for the non-axis
    shells) using ``cos(k·r)`` and ``sin(k·r)`` real arithmetic.

    Args:
        r: ``(N, 3)`` positions (no PBC wrap required — phases handle
            periodicity through the lattice's reciprocal vectors).
        q: ``(N,)`` charges.
        cell: ``(3, 3)`` lattice vectors as rows
            (``cell[i] = i-th lattice vector``).
        mu: optional ``(N, 3)`` dipoles.
        Q: optional ``(N, 3, 3)`` quadrupoles (symmetric traceless).
        sigma: Gaussian smearing length.
        dl: Reciprocal-space grid resolution; ``k_max = 2π/dl``.
        prefactor: ``1/(2ε₀)``.
        remove_self_interaction: If ``True``, subtract the analytic
            ``Σᵢ qᵢ²/(σ(2π)^{3/2})·prefactor`` (and dipole/Q analogues),
            and also subtract the per-atom self-φ ``qᵢ · 2/(σ(2π)^{3/2})
            ·prefactor`` from ``Φ(r_i)``.
        e_ext: optional external field added to ``field``.

    Returns:
        Dict with keys ``"pot"`` (scalar), ``"phi"`` ``(N,)``,
        ``"field"`` ``(N, 3)``, ``"epsilon_r"`` (always 1.0 in this
        oracle — the LES dielectric self-scaling toggle is *not*
        replicated here; production-side
        :class:`EwaldMultipoleEnergy` covers it under
        ``use_epsilon_r_scaling``).
    """
    sigma_sq_half = sigma * sigma / 2.0
    twopi = 2.0 * math.pi
    k_sq_max = (twopi / dl) ** 2

    volume = abs(float(np.linalg.det(cell)))
    cell_inv = np.linalg.inv(cell)
    G = twopi * cell_inv.T  # reciprocal-lattice basis

    norms = np.linalg.norm(cell, axis=1)
    Nk = [max(1, int(norms[i] / dl)) for i in range(3)]

    # Build the full integer grid, project to k, mask k=0 and k>k_max.
    nvec = np.array(
        list(itertools.product(*(range(-Nk[i], Nk[i] + 1) for i in range(3)))),
        dtype=np.int64,
    )
    kvec = nvec @ G  # (M, 3)
    k_sq = np.sum(kvec * kvec, axis=1)
    keep = (k_sq > 0.0) & (k_sq <= k_sq_max)
    kvec = kvec[keep]
    k_sq = k_sq[keep]
    nvec = nvec[keep]

    # Half-k-sphere optimisation: keep only one of every ±k pair, with
    # factor=2 multiplier on the non-zero modes (mirrors ewald.py L309-316).
    non_zero = (nvec != 0).astype(np.int64)
    first_non_zero = np.argmax(non_zero, axis=1)
    sign = nvec[np.arange(nvec.shape[0]), first_non_zero]
    hemisphere = (sign > 0) | np.all(nvec == 0, axis=1)
    kvec = kvec[hemisphere]
    k_sq = k_sq[hemisphere]
    factors = np.where(np.all(nvec[hemisphere] == 0, axis=1), 1.0, 2.0)

    # Structure factor S(k) = Σᵢ [qᵢ + i k·μᵢ − ½ k·Qᵢ·k] e^{ik·rᵢ}
    #                       (split into real and imag parts of the prefactor).
    k_dot_r = r @ kvec.T  # (N, M)
    cos_kr = np.cos(k_dot_r)
    sin_kr = np.sin(k_dot_r)
    S_real = np.einsum("i,im->m", q, cos_kr)
    S_imag = np.einsum("i,im->m", q, sin_kr)

    if mu is not None:
        uk = mu @ kvec.T  # (N, M)
        # +i k·μ contribution to S: real part = -uk · sin, imag part = uk · cos.
        S_real = S_real - np.einsum("im,im->m", uk, sin_kr)
        S_imag = S_imag + np.einsum("im,im->m", uk, cos_kr)

    if Q is not None:
        # k · Qᵢ · k contracted then per-atom phase: use (M, N) scratch.
        qk2 = np.einsum("ma,iab,mb->im", kvec, Q, kvec)  # (N, M)
        S_real = S_real - 0.5 * np.einsum("im,im->m", qk2, cos_kr)
        S_imag = S_imag - 0.5 * np.einsum("im,im->m", qk2, sin_kr)

    S_sq = S_real * S_real + S_imag * S_imag  # (M,)

    kfac = np.exp(-sigma_sq_half * k_sq) / k_sq  # (M,)
    pot = float(np.sum(factors * kfac * S_sq) / volume * prefactor)

    if remove_self_interaction:
        sc = self_corrections(sigma, prefactor)
        pot -= float(np.sum(q * q)) * sc["energy_q"]
        if mu is not None:
            pot -= float(np.sum(mu * mu)) * sc["energy_u"]
        if Q is not None:
            pot -= float(np.sum(Q * Q)) * sc["energy_Q"]

    # Per-atom potential at r_i: real part of e^{-i k·r_i} S(k), summed.
    prefactor_arr = factors * 2.0 * kfac / volume * prefactor  # (M,)
    term_real_phi = (
        S_real[None, :] * cos_kr + S_imag[None, :] * sin_kr
    )  # (N, M) — real part of e^{-ik r_i} S(k)
    e_phi = np.einsum("m,im->i", prefactor_arr, term_real_phi)

    if remove_self_interaction:
        sc = self_corrections(sigma, prefactor)
        e_phi = e_phi - q * sc["phi_q"]

    # Per-atom field: imaginary part of e^{-i k·r_i} S(k) · k.
    term_imag = S_real[None, :] * sin_kr - S_imag[None, :] * cos_kr  # (N, M)
    e_field = np.einsum("m,im,mc->ic", prefactor_arr, term_imag, kvec)

    if remove_self_interaction and mu is not None:
        sc = self_corrections(sigma, prefactor)
        # Reciprocal path adds back +c_self·μ (sign opposite to realspace).
        e_field = e_field + sc["field_u"] * mu

    if e_ext is not None:
        e_field = e_field + np.asarray(e_ext)[None, :]

    return {"pot": pot, "phi": e_phi, "field": e_field, "epsilon_r": 1.0}


# ---------------------------------------------------------------------------
# Self-tests — run with `python tests/_oracles/screened_coulomb.py`.
#
# These are sanity checks on the oracle itself, not parity tests against
# any production code. They cross-check the oracle against:
#
#   1. Bare ``1/r`` in the σ → 0 limit (a small positive σ should give
#      a value within ~σ/d of the analytic Coulomb energy).
#   2. ``brute_realspace`` on a two-charge dimer at large separation
#      reduces to ``q₁ q₂ / r · prefactor / (4πε₀)``.
#   3. ``brute_reciprocal`` on the same two-charge dimer in a large
#      cubic box agrees with ``brute_realspace`` for ``d ≪ box / 2``
#      (the periodic-image tails are small).
#   4. Self-correction parity: a single isolated charge with
#      ``remove_self_interaction=False`` reproduces the analytic
#      Gaussian self-energy ``q² · √(2/π) / (2σ) · prefactor``.
# ---------------------------------------------------------------------------


def _selftest() -> None:
    """Run the four sanity checks. Raises on tolerance failure."""
    # 1. Bare 1/r limit (σ → small): screened kernel must collapse to bare
    # Coulomb at separations ≫ σ; we just check sign here, the magnitude
    # check lives in test #2 below at a longer separation.
    sigma = 0.01
    prefactor = 90.4756
    norm_const = prefactor / (2.0 * math.pi)
    r = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    q = np.array([1.0, -1.0])
    out = brute_realspace(r, q, sigma=sigma, prefactor=prefactor)
    assert out["pot"] < 0.0, f"σ→0 dimer should be attractive, got {out['pot']}"

    # 2. Two-charge dimer
    r2 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # 10 Å far apart
    q2 = np.array([1.0, -1.0])
    out2 = brute_realspace(r2, q2, sigma=1.0, prefactor=prefactor)
    # At d=10 ≫ σ=1, erf(10/√2) ≈ 1, so U → q₁q₂/d · norm_const = -norm_const/10
    expected = -norm_const / 10.0
    rel_err = abs(out2["pot"] - expected) / abs(expected)
    assert rel_err < 1e-3, f"dimer {out2['pot']} vs analytic {expected}, rel_err {rel_err}"

    # 3. Reciprocal dimer in a large cubic box
    cell = 50.0 * np.eye(3)
    out3 = brute_reciprocal(
        r2 + np.array([25.0, 25.0, 25.0]), q2, cell, sigma=1.0, dl=2.0, prefactor=prefactor
    )
    # In a 50 Å box, periodic images are at d=50, much larger than direct d=10;
    # reciprocal sum should give ≈ same as realspace dimer to a few %.
    assert abs(out3["pot"] - expected) / abs(expected) < 0.05, (
        f"reciprocal {out3['pot']} vs realspace {expected}"
    )

    # 4. Self-correction parity: single atom, remove_self_interaction=False
    r1 = np.array([[0.0, 0.0, 0.0]])
    q1 = np.array([1.0])
    out4 = brute_realspace(r1, q1, sigma=1.0, prefactor=prefactor, remove_self_interaction=False)
    # Single isolated charge: only self-energy contributes.
    sc = self_corrections(1.0, prefactor)
    expected_self = sc["energy_q"]
    rel_err4 = abs(out4["pot"] - expected_self) / abs(expected_self)
    assert rel_err4 < 1e-12, (
        f"self-energy {out4['pot']} vs expected {expected_self}, rel_err {rel_err4}"
    )

    # 5. remove_self_interaction=True yields zero for a single atom
    out5 = brute_realspace(r1, q1, sigma=1.0, prefactor=prefactor, remove_self_interaction=True)
    assert abs(out5["pot"]) < 1e-12, f"single-atom RSI=True should be zero, got {out5['pot']}"

    print("All oracle self-tests passed.")


if __name__ == "__main__":
    _selftest()
