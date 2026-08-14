"""Public-API regression for the Parsley energy-weighted TorsionProfileLoss.

Spec: ``universal-class-i-12-torsion-weights`` (acceptance ac-010).

One minimal scenario, public API only: a single four-point torsion scan is
scored with :class:`~molix.core.losses.TorsionProfileLoss`, first in its
default (unweighted) form and then with the opt-in Parsley per-configuration
policy :class:`~molix.core.losses.ParsleyEnergyWeight`, finally wrapped in
:class:`~molix.core.losses.ScaleNormalizedLoss` to supply the eq.-3 ``1/dE²``
prefactor by composition. Everything runs on CPU in float64. No third-party
oracle is imported, subprocessed, or consulted — here or at capture time:
every golden below is closed-form arithmetic on the literals in this file.

Domain basis — Qiu et al., "Development and Benchmarking of Open Force Field
v1.0.0 — the Parsley Small-Molecule Force Field", *J. Chem. Theory Comput.*
**17**, 6262-6280 (2021), §2.1.7 eqs. 3-5,
https://doi.org/10.1021/acs.jctc.1c00571. For one scan ``g`` (kcal/mol)::

    L_g = sum_{i in g} w(dE_i) . r_i^2  /  sum_{i in g} w(dE_i)
    r_i = (E_MM,i - c_MM) - (E_QM,i - c_QM),   c = weighted group mean
    dE_i = E_QM,i - min_{j in g} E_QM,j

    w(dE) = 1                                    dE < denom
    w(dE) = denom / sqrt(denom^2 + (dE - denom)^2)   denom <= dE < upper
    w(dE) = 0                                    dE >= upper

with ``denom = 1``, ``upper = 5`` kcal/mol. The trailing "-1/2" of the
requirement text is read as the **exponent** (``w = (1 + (dE-1)^2)^(-1/2)``);
the literal "minus one half" reading is discontinuous at ``dE = denom`` and
negative above 2 kcal/mol (spec §Domain basis).

**Fixture P — all three weight branches at once** (``d = E_MM - E_QM``)::

    E_QM  = [0.0, 0.5, 2.0, 7.0]        -> dE = [0.0, 0.5, 2.0, 7.0]
    w     = [1,   1,   1/sqrt(2), 0]       (branch 1, 1, 2, 3)
    E_MM  = [-0.5, 0.0, 2+sqrt(2), 57.0]
    d     = [-0.5, -0.5, sqrt(2), 50.0]

    sum w.d  = -0.5 - 0.5 + (1/sqrt(2)).sqrt(2) + 0 = 0   -> offset c = 0
    sum w.d^2 = 0.25 + 0.25 + (1/sqrt(2)).2 + 0 = 0.5 + sqrt(2)
    sum w     = 2 + 1/sqrt(2) = (4 + sqrt(2))/2
    L = (0.5 + sqrt(2)) / (2 + 1/sqrt(2)) = (1 + 2 sqrt(2))/(4 + sqrt(2))
      = 1/sqrt(2) = **0.7071067811865476**
        [since (4 + sqrt(2))/sqrt(2) = 2 sqrt(2) + 1]

**Fixture Q — zero-weight exclusion** (same scan, uniform +1 offset, and a
50 kcal/mol error parked on the ``dE = 7`` configuration, where ``w = 0``)::

    E_MM = [1.0, 1.5, 3.0, 57.0];  d = [1, 1, 1, 50]
    c = sum w.d / sum w = (2 + 1/sqrt(2)) / (2 + 1/sqrt(2)) = 1
    r = d - c = [0, 0, 0, 49],  w = [1, 1, 1/sqrt(2), 0]
    L = (0 + 0 + 0 + 0.49^2 . 0) / (2 + 1/sqrt(2)) = **0.0**

    Barrier term (unweighted, per-group max - min, so it is untouched by the
    weights and stays additive): pred 57 - 1 = 56, ref 7 - 0 = 7
    barrier_weight=1.0 -> L + (56 - 7)^2 = 0 + **2401.0**

**Default-off equivalence** (the pre-existing unweighted fixture)::

    E_MM = [0, 4], E_QM = [0, 2], one group
    mean-centered: pred [-2, 2], ref [-1, 1]; MSE = (1 + 1)/2 = **1.0**
    TorsionProfileLoss() and TorsionProfileLoss(energy_weight=None) agree
    bit-for-bit -- the default constructor is untouched by this spec.

**Eq.-3 prefactor by composition** (``dE = 1 kcal/mol``, so numerically the
identity; it only fixes the dimensionless bookkeeping)::

    ScaleNormalizedLoss(TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()),
                        scale=1.0)  on fixture P
      = L_P / 1.0^2 = **0.7071067811865476**

    capture: none — every golden above is closed-form arithmetic on the
             literals in this file; nothing was captured from a run, and no
             oracle software of any kind is involved.
    verify:  PYTHONPATH=src python regressions/universal-class-i-12-torsion-weights.py
    commit:  edef6fd (+ working tree implementing universal-class-i-12-torsion-weights)
    torch:   2.12.1+cpu
    device:  cpu, float64 (atol 1e-12 on every scalar)
    date:    2026-08-14
"""

from __future__ import annotations

import sys

import torch

from molix.config import config
from molix.core.losses import ParsleyEnergyWeight, ScaleNormalizedLoss, TorsionProfileLoss

#: One side of the loss call — the ``pred`` / ``target`` mappings it reads.
Scan = dict[str, torch.Tensor]

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring); energies kcal/mol, weights dimensionless
# ---------------------------------------------------------------------------

#: ``sqrt(2)`` and ``1/sqrt(2)`` as exact float64 literals.
_SQRT2 = 1.4142135623730951
_INV_SQRT2 = 0.7071067811865476

#: Fixture P/Q reference (QM) scan energies — dE = [0, 0.5, 2, 7] kcal/mol, so
#: the Parsley weights are [1, 1, 1/sqrt(2), 0]: all three branches.
_E_QM: list[float] = [0.0, 0.5, 2.0, 7.0]
#: Fixture P predicted (MM) energies — residual d = [-0.5, -0.5, sqrt(2), 50].
_E_MM_P: list[float] = [-0.5, 0.0, 2.0 + _SQRT2, 57.0]
#: Fixture Q predicted (MM) energies — uniform +1 offset, 50 kcal/mol error on
#: the zero-weight configuration.
_E_MM_Q: list[float] = [1.0, 1.5, 3.0, 57.0]
#: Scan (group) membership: one torsion profile.
_GROUP: list[int] = [0, 0, 0, 0]

#: Pre-existing unweighted fixture: predicted / reference energies and grouping.
_E_MM_DEFAULT: list[float] = [0.0, 4.0]
_E_QM_DEFAULT: list[float] = [0.0, 2.0]
_GROUP_DEFAULT: list[int] = [0, 0]

#: Parsley eq.-3 denominator ``dE``, kcal/mol — composed via ScaleNormalizedLoss.
_ENERGY_DENOM = 1.0
#: Barrier weight for the fixture-Q barrier check (0.0 elsewhere, the default).
_BARRIER_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Hard-coded goldens (closed-form, derived in the module docstring)
# ---------------------------------------------------------------------------

#: Fixture P weighted profile loss = ``1/sqrt(2)`` (hard-coded golden).
_GOLDEN_P = 0.7071067811865476
#: Fixture Q weighted profile loss = 0 exactly — the +1 offset is absorbed by
#: weighted centering and the 50 kcal/mol error sits on a ``w = 0`` point
#: (hard-coded golden).
_GOLDEN_Q = 0.0
#: Fixture Q with ``barrier_weight=1.0`` = ``(56 - 7)**2`` — the barrier term
#: stays unweighted and additive (hard-coded golden).
_GOLDEN_Q_BARRIER = 2401.0
#: Default-off profile loss, both implicit and explicit ``energy_weight=None``
#: (hard-coded golden; dyadic-exact, asserted without tolerance).
_GOLDEN_DEFAULT_OFF = 1.0
#: Fixture P through ``ScaleNormalizedLoss(..., scale=1.0)`` (hard-coded golden).
_GOLDEN_P_SCALED = 0.7071067811865476
#: Absolute tolerance for every float64 scalar that is not dyadic-exact.
_ATOL = 1e-12


def _scan(e_mm: list[float], e_qm: list[float], group: list[int]) -> tuple[Scan, Scan]:
    """Build the ``(pred, target)`` mappings for one torsion scan.

    Args:
        e_mm: Predicted (MM) scan energies, kcal/mol.
        e_qm: Reference (QM) scan energies, kcal/mol.
        group: Per-configuration scan id.

    Returns:
        The ``pred`` / ``target`` mappings in the key convention the loss reads
        (``"energy"``, ``"energy"`` / ``"molecule_id_index"``).
    """
    pred = {"energy": torch.tensor(e_mm, dtype=torch.float64)}
    target = {
        "energy": torch.tensor(e_qm, dtype=torch.float64),
        "molecule_id_index": torch.tensor(group, dtype=torch.int64),
    }
    return pred, target


def _check(name: str, got: torch.Tensor, golden: float, *, atol: float = _ATOL) -> None:
    """Assertion helper: *got* is a float64 scalar tensor equal to *golden*."""
    if not isinstance(got, torch.Tensor) or got.ndim != 0:
        raise AssertionError(f"{name}: expected a scalar tensor, got {got!r}")
    if got.dtype != torch.float64:
        raise AssertionError(f"{name}: expected float64, got {got.dtype}")
    delta = abs(float(got) - golden)
    if delta > atol:
        raise AssertionError(f"{name} drift: {float(got)!r} vs {golden!r} (|d| = {delta})")


def main() -> int:
    config.set_precision("fp64")

    # 1. Fixture P — all three weight branches; L = 1/sqrt(2).
    pred_p, target_p = _scan(_E_MM_P, _E_QM, _GROUP)
    weighted = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())
    _check("fixture P weighted profile", weighted(pred_p, target_p), _GOLDEN_P)

    # 2. Fixture Q — zero-weight exclusion (L = 0), and the unweighted,
    #    additive barrier term on top of it ((56 - 7)^2 = 2401).
    pred_q, target_q = _scan(_E_MM_Q, _E_QM, _GROUP)
    _check("fixture Q weighted profile", weighted(pred_q, target_q), _GOLDEN_Q)
    with_barrier = TorsionProfileLoss(
        energy_weight=ParsleyEnergyWeight(),
        barrier_weight=_BARRIER_WEIGHT,
    )
    _check("fixture Q + barrier", with_barrier(pred_q, target_q), _GOLDEN_Q_BARRIER)

    # 3. Default-off equivalence — the plain mean-centered MSE is untouched,
    #    and the explicit ``energy_weight=None`` matches it bit-for-bit.
    pred_d, target_d = _scan(_E_MM_DEFAULT, _E_QM_DEFAULT, _GROUP_DEFAULT)
    implicit = TorsionProfileLoss()(pred_d, target_d)
    explicit = TorsionProfileLoss(energy_weight=None)(pred_d, target_d)
    _check("default-off (implicit)", implicit, _GOLDEN_DEFAULT_OFF, atol=0.0)
    _check("default-off (explicit None)", explicit, _GOLDEN_DEFAULT_OFF, atol=0.0)
    if float(implicit) != float(explicit):
        raise AssertionError(
            f"default-off drift: energy_weight=None changed the plain path "
            f"({float(explicit)!r} vs {float(implicit)!r})"
        )

    # 4. Eq.-3 ``1/dE^2`` prefactor by composition (dE = 1 kcal/mol).
    scaled = ScaleNormalizedLoss(
        TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()),
        scale=_ENERGY_DENOM,
    )
    _check("fixture P through ScaleNormalizedLoss", scaled(pred_p, target_p), _GOLDEN_P_SCALED)

    print(
        "universal-class-i-12-torsion-weights: Parsley-weighted torsion profile (cpu, fp64) -> "
        "fixture P 1/sqrt(2)=0.7071067811865476, fixture Q 0.0 (zero-weight point excluded), "
        "fixture Q + unweighted barrier 2401.0, default-off 1.0 bit-for-bit for both "
        "TorsionProfileLoss() and energy_weight=None, ScaleNormalizedLoss(scale=1 kcal/mol) "
        "on fixture P 0.7071067811865476 OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
