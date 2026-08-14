---
title: Parsley per-configuration energy weights for TorsionProfileLoss
status: approved
created: 2026-08-14
chain: universal-class-i
grilled: true
---

# Parsley per-configuration energy weights for TorsionProfileLoss

## Summary

`TorsionProfileLoss` currently scores a torsion scan as a plain mean-centered MSE over
every configuration, so a 24-point scan contributes 24 residuals of equal importance and
a 6-point scan contributes 6 — high-energy, chemically irrelevant grid points count as
much as the barrier region, and long scans dominate short ones. This spec adds the
Parsley per-configuration energy weighting and the profile-level `Σw` normalization as an
opt-in constructor extension: a small `ParsleyEnergyWeight` policy object maps each
configuration's relative in-scan QM energy to a weight (1 below 1 kcal/mol, decaying to
0 at 5 kcal/mol), and the loss reduces each scan to a `Σw`-normalized weighted MSE so one
scan contributes `O(1)` regardless of its grid count. The default constructor is
unchanged and byte-for-byte equivalent to today's behavior, so the sibling
`universal-class-i-11-step` routing contract (`(pred, target)` mapping, key names, forward
signature) is untouched.

## Domain basis

Parsley torsion-profile objective — Qiu et al., "Development and Benchmarking of Open
Force Field v1.0.0 — the Parsley Small-Molecule Force Field", *J. Chem. Theory Comput.*
**17**, 6262–6280 (2021), §2.1.7, eqs. 3–5. https://doi.org/10.1021/acs.jctc.1c00571

For one torsion scan (group) `g`:

```
L_g = (1/dE²) · Σ_{i∈g} w(ΔE_i) · [E_QM,i − E_MM,i]²  /  Σ_{i∈g} w(ΔE_i)

ΔE_i = E_QM,i − min_{j∈g} E_QM,j            (relative in-scan QM energy, kcal/mol)
dE   = 1.0 kcal/mol

w(ΔE) = 1                                    ΔE < dE
w(ΔE) = dE / sqrt(dE² + (ΔE − dE)²)          dE ≤ ΔE < ΔE_max
w(ΔE) = 0                                    ΔE ≥ ΔE_max = 5 kcal/mol
```

At `dE = 1 kcal/mol` the middle branch is `w(ΔE) = (1 + (ΔE − 1)²)^(−1/2)`.

**Interpretation decision (flag for operator veto at review).** The requirement text wrote
the middle branch as `1/(1+(E−1)²)−½`. This spec reads the trailing `−½` as the
**exponent** (equivalently `1/sqrt(1+(E−1)²)`), not as "minus one half". Rationale: the
literal "minus one half" reading is discontinuous at `ΔE = 1` (jumps `1 → 0.5`) and goes
**negative** for `ΔE > 2` — a negative weight rewards increasing the error on those
configurations, which cannot be the published objective. The exponent reading is
continuous at `ΔE = dE` (`w = 1`), positive, monotonically decreasing, and matches the
`energy_denom` / `energy_upper` energy attenuation used by the OpenFF/ForceBalance
torsion-profile targets (Wang et al., *JCTC* **10**, 4556 (2014),
https://doi.org/10.1021/ct500643c) that produced the Parsley numbers. ForceBalance's
overall `1/dE` factor on `w` cancels under the `Σw` normalization, so weights are
normalized here to `w(0) = 1`. ForceBalance is an **offline** interpretation reference
only — it is never imported, subprocessed, or run by this repo's tests.

**Residual offset.** `[E_QM,i − E_MM,i]` is taken after removing the per-scan constant
offset, because a scan's absolute energy zero is arbitrary. Under a weighted objective the
minimizing offset is the **weighted** group mean of `d_i = E_MM,i − E_QM,i`
(`argmin_c Σ w (d − c)² = Σwd/Σw`), which reduces exactly to the plain group mean when
all weights are equal. Zero-weight configurations therefore drop out of the numerator,
the denominator, **and** the offset — which is what "excluded" has to mean for the
weighting to be meaningful.

**`Σw ≥ 1` is structural.** The scan minimum always has `ΔE = 0`, hence `w = 1`, so the
denominator can never be zero for a non-empty group; no clamp or epsilon is needed.

**Prior-art support for profile-level objectives.** Sage (OpenFF 2.0.0) retained the
torsion-profile weighting while dropping vibrational-frequency targets — Boothroyd et al.,
*JCTC* **19**, 3251–3275 (2023), https://doi.org/10.1021/acs.jctc.3c00039 (PMC10269353).

**Units.** All energies kcal/mol; weights dimensionless. The weighted loss is in
`(kcal/mol)²`. The eq.-3 `1/dE²` prefactor is *not* re-implemented here — it is composed
by the caller with the existing `ScaleNormalizedLoss(..., scale=dE)`; with `dE = 1
kcal/mol` it is numerically the identity and only fixes the dimensionless bookkeeping.

## Design

**`ParsleyEnergyWeight(nn.Module)`** — new, in `src/molix/core/losses/molecular.py`.
Keyword-only `denom: float = 1.0`, `upper: float = 5.0` (kcal/mol), both stored via
`register_buffer(..., dtype=config.ftype)` so they ride `.to()` / `state_dict` and are
never learnable (`WeightedLoss` precedent, `combined.py:33-37`; dtype per the
`init-dtype-at-construction` note). `__init__` raises `ValueError` for `denom <= 0` or
`upper < denom` (validation style of `ProjectedInternalHessianLoss.__init__`).
`forward(relative_energies: Tensor) -> Tensor` is one primitive: it detaches its input,
raises `ValueError` on any negative entry (guards against a caller passing absolute
instead of relative energies, which would silently weight everything at 1), evaluates the
three-branch `w`, and returns weights in the input dtype with `requires_grad=False` —
weights are constants of the objective, never a gradient path.

**`center_by_group`** — generalized with a keyword-only `weights: Tensor | None = None`.
`None` keeps the existing plain-mean path unchanged; a supplied `weights` makes each
group subtract its weight-normalized mean. Same shape contract, same length validation,
same per-group loop.

**`TorsionProfileLoss`** — constructor gains one keyword-only argument
`energy_weight: nn.Module | None = None` (a submodule, so device/dtype and `state_dict`
follow the parent; not a bool mode flag, so the policy's `denom` / `upper` travel with the
object and are unit-testable in isolation). `None` — the default — leaves `forward`'s
existing code path literally untouched. When supplied, `forward`:

1. computes `ΔE` from the **detached** reference energies via a new private
   `_relative_to_group_min(values, group_ids)` (mirrors the existing private
   `_barrier_heights` per-group-loop style; one call site, so it stays a private method),
2. gets `w = self.energy_weight(ΔE)`,
3. weighted-centers pred and ref with `center_by_group(..., weights=w)`,
4. returns the per-group `Σw·r² / Σw` vector reduced by the existing `reduction` string —
   `"mean"` → mean over **profiles**, `"sum"` → sum over profiles, `"none"` → `(G,)`.
   This per-profile reduction is the deliberate difference from the unweighted path
   (which reduces over configurations) and is what delivers grid-count balancing.

The barrier term is unchanged and stays **unweighted** — it is a per-group scalar already,
so `Σw` normalization does not apply; total is `weighted_profile + barrier_weight ·
barrier_mse`, exactly as today.

No new public constructor style, no `make_*`, no `method=` switch, no change to
`forward(pred, target)` or to any key name — the 11-step router keeps calling
`TorsionProfileLoss(...)` and passing the same mappings.

**Reuse decision** (no `librarian_report` was supplied with this task; candidates below
were resolved by direct reading of `src/molix/core/losses/`):

- `center_by_group` (`molecular.py:50`) — **generalize**: add keyword-only `weights=`
  instead of writing a second weighted-centering routine. Spawns the "Generalize" task.
- `ScaleNormalizedLoss` (`molecular.py:416`) — **reuse**: supplies the Parsley `1/dE²`
  prefactor by composition; no duplicate scale knob is added to `TorsionProfileLoss`.
- `WeightedLoss` buffer pattern (`combined.py:33-37`) — **reuse (pattern)**: non-learnable
  scalars via `register_buffer` + declared attribute annotation.
- `TorsionProfileLoss._barrier_heights` (`molecular.py:369`) — **reuse (pattern)**:
  per-group unique-loop shape for the new `_relative_to_group_min`.
- `ProjectedInternalHessianLoss` (`molecular.py:239`) — **reuse (pattern)**: keyword-only
  constructor + `ValueError` validation + Google docstring with a `Reference:` block.
- `nn.MSELoss` reduction vocabulary — **reuse**: same `"mean"/"sum"/"none"` strings, applied
  to the per-group vector.
- `ParsleyEnergyWeight` — **new**: no energy-weighting or attenuation policy exists
  anywhere under `src/molix/core/losses/` (the only `boltz*` hits in `src/` are the MD
  Boltzmann constant), and inlining the branch table inside `forward` would make the
  three-branch physics untestable as its own unit.

## Files to create or modify

- `src/molix/core/losses/molecular.py` — add `ParsleyEnergyWeight`, generalize
  `center_by_group` with `weights=`, extend `TorsionProfileLoss.__init__` / `forward` and
  add `_relative_to_group_min`, insert `"ParsleyEnergyWeight"` into the alphabetized
  `__all__`.
- `src/molix/core/losses/__init__.py` — re-export `ParsleyEnergyWeight` and keep `__all__`
  alphabetized.
- `tests/test_molix/test_core/test_losses_molecular.py` — add `TestParsleyEnergyWeight`;
  extend `TestCenterByGroup` and `TestTorsionProfileLoss` without editing any existing case.
- `regressions/universal-class-i-12-torsion-weights.py` (new) — public-API regression with
  hard-coded goldens.

## Tasks

- [ ] Write failing unit tests for `ParsleyEnergyWeight` branch table and guards (tests/test_molix/test_core/test_losses_molecular.py → `TestParsleyEnergyWeight`)
- [ ] Implement `ParsleyEnergyWeight` in `src/molix/core/losses/molecular.py` (buffers at `config.ftype`, detached output, `ValueError` guards)
- [ ] Write failing unit tests for weighted centering and the weighted profile term (extend `TestCenterByGroup` and `TestTorsionProfileLoss` in the same file)
- [ ] Generalize `center_by_group` in `src/molix/core/losses/molecular.py` to accept keyword-only `weights` (default `None` = current plain-mean behavior)
- [ ] Implement the `energy_weight` path in `TorsionProfileLoss` (`_relative_to_group_min`, weighted centering, `Σw`-normalized per-profile term, reduction over groups)
- [ ] Export `ParsleyEnergyWeight` from `src/molix/core/losses/__init__.py` with alphabetized `__all__` in both modules
- [ ] Add docstrings per google style with kcal/mol units, the Parsley/Sage `Reference:` block, and the `−½`-as-exponent interpretation note
- [ ] Add regression example `regressions/universal-class-i-12-torsion-weights.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify default-off equivalence against the pre-existing `TestTorsionProfileLoss` / `TestCenterByGroup` cases with zero edits to those lines
- [ ] Run full check + test suite

## Testing strategy

Unit tests live in the existing `tests/test_molix/test_core/test_losses_molecular.py`
(the in-tree mirror file for `src/molix/core/losses/molecular.py`; do **not** fork a second
`test_losses/test_molecular.py`). Every case targets a single function/method. New fixtures
use `dtype=torch.float64` with `atol=1e-12` for the analytic goldens; pre-existing cases and
their `_ENERGY_ATOL = 1e-10` stay untouched.

Shared fixtures (kcal/mol, one scan unless stated; `d = e_pred − e_true`):

| fixture | `e_true` | `e_pred` | weights `w` |
|---|---|---|---|
| P | `[0.0, 0.5, 2.0, 7.0]` | `[-0.5, 0.0, 2.0+sqrt(2), 57.0]` | `[1, 1, 1/sqrt(2), 0]` |
| Q | `[0.0, 0.5, 2.0, 7.0]` | `[1.0, 1.5, 3.0, 57.0]` | `[1, 1, 1/sqrt(2), 0]` |
| E | `[0.0, 0.5]` | `[0.0, 2.5]` | `[1, 1]` |

`TestParsleyEnergyWeight` (happy path, edges, domain validation — all hard-coded):

- branch table at `denom=1, upper=5`: `ΔE = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]` →
  `w = [1.0, 1.0, 1.0, 0.7071067811865476, 0.4472135954999579, 0.0, 0.0]`.
- continuity at the lower boundary: `w(1.0 - 1e-9) ≈ w(1.0) = 1.0` within `1e-6`
  (fails under the rejected "minus one half" reading).
- non-negativity and monotone non-increase on `arange(0, 8, 0.01)`: all `w >= 0`,
  all `diff(w) <= 0` (second guard on the rejected reading).
- non-default policy `denom=2.0, upper=6.0`: `ΔE = [1.0, 2.0, 4.0, 6.0, 10.0]` →
  `w = [1.0, 1.0, 0.7071067811865475, 0.0, 0.0]`.
- `ValueError` for `denom=0.0`, for `upper=0.5` with `denom=1.0`, and for a negative
  `ΔE` input.
- detachment: input with `requires_grad=True` → returned `w.requires_grad is False`.

`TestCenterByGroup` (additions only):

- weighted centering with `weights=[1., 1., 1/sqrt(2), 0.]` on fixture P's `d` returns `d`
  itself (weighted mean is 0) to `atol=1e-12`.
- `weights=None` reproduces the existing plain-mean golden `[-1.0, 1.0, -2.0, 2.0]`
  bit-for-bit (regression guard on the generalization).

`TestTorsionProfileLoss` (additions only):

- fixture P, `energy_weight=ParsleyEnergyWeight()` → `0.7071067811865476` (= `1/sqrt(2)`;
  covers all three weight branches at once).
- fixture Q → `0.0` exactly: the uniform `+1` offset is absorbed by weighted centering and
  the 50 kcal/mol error sits on a `w = 0` configuration (zero-weight exclusion).
- fixture Q with `barrier_weight=1.0` → `2401.0` (`(56 − 7)²`; barrier term stays
  unweighted and additive).
- gradient exclusion: fixture P with `e_pred.requires_grad_()` → after `backward()`,
  `e_pred.grad[3] == 0.0` exactly and the other three entries are finite and non-zero.
- profile-level invariance: group 0 = fixture P, group 1 = fixture P with every grid point
  duplicated (8 points) → both per-profile losses equal `0.7071067811865476`;
  `reduction="sum"` → `1.4142135623730951`; `reduction="none"` → shape `(2,)`.
- equal-weight consistency (fixture E): weighted path and default path both return `1.0`
  and agree exactly — with uniform weights the objective degenerates to the plain
  mean-centered MSE.
- singleton group with weights on (`e_true=[3.0]`, `e_pred=[9.0]`) → `0.0`.
- default-off equivalence: on the pre-existing fixture (`pred [0., 4.]`, `true [0., 2.]`,
  group `[0, 0]`) `TorsionProfileLoss()` and `TorsionProfileLoss(energy_weight=None)` both
  return `1.0` and are equal bit-for-bit.

Regression example: `regressions/universal-class-i-12-torsion-weights.py` — public API only
(`from molix.core.losses import ParsleyEnergyWeight, ScaleNormalizedLoss,
TorsionProfileLoss`), no pytest, no third-party import, exits 0 on success and 1 on drift.
It reproduces the hard-coded goldens `0.7071067811865476` (fixture P), `0.0` and `2401.0`
(fixture Q), `1.0` (default-off), and demonstrates the eq.-3 prefactor by composition:
`ScaleNormalizedLoss(TorsionProfileLoss(energy_weight=ParsleyEnergyWeight()), scale=1.0)`
on fixture P → `0.7071067811865476`. Header comment records the capture command
(`PYTHONPATH=src python regressions/universal-class-i-12-torsion-weights.py`), the commit
sha, the torch version, the date, and device/precision (CPU, float64), per
`regressions/README.md`. All goldens are analytic (closed-form `1/sqrt(2)`, `0`, `2401`,
`1`) — no oracle software of any kind is involved.

## Out of scope

- Boltzmann / temperature-based conformer weighting, and any other weighting policy beyond
  the Parsley branch table (alternatives considered: a generic `weight_fn` callable slot and
  a Boltzmann policy sibling — both rejected as speculative with zero in-tree callers today;
  the `energy_weight` submodule slot leaves room for one later without a second signature
  change).
- Changing the sibling `universal-class-i-11-step` routing contract, target/pred key names,
  or the `forward(pred, target)` signature.
- Weighting the barrier term, or making `barrier_weight` energy-dependent.
- Turning the weights on by default, or touching any training script / config default
  (considered and rejected: flipping the default would silently change every in-flight
  torsion objective; opt-in only until a calibration experiment says otherwise).
- Vibrational-frequency / Hessian targets (Sage dropped them; `ProjectedInternalHessianLoss`
  is unchanged here).
- Dataset-side production of torsion-scan grouping or QM relative energies.
- Any training submission — this is library work only; the training gate in CLAUDE.md is
  untouched.
