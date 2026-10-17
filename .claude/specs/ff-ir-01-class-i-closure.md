# ff-ir-01-class-i-closure

## Summary

One intermediate representation (IR) for Class-I force fields, owned by molrs,
into which OPLS-AA, AMBER/GAFF/GAFF2, CHARMM/CGenFF and the Rondo learned field
all **embed exactly**, and out of which each converts back exactly when — and
only when — the field lies in that family's image. Everything else is a
*projection*, declared as such, with a residual.

Why now: a learned Class-I fit has near-degenerate directions (redundant valence
angles, torsion vs 1-4, LJ vs bonded) that need a convention (gauge). With an IR,
a convention is data — "the reference parameters of field X, term by term" — and
switching convention is switching reference field, not changing the model. It is
also what makes a learned field exportable to any engine without silent loss.

Builds on `ff-params-01-canonical-vocabulary` (names, units, the general Fourier
torsion as canonical). This spec adds the **closure**, the **canonical form**, and
the **exact-or-refuse / project-with-residual** contract.

## Domain basis (mathematics)

- A Class-I field is `E(x; p) = Σ_terms f_style(q_term(x); p_term)`, `q` an
  internal coordinate or a pair distance.
- Two parameter sets are **equivalent** iff `E_A(x) = E_B(x) + c` for all `x`.
- The IR is a family `F_IR` with, for every vendor form `v`, a linear/affine
  embedding `ι_v: P_v → P_IR` such that `E_v(x; p) = E_IR(x; ι_v(p))`.
- **A → B is exact ⟺ `ι_A(p_A) ∈ image(ι_B)`.** Otherwise only a projection
  exists: nearest in a declared metric (second-order Taylor at a reference
  geometry, or energy/force-weighted least squares over a configuration set).
- **Canonical form**: remove every notational freedom so that equivalence ⟺
  equality of IR values: units; the factor convention (the IR adopts the
  LAMMPS standard: un-halved `K`, no hidden ½); `Rmin/2 → σ`
  (`σ = 2^(5/6)·Rmin/2`); AMBER `IDIVF` folded; repeated torsion rows on one
  quadruple summed into one series; torsion constant term dropped; improper
  atom order fixed per style; Coulomb constant stated (AmberTools 332.0522173
  vs CODATA 332.06371 — the InterMol SAMPL5 study traced 98.8 % of a
  Coulomb-14 mismatch to it).

## Term-by-term verdicts

| Term | Vendor forms | IR closure | Verdict | Basis |
|---|---|---|---|---|
| bond | harmonic everywhere (½ factor varies) | `K(r−r0)²` (LAMMPS `bond harmonic`) | exact | — |
| angle | harmonic θ; CHARMM adds Urey–Bradley `K_UB(r13−s0)²` | `K(θ−θ0)²`; UB as LAMMPS `angle charmm` `K(θ−θ0)² + K_ub(r13−r_ub)²`, a 1-3 term that creates no exclusion | exact without UB; **UB has no angle-only form** (it couples bond lengths and angle: r–θ cross blocks in the Hessian) | GROMACS angle funct 5 / OpenMM AmoebaUreyBradley carry it exactly |
| proper torsion | OPLS F1..F4; RB C0..C5; AMBER/GAFF multi-term periodic; CHARMM any-phase periodic | Fourier `Σ aₙ cos nφ + bₙ sin nφ` (Rondo's Cartesian form) | every embedding exact; IR→RB needs `bₙ = 0, n ≤ 5`; IR→OPLS needs `n ≤ 4`, phases ∈ {0, π}, ΣC = 0 | GROMACS manual eq. 200–201; ParmEd `from_rbtorsion`; GMSO refuses C5 ≠ 0 |
| improper | AMBER/GAFF/OPLS periodic `k[1+cos(2ψ−π)]` (centre 3rd); CHARMM harmonic `k(ψ−ψ0)²` (centre 1st); SMIRNOFF centre 2nd, three permutations averaged | both families, each in the atom order whose dihedral it prices (`improper periodic` AMBER's, centre third; `improper harmonic` centre first) | periodic ↔ harmonic only to **second order**: `K = n²k/2` in LAMMPS's un-halved K (`2k` at n = 2; the `k_h = n²k` sometimes quoted is the ½-form); quartic terms differ by −k n⁴ δ⁴/24 | derivation; Espaloma notes AMBER improper order is not unique |
| CMAP | CHARMM only, bicubic φ/ψ grid | optional term | **no sum-of-1D-torsions form** (not separable) | — |
| LJ 12-6 | σ or Rmin/2; Lorentz–Berthelot vs geometric; NBFIX | **per-pair `(ε_ij, σ_ij)` table** | exact at pair level; across mixing rules with per-atom parameters **impossible**: self pairs force `σ'_i = σ_i`, cross pairs then need `√(σᵢσⱼ) = (σᵢ+σⱼ)/2 ⇒ σᵢ = σⱼ` | proof as stated; ParmEd/Interchange need NBFIX/pair tables |
| 1-4 | AMBER 1/1.2 & 0.5; OPLS 0.5/0.5; CHARMM 1.0 + dedicated 1-4 LJ | per-pair exception table | exact with the exception table; converting to a target's global scaling is **not exact**: `ΔE₁₄(φ)` through `r14(φ) = √(a + b cos φ)` is an infinite Fourier series and depends on other coordinates — least-squares projection onto torsions only | derivation; Abdullah et al. 2025 (arXiv 2504.14398) |
| charge | fixed point charges | same | exact | — |
| Class-II / MMFF / GROMOS | cubic/quartic stretch, stretch-bend, buffered 14-7; GROMOS quartic bond, cosine angle | outside the Class-I IR | second-order match or refit only (GROMOS `k = 2k_b b0²`, `k = k_θ sin²θ0`) | GROMACS manual |

Field-level settings the IR carries (not parameters, but they change energies):
cutoffs and switching, long-range method, Coulomb constant, unit system, mixing
rule (when a pair table is not materialised), exclusion policy.

## Degeneracy and convention

- **Exact degeneracies** are removed by the canonical form.
- **Near-degeneracies** survive it — different functions that agree on the data's
  reach: the uniform `θ0` shift at a symmetric centre (redundant internal
  coordinates; first-order data cannot see it, `θ0` enters only the Hessian
  through `k(θ̄ − θ0)·∇²Σθᵢ`), torsion vs 1-4, LJ vs bonded. They do not affect
  conversion (an equality question) but make fitting ill-posed. Literature: the
  force-constant matrix is not unique with redundant coordinates (Groner &
  Günthard 1976); restrained least squares (Vanommeslaeghe et al. 2015,
  `lsfitpar`); ByteFF L1 on redundant DOF; Espaloma's two-reference linear
  mixture `K = K1 + K2`, `b = (K1 b1 + K2 b2)/K`.
- Measured in Rondo (GH200 3296327 / 3296603, 2026-10-02): with stiffness held,
  every carbon centre's `θ0` drifted to ~93° while the loss changed by ~0.2 %
  over a 16.6° uniform shift, and both loss halves pushed back up — a flat valley.

## Design

1. **Owner**: molrs. molnex's `PotentialIR` (`molpot/ir/*`) becomes a view of it
   or is retired; two IRs is the failure mode this spec exists to end.
2. **Pair table**: `pair` styles hold explicit cross rows; kernels price a pair by
   its cross row when one exists, by the mixing rule otherwise (landed for
   `lj/cut`, see below). Readers keep cross rows (NBFIX, ACOEF/BCOEF,
   `nonbond_params`) instead of refusing or dropping them.
3. **1-4 exceptions**: a per-pair exception table (scale or explicit `ε, σ, q q`),
   replacing the single global `SpecialBonds` when a field needs it (CHARMM).
4. **New categories**: Urey–Bradley (1-3 harmonic, no exclusion), CMAP (grid +
   bicubic kernel).
5. **Improper atom order** (as landed): no centre-first order has the dihedral
   an AMBER improper prices (its axis runs through the centre), so
   `improper periodic` is stored in AMBER's order (centre third, the order a
   LAMMPS data file lists for `cvff` to reproduce AMBER) and `improper
   harmonic` / `cvff` / the out-of-plane styles centre first; readers map
   OpenMM's `(c1..c4)` to `(c2, c3, c1, c4)` and writers invert it (molrs
   forcefield-ir guide, "Improper atom order").
6. **Operations** (methods on the owning types, one job each — no all-in-one
   converter):
   - `ForceField.canonical()` — canonical form;
   - per form, an exact `to_<form>()` that **refuses** when not in the image
     (e.g. a Fourier series with `bₙ ≠ 0` has no RB form);
   - a separate projection primitive taking a declared metric and returning
     parameters **and** the residual.
7. **Equivalence tests**: InterMol-style per-term energies, same molecules ×
   random configurations, across LAMMPS / OpenMM XML / GROMACS / prmtop.

## Landed (P1 — silent miscalculations on the conversion path, 2026-10-02)

Each fix has a regression test that failed before and passes after.

1. **OpenMM `PeriodicTorsionForce` read as zeros** (`molrs .../readers/opls.rs`).
   The reader parsed every row as CL&P `c0..c3`, so OpenMM's own
   `k{m}/periodicity{m}/phase{m}` became all-zero torsions, and `<Improper>`
   rows were skipped. Now decided per row: OpenMM spelling → `dihedral periodic`
   / `improper periodic`; CL&P → `dihedral opls`; neither or both → error. The
   XML **writer** had the same tag for its periodic torsions, so write → read lost
   every periodic torsion; it also wrote a non-OpenMM `<PeriodicImproperForce>`
   and harmonic impropers as periodicity-0 rows. Now impropers go out as OpenMM
   `<Improper>` rows (cvff mapped through its sign), and non-periodic improper or
   termless dihedral styles are refused.
2. **NBFIX ignored** (`molrs .../pair/lj_cut.rs`, `.../readers/lammps.rs`). Cross
   `pair_coeff i j` lines were dropped at read, and cross rows that did reach the
   kernel were ignored in favour of mixing. Both constructors now prefer a cross
   row (`LJCut::with_type_pair` for the type-pair table); a later `pair_coeff`
   replaces an earlier one as in LAMMPS; a wildcard cross line is an error.
3. **Harmonic improper energy doubled on the LAMMPS path**
   (`molrs .../improper/harmonic.rs`, GROMACS reader/writer). The kernel applied
   `k(χ−χ0)²` while the LAMMPS reader stored `k = 2K` for a `½k` form. Fixed
   in P1 by a `½k` kernel; **superseded** in 0.16, when the IR adopted the
   LAMMPS standard: the kernel is LAMMPS's un-halved `K(χ−χ0)²` with `K` stored
   as LAMMPS writes it, and GROMACS funct 2 maps `K = k_ξ/2`. A test prices a
   LAMMPS-read improper against the LAMMPS manual's `K(χ−χ0)²`.
4. **Export of impropers** (`molnex src/molix/ff_export/`). The LAMMPS adapter
   recorded any periodic improper as cvff; it now maps exactly (one non-zero
   term, phase 0/π → `d = ±1`) or raises `UnsupportedTermError`. The LAMMPS deck
   writer silently omitted improper energy (F25 note); `write_settings` now
   refuses a field with non-zero improper force constants on a system with
   impropers. molrs's LAMMPS writer gained the canonical `dihedral periodic`
   (written as `fourier`).

## Phases

| Phase | Scope | Acceptance | Status |
|---|---|---|---|
| P2 | Design items 2–5: pair tables in every reader/writer, 1-4 exception table, UB, CMAP, improper canonical order | each new term: energy = hand/manual value; readers stop refusing NBFIX/UB/cmaptypes where the IR can hold them | **landed** (waves 2–3, 2026-10-05/06): `lj/charmm` + `coul/charmm` with `one_four`, per-pair override columns, `angle charmm` (UB), `cmap charmm`, cross rows in every reader and writer, the improper order of item 5; LAMMPS, OpenMM XML, GROMACS and prmtop readers (prmtop read-only) |
| P3 | Design item 6: `canonical()`, exact `to_<form>()` with refusal, projection with residual | random parameters × random configurations: `E_vendor = E_IR` to 1e-12; out-of-image conversion raises; projection residual reported and monotone in the metric | **partly landed**: the torsion algebra (`forcefield::torsion` — every torsion form ↔ Fourier series, exact or refused naming the term; canonical series; every registered torsion kernel held to its series on random geometries). `canonical()` on the field, `to_form` / `fit_form` and the projection move to ff-ir-02 (WP9, form codecs on the protocol) |
| P4 | Design item 7: cross-engine equivalence table | per-term relative energy error < 1e-6 on exact paths (LAMMPS ↔ OpenMM XML ↔ GROMACS ↔ prmtop) | **landed** (2026-10-06, molrs `ir/p4-equiv`): five sources (ff14SB, GAFF2 and CHARMM36 prmtops; CHARMM36 OpenMM XML with NBFIX; OPLS-AA GROMACS `.top`) × LAMMPS, OpenMM XML, GROMACS and the native engine (sander / OpenMM / GROMACS) × three seeded configurations, energies per term and forces: every exact path ≤ 1e-11, engine-side differences — GROMACS 1-4 tables ≈1e-9, sander's 8-digit ACOEF ≈1e-9, OpenMM's periodic-spline CMAP slopes ≤ 1e-7 kcal/mol, LAMMPS's improper-harmonic force clamp at sin χ < 0.001 — all < 1e-6; every Coulomb at its engine's own constant. Pinned in `ff::equivalence_check` (`testdata/equivalence/engines.tsv`), rerun by `scripts/ff_equivalence_check.sh`. With it the IR completeness matrix (`ff::completeness`, generated into the forcefield-ir guide): every registered kernel and field setting × every reader/writer and molrec v2, ✓ backed by named tests or refused by name; gaps fixed (GROMACS system writer; GROMACS/OpenMM writers' Coulomb-constant and units handling; OpenMM writer class and same-label conflicts; LAMMPS `lj/cut` without Coulomb, `coul/long` → `coul/long/pme`, `pair_modify shift`; GROMACS reader's own Coulomb constant and intermolecular pairs) |
| P5 | Rondo: per-term reference columns from a chosen field (e.g. GAFF2 via molrs `GaffTypifier`, not yet exposed to Python); optionally frozen reference LJ (closes Rondo risk R1) | unblocked (operator, 2026-10-02: earlier rulings — "no GAFF compatibility" 2026-09-21, departure D2 — no longer bind) | in progress (GAFF typifier Python bindings, 2026-10-06) |

## Out of scope

Polarisable models (Drude, AMOEBA multipoles), Class-II as an IR member, virtual
sites (recorded as a gap, not designed here), CHARMM file I/O beyond what P2 needs.

## Open

- ~~`molpot.forcefield` cannot be imported against any current molrs build~~ —
  resolved 2026-10-02: molrs core / ff pyclasses are subclassable again (the
  "one class per concept" stance of molrs 9d0ea7d is dropped, operator), molnex's
  container is ported to the 0.15 binder API (`def_style`, handle `def_type`,
  `special_bonds` getter), and molnex runs in its own environment built from local
  molrs/molpy (`scripts/env/build_env.sh`), which carries the P1 fixes. Fast
  tier 3320 passed, molrs Python 858, molrs Rust 3060 + 105.
