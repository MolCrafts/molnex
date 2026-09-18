# ff-params-01-canonical-vocabulary

## Summary

One parameter vocabulary for classical force fields across molrs and molnex.
Every file format — AMBER prmtop, GROMACS top, OPLS XML, LAMMPS, moltemplate —
is normalised **at read** into that vocabulary; the format's own spelling never
reaches an internal structure. The canonical torsion is the **general periodic
(Fourier) series**, not the OPLS 4-term special case, and the Class-I fit
predicts that general form.

Today the two sides do not merely disagree on names — they mis-compute:

| Defect | Effect |
|---|---|
| OPLS torsion coefficients are `k1..k4` (molnex), `f1..f4` (molrs kernel), `c1..c4` (molnex→molpy `realize()`) | a molnex field evaluated in molrs gets `0.0` for every torsion — `dihedral_opls_ctor` defaults a missing key (`molrs/src/ff/potential/dihedral/opls.rs:101-104`) |
| LJ mixing is a declared option in molnex, hard-coded Lorentz–Berthelot in molrs (`pair/lj_cut.rs:385-386`) | an OPLS-AA (geometric-σ) model silently changes σ_ij across the boundary |
| Buckingham is `A`/`rho`/`C` in molrs, `a`/`rho`/`c` in molpy (`builder.py:442`) | a molpy-built `buck` type cannot construct its molrs kernel |

Eleven ad-hoc alias/fallback layers already exist to paper over this (§ Inventory);
none of them is a schema, and none is shared.

## Domain basis

A Class-I (diagonal) force field is fixed by its functional forms, not by any
vendor's spelling:

- **bond** `E = ½ k (r − r₀)²`
- **angle** `E = ½ k (θ − θ₀)²`
- **proper / improper torsion** `E = Σ_m k_m [1 + cos(n_m φ − γ_m)]`
- **LJ 12-6** `E = 4ε[(σ/r)¹² − (σ/r)⁶]`, with a declared mixing rule
- **Coulomb** `E = k_e q_i q_j / r`, with 1-2 / 1-3 / 1-4 scales

Everything a format adds on top is a *file convention*, not a form:

- AMBER's `IDIVF` divides a barrier that the file lists once per path. It is
  exactly `k ← k / idivf`; folding it at read loses no physics.
- OPLS's `k1..k4` **is** the series above at `n = 1,2,3,4` with
  `γ = 0, π, 0, π` and `k_m = K_m / 2`.
- GROMACS's Ryckaert–Bellemans `c0..c5` is the same series re-expanded.
- The Cartesian pair `(a_m, b_m)` is the same term at
  `k = √(a²+b²)`, `γ = atan2(b, a)`.

So the general series is the only representation that can hold all of them
without a lossy branch, which is why it is the canonical one.

## Design

### 1. Canonical vocabulary

LAMMPS **style names** are adopted verbatim (they are already the registry
vocabulary on both sides). LAMMPS **parameter letters** are *not*: they are
positional in `*_coeff` lines, and the letters collide across styles — `n` is a
torsion multiplicity and a Tang–Toennies order, `d` is a torsion phase and a
Morse well depth, `k2..k4` are OPLS-adjacent on one side and class2 anharmonic
coefficients on the other. The canonical names are therefore descriptive and
unambiguous, and the letters live only in the reader/writer tables.

| Term | Canonical parameters | Unit | Notes |
|---|---|---|---|
| `bond:harmonic` | `k`, `r0` | kcal/mol/Å², Å | `E = ½k(r−r0)²`; LAMMPS `K = k/2` (already agreed on both sides) |
| `angle:harmonic` | `k`, `theta0` | kcal/mol/rad², rad | radians internally, degrees only at a file boundary |
| `dihedral:periodic` | `k`, `periodicity`, `phase` | kcal/mol, 1, rad | **the canonical torsion**; one triple per term |
| `improper:periodic` | `k`, `periodicity`, `phase` | kcal/mol, 1, rad | centre atom is `atomi` |
| `improper:harmonic` | `k`, `chi0` | kcal/mol/rad², rad | |
| `pair:lj/cut` | `epsilon`, `sigma` | kcal/mol, Å | plus the declared `mixing` |
| `pair:coulomb` | `charge` | e | see § 4 |

| `improper:cvff` | `k`, `periodicity`, `sign` | kcal/mol, 1, ±1 | `E = k[1 + s·cos(nχ)]`; `sign` is a sign, never a phase |
| `pair:buck` | `a`, `rho`, `c` | kcal/mol, Å, kcal/mol·Å⁶ | GROMACS' `b = 1/rho` is inverted at read |
| `pair:coul/tt` | `b`, `order`, `c` | 1/Å, 1, 1 | `order` is the Tang–Toennies damping order |

Deleted from the internal vocabulary: `idivf`, `f1..f4`, `c1..c4`,
`a`/`b` (Cartesian), `n`, `d`, `phi0`, `k0`, `q`, Buckingham `A`/`C`. Each
becomes a reader mapping. `k1..k4` survives only as the OPLS quartet.

### 2. Readers normalise; nothing else does

Every reader converts into the canonical form **before** anything is
constructed. No downstream code branches on the source format.

| Source | Reads | Canonical result |
|---|---|---|
| AMBER prmtop | `PK`, `PN`, `PHASE`, `IDIVF` | `k = PK/IDIVF`, `periodicity = |PN|`, `phase = PHASE` |
| OPLS XML | `k1..k4` | `k_m = K_m/2`, `periodicity = (1,2,3,4)`, `phase = (0,π,0,π)` |
| GROMACS RB | `c0..c5` | expand to `k_m`, `periodicity = 1..5`, `phase ∈ {0,π}` |
| GROMACS ub | `theta0,k,r0,k_ub` | an `angle:harmonic` plus a `bond:harmonic` on the 1-3 pair |
| LAMMPS `opls` | `K1..K4` | as OPLS XML |
| LAMMPS `fourier`/`charmm` | `m, K_i, n_i, d_i` | direct |
| moltemplate | positional per `_PARAM_NAMES` | direct |

Writers invert this. `LammpsAdapter._is_opls_manifold` already detects the OPLS
special case from the general series and emits `dihedral_style opls`; that stays
and becomes the model for every other format's writer.

### 3. Storage: general per-term, dense-with-mask at runtime

The canonical form is ragged — torsion types differ in how many terms they carry
and in which multiplicities. Storage keeps that generality; the **runtime**
packs it to a fixed `(n_types, n_terms_capacity)` array with a boolean term mask
and `k = 0` on dead terms, which is the same fixed-capacity-plus-mask rule the
batch padder follows and keeps the energy compile-clean.

This replaces molnex's current dense `k[n_types, n_terms]` with a *shared*
`periodicity[n_terms]` — that shared axis is the OPLS assumption in disguise
(every type using `n = 1,2,3,4`), and it cannot express AMBER or CHARMM torsions.
`periodicity` and `phase` become per-type-per-term, shaped like `k`.

### 4. `charge`, not `q`

molrs, OPLS XML, AMBER and moltemplate all say `charge`; only molnex says `q`,
and molnex already writes both to bridge it (`charge/assignment.py:86-89`).
Canonical is `charge`; the dual write and the `q` spelling go. `NonbondedScaling`
kinds become `("charge", "lj")` — LAMMPS's `special_bonds coul` keyword is a
writer detail, as `coul/long` already is.

### 5. Class-I fit predicts the general form

`ProperTorsionParamHead(form="opls")` predicting `k1..k4` becomes a general head
predicting `(k, phase)` per term over a declared periodicity set, with the
Cartesian `(a, b)` parameterisation as the trainable interior (it is smooth
through `k = 0`, where `(k, phase)` is not). OPLS becomes what it is — a
constrained initialisation and an export detector, not the model's shape.
`Parameterizer._OPLS_PROPER_KEYS` and the `form=` switch go.

### 6. Mixing rule crosses the boundary

`pair:lj/cut` gains `mixing` in molrs — a **string** style param, since `Params`
is otherwise `f64`. Values are `arithmetic` (Lorentz–Berthelot; `lorentz` and
`lorentz-berthelot` are accepted spellings), `geometric` and `sixthpower`. The
kernel defaults to `arithmetic`, which is what a reader that declares nothing
(AMBER prmtop, GAFF) actually means. Every reader that *does* know states it:
OpenMM / foyer XML from `<ForceField combining_rule>`, LAMMPS from
`pair_modify mix`, defaulting to `geometric` there because that is LAMMPS' own
default for `lj/cut`. The LAMMPS writer emits `pair_modify mix` back.

## Files

molnex: `src/molpot/forcefield/schema.py` (registry), `ir/bags.py`,
`ir/scaling.py`, `forcefield/potentials.py` (`_opls_to_cosine`,
`_called_proper_tables`, the identity-table tuples), `forcefield/container.py`
(`_propers_to_ir`, `_proper_params`, `_atom_params`), `forcefield/parameterizer.py`,
`composition/mm_heads.py`, `molix/ff_export/{lammps,openmm}_adapter.py`,
`conventions.py`.

molrs: `src/ff/potential/dihedral/{opls,periodic,charmm}.rs`,
`improper/{cvff,periodic}.rs`, `pair/{lj_cut,buck,morse}.rs`,
`forcefield/readers/{lammps,gromacs,opls,prmtop}.rs`, `writers/{lammps,gromacs}.rs`.

molpy: `parser/moltemplate/{builder,lt_writer}.py`.

## Out of scope

- class2 / MMFF / UFF styles (molrs-only today; they inherit the same rule when
  molnex grows them).
- Buckingham and Morse case unification — real, but separate from the Class-I
  path; tracked as follow-on.
- The `propers`/`impropers` (molnex) vs `dihedrals`/`impropers` (molrs)
  **namespace** names: topology columns, not parameters.

## Decisions (operator, 2026-09-15)

1. **Granularity.** LAMMPS **style names** verbatim; **parameter names**
   descriptive and unambiguous. The letters stay in the reader/writer tables.
2. **`IDIVF`.** Folded into `k` at read (`k ← PK / IDIVF`); the canonical form
   has no divisor. A prmtop is therefore not byte-reproducible — the energies
   are, the divisor split is not. Accepted.

## Landed (2026-09-15)

molnex, full suite green (3449) and the slow tier (57):

- `q` → `charge` everywhere; the `{"q": …, "charge": …}` dual write is gone.
  `NonbondedScaling.charge`, `ChargeBag.charge`, kinds `("charge", "lj")`.
- `idivf` deleted from the vocabulary. Energy is `Σ k_m [1 + cos(n_m φ − γ_m)]`
  with `k` already divided; every `k / idivf` site disappeared. Found and fixed
  on the way: `_opls_to_cosine` did not emit `periodicity`, so a caller handing
  in this batch's OPLS table for a new topology gathered the **seed's**
  multiplicities — now the dialect yields the whole canonical triple and it is
  guarded like `k` and `phase`.
- `periodicity` is per type, per term (`[n_types, n_terms]`), shaped like `k`
  and `phase`, so AMBER and CHARMM torsions fit. A shared `[n_terms]` row is
  accepted and broadcast — it is the general table's special case, not a
  different contract. The Chebyshev basis gathers per interaction at no cost
  (verified equal to the shared path on the same input).
- `ClassIForceField`: the general Class-I container (`dihedral:periodic`).
  `espaloma_like_template()` returns it instead of `OPLSAAForceField`, so a
  learned field is no longer born inside one vendor's functional form.
  `OPLSAAForceField` stays — it *is* OPLS-AA.
- `ProperTorsionParamHead` has one output form: Cartesian `(a, b)` over a
  declared periodicity set. Cartesian because a barrier passing through zero is
  ordinary during a fit and `(k, γ)` has a coordinate singularity there.
  `form=`, `_OPLS_PROPER_KEYS` and the template/head dialect check are gone.

molrs (`cargo check` clean; tests queued):

- OPLS coefficients `f1..f4` → `k1..k4`; molnex's molpy bridge and molpy's
  moltemplate table stop writing `c1..c4`. This closed the silent zero: molnex
  wrote `c1..c4`, the kernel read `f1..f4` and defaulted each miss to `0.0`, so
  every torsion of a realized field evaluated to nothing.
- The kernel now refuses a type carrying **none** of `k1..k4` (that is
  unparameterised, not sparse), and refuses one carrying `n1`/`d1` (a
  `periodic` term table on the wrong style, which would price a plain barrier
  as a half barrier). The writer's `f→c` fallback is gone.
- molpy moltemplate: `periodic` / `charmm` take `k` / `periodicity` / `phase`;
  `phase` joins `_DEGREE_PARAMS` so the boundary still converts degrees.

### molrs, round 2 (2026-09-15)

The four remaining items are done. Evidence, all on a compute node and all
after a `maturin develop --release` rebuild of the extension:

| suite | result |
|---|---|
| molrs `cargo test -p molcrafts-molrs` | 1926 + 15 + 73 passed, 0 failed |
| molrs `cargo fmt --check` | clean |
| molnex fast tier | 3449 passed, 15 skipped |
| molnex slow tier | 57 passed, 3 skipped — both torsion compile gates hold |
| molpy | 1685 passed, 10 failed — all 10 pre-existing, itemised below |

- **Torsion keys.** `periodic` (indexed and single-triple), `charmm` and both
  impropers now read `periodicity` / `phase`. CVFF reads `periodicity` / `sign`:
  its `d` is ±1 in `E = k[1 + d·cos(nχ)]`, a sign and not an angle, so it gets
  its own canonical name rather than being folded into `phase`. Every producer
  moved with them — GAFF tables, prmtop, the LAMMPS reader and writer, the
  OpenMM XML writer, and the typifier's estimate cascade and candidate set.
  The OPLS kernel's wrong-style guard now tests `periodicity1` / `phase1`.
- **Found on the way:** LAMMPS `dihedral_style harmonic` is `E = K[1 + d·cos(nφ)]`
  — its `d` is also a ±1 sign, and the reader was running it through
  `to_radians()`, turning ±1 into ±0.01745. It round-tripped (the writer undid
  it) and no kernel is registered for that style, so it never mispriced an
  energy; it now reads and writes `sign` verbatim.
- **`mixing`** on `pair:lj/cut`, per § 6, with `Params::get_str` added to carry
  it. The OPLS XML reader was dropping `combining_rule` on the floor, so every
  OPLS-AA pack loaded with Lorentz–Berthelot σ — the silent shift this spec
  named. `pair_modify` was likewise an ignored keyword in the LAMMPS reader.
- **Buckingham** is `a` / `rho` / `c` on both sides now, matching molpy.
- **GROMACS normalisation.** The reader converted nothing for dihedrals or
  pairs: phases stayed in degrees, barriers in kJ/mol, and buckingham was stored
  as `A`/`B`/`C` with `B = 1/rho` — keys the kernel does not read, so a
  GROMACS-loaded buckingham could not be built at all. Reader and writer now
  convert both ways (degrees↔radians, kJ↔kcal, nm↔Å).
- **Tang–Toennies** `n` → `order`, closing the `n` collision the design names.
- **molpy followed**: the moltemplate `cvff` row is `k` / `sign` / `periodicity`,
  the LAMMPS `coul/tt` formatter reads `order`, and the style tests state the
  canonical names instead of the LAMMPS letters. The moltemplate **writer** was
  ordering `*_coeff` values by a hand-kept preferred list that named `d` and
  `n`; with those gone a CHARMM torsion would have written `K w n d`. It is now
  driven by the reader's own `_PARAM_NAMES` table, read backwards — one table,
  both directions — and it converts reference angles back to degrees, which the
  read path did and the write path did not.
- **molnex `_proper_params`**: a periodic table emitted the first amplitude
  alone as `k`, so a realized `ClassIForceField` had no multiplicity for the
  molrs kernel to read and every term after the first was dropped. It now emits
  the full `k{m}` / `periodicity{m}` / `phase{m}` set through `_propers_to_ir`,
  the same converter the IR uses, so Cartesian `(a, b)` and a shared
  periodicity row work identically on both paths.

### `k0` retired, and the factor of two it was hiding (2026-09-15)

`k0` is gone. Every harmonic force constant is `k`, in molrs's own
`E = ½k(x − x₀)²` convention, everywhere.

It was never a pure rename, and the reason is the point. `k0` named **two
different quantities**:

- in the OPLS table, the OPLS XML reader and the emitted `ForceField`, it was
  molrs's `k` — the halved convention the kernels evaluate;
- in the GAFF candidate rows and in the empirical Badger/Wang formulas, it was
  AMBER's un-halved `K`, twice as large.

The two met at one place, `gaff()`, which multiplied by two on its way into the
`ForceField` and so came out right. **Any other consumer of the same tables got
a constant half the strength it should be**, silently: a bond too soft, an
energy that is wrong but finite and plausible. The live instance is the OPLS
typifier's estimator seam — `OPLSAATypifier::with_default_estimator()` builds a
`Parmchk2Estimator` whose `with_context` constructor **always** installs the GAFF
empirical tables, and `assign::write_params` copies estimated params straight
onto a molecule's bonded terms with no conversion. A matched OPLS bond and an
estimated neighbour on the same molecule therefore differed by 2×, under one key.

The fix normalises at the producer, which is what the spec asks of every other
reader — and what every *other* reader already did. LAMMPS doubles
(`lammps_k_to_molrs_half_k`), AMBER prmtop doubles inline, GROMACS and OpenMM
XML do not because those formats already write `½k`. GAFF was the one table
that shipped its own convention downstream. `bond_params` / `angle_params` and `empirical_bond` / `empirical_angle`
all apply the ×2 themselves and emit `k`; `gaff()` no longer converts. One
convention leaves the tables, so every consumer is right by construction rather
than by being the one that remembered.

Also done: the harmonic kernels read `k` only; the OPLS XML reader stops
dual-writing `k` and `k0`; `OplsBondRow` / `OplsAngleRow` name the field
`force_constant` (the angle row's `k` was already the third atom's class — the
exact collision this spec exists to avoid), matching `ParmBondRow` /
`ParmAngleRow`; and molrs's own generic XML reader maps a legacy `k0="…"`
attribute onto `k` at the boundary, so files it wrote before this still load.

`gaff.rs` had **no test module at all**, so the factor of two — the most
dangerous number in the file — was entirely unpinned. It now has three tests:
the bond and angle boundaries double, and a table hit and an estimate agree.

The **emitted `ForceField` is numerically unchanged**: the ×2 moved from the
end of the pipeline to the start, so `gaff()` produces exactly what it did. What
changed is the estimator seam, which now hands out the same convention as the
tables instead of half of it. Nothing failed when the seam was wrong because
nothing tested it — that is why the two boundary tests and the agreement test
exist now.

| suite | after the `k0` work |
|---|---|
| molrs `cargo test` | 1932 + 15 + 73 passed, 0 failed |
| molrs `cargo fmt --check` | clean |
| molnex fast tier | 3449 passed, 15 skipped |
| molnex slow tier | 57 passed, 3 skipped |
| molpy | 1685 passed, 10 failed — the same 10, unchanged |

## The Cartesian torsion dialect, landed 2026-09-16

Moving `ProperTorsionParamHead` to unconstrained Cartesian `(a, b)` was the
right call for the fit — `(k, γ)` has a coordinate singularity at `k = 0`,
which a barrier passing through zero hits routinely — but the dialect was only
half-plumbed. A Class-I training smoke (`quasi_prod.smoke`, GH200) failed three
times in a row, each failure hiding the next:

| # | Failure | Root cause | Fix |
|---|---|---|---|
| 1 | `propers params cannot mix cartesian (a, b) with polar (k, phase)` | `Potentials.term_energies` chose its backfill names as OPLS-or-polar, so a Cartesian table was topped up with the seed term's `k`/`phase` and the merged dict spoke both dialects | third branch: `("a", "b", "periodicity")` |
| 2 | `Could not guard on data-dependent expression` | `_cosine_fields` converted through `ProperTorsionBag.from_cartesian`, whose `__post_init__` validation does `bool((periodicity > 0).all())` — a host sync inside the compiled region | `ProperTorsionBag.polar_from_cartesian`: tensor math only, no bag, no validation; `from_cartesian` now calls it too |
| 3 | `Smoke warmup produced a non-finite loss` | every **padded** torsion row is exactly `(a, b) = (0, 0)` because the mask zeroes parameters rather than energies, and both `hypot` and `atan2` have `0/0` gradients there | evaluate on substituted inputs and select after: dead rows get `k = 0` with zero gradient |

Defect 3 is the one worth remembering. The forward was *completely finite* —
every term, the energy, the forces, the loss (41.3 + 0.39). Only the parameter
gradients were NaN, 54 of them, and they were NaN for rows that contribute no
energy at all. A capacity-first static pad guarantees those rows exist in every
batch, so this was not a rare input: it was every step of every run. Evidence:
`.scratch/molnex_tests/probe_class_i_nan.py` (GH200 jobs 2528458 / 2528566),
`grad nonfinite_params=54` → `0`.

Smoke after the three fixes (job 2528677, `runs/smoke-20260916T091826Z`):
resume 1 step → `last.pt` → resume → step 2 `ok`, 9/9 steps finite, energy term
38→45, force term 1.2→3.2, `grad_norm` 177→209 against `clip_max_norm=1e3`.
The August baseline on the same split read energy term 1.36e6 and `grad_norm`
1.3e8 — that gap is the residualized Huber objective finally reaching this
path, not a change in the data.

Tests added at the layer that owns each fact: `test_potentials.py`
`test_cartesian_proper_tables_price_a_torsion_without_the_seed_phase` (dialect),
`test_cartesian_proper_tables_trace_as_one_graph` (slow, compile seam),
`test_bags.py` `test_a_zero_row_keeps_the_gradient_finite` (the singular point).

## Production-coordinate startup, 2026-09-16

With the torsion defects fixed, the same smoke on **official 4m** coordinates
(`prod_day.smoke_official`, medium tier, pack 131072) hit three more walls,
none of them force-field vocabulary and all of them fixed here:

| # | Wall | Measurement | Fix |
|---|---|---|---|
| 4 | startup 27 min before step 1 | `index_s=1817`, then `1632` and `1634` on reruns with ε[Z] already cached (Slurm 2528101 / 2528891 / 2530538) | `get_distrib("graph.natoms")` decodes all 3.99 M rows into `Frame`s to read one scalar each. The vector is now cached on disk beside ε[Z] — the same cache this module had before it moved onto the distrib API, whose own comment read *"the 8-hour Frame scan was the bug"*. `index_s` 1657 s → **0.3 s** |
| 5 | `no frozen Class-I extent … run the scan first` | the frozen 4m scan **was on disk** (`f1933ed9b118f5a1.npz`, 34 MB, scanned 09-15) | it stores the cutoff column as `edges`; `MoleculeExtent.FIELDS` says `pairs` since the 09-15 rename, so `load()` raised `KeyError('pairs')` — which `try_for_dataset` swallowed into "never scanned". The reader now accepts the legacy spelling, and `try_for_dataset` answers `None` **only** for a missing file |
| 6 | host-RAM OOM kill in warmup | `ReqMem 11672M`, killed at `MaxRSS 9.5 G`; the passing run peaked at **39.8 G** | not a code defect: the HBM sweep sized GPU memory (medium = 14.1 GiB) and nothing ever sized host memory at the 131072 operating point. A production submission needs an explicit `--mem` — 8 cores' default share is 3.4× short |

Wall 5 is the expensive kind of silence: an error handler that treats
"unreadable" as "absent" turned a stale column name into an instruction to
redo a multi-hour scan. The rename swept ~2700 call sites and could not sweep
a `.npz` written before it.

Evidence, warm caches, GH200-class node with `--mem=160G` (Slurm 2533603):
`official_load_s=10.8`, padder `atoms 131072 / pairs 9.9 M / graphs 2988`,
16 steps, train loss 258.7, val MAE 124.7 kcal/mol, 0.46 steps/s — on an L40,
so the rate is not a GH200 number. No NaN, no OOM.

## The calibration gates, re-run on current code (2026-09-16)

Operator ruling recorded first, because it changes what the gate *is*: a
**molexp record and a committed tree are not gates**. The checklist and
`CLAUDE.md` now say so. What stays binding is the evidence that the
configuration works — data, parity, learnability, batching, a bounded smoke —
because that is what goes stale when code changes.

Re-running align / overfit / profile against current code found seven more
defects, none of them force-field vocabulary. Every one was invisible while the
gates were pointed at a configuration production never runs.

| # | Defect | Why it was invisible |
|---|---|---|
| 7 | `ImproperPeriodic`/`ProperTorsionPeriodic` raised on an **empty** table (`max()` over 0 elements) | a learned field declares styles with no rows and the head fills them per batch, so the container is assembled by whatever batch comes first — and a small molecule has no improper |
| 8 | With the depth then guessed as 1, the next batch carrying `n = 2` gathered past the Chebyshev ladder — a CUDA index assert nowhere near the cause | the depth is fixed at construction to keep the hot path sync-free; a 0-row table states no depth, so it now takes the table's **width** |
| 9 | `align.py` cloned a `state_dict` with `{k: v.detach()}` | Class-I state dicts carry nested force-field tables. The trainer learned this on 2026-09-13; align kept its own copy of the idiom. `snapshot_state_dict` is now public and shared |
| 10 | The same loop then subtracted **bool** buffers | an Adam step moves float parameters; masks and multiplicities are structure |
| 11 | `align._valence_rows` called `.get` on a molpy `Frame` | the annotation said `dict`; the samples have been prepared `Frame`s all along |
| 12 | The overfit gate padded **8 molecules into a 131072-atom batch** — 99.99 % ghosts, 12 s/step, so its 4000 steps could not finish in any wall clock | it inherited the production pack. Capacity is a batching knob; the gate now sizes it to its own tiny set |
| 13 | The net-charge check's fixed `1e-6 e` could no longer tell fp32 noise from a real violation | it was sized on ≤32-atom molecules and fired 33 times on one production-shaped batch, every one arithmetic. Fixing it took three passes and the last one is the interesting one: `4·u·Σ|q|` left 17, `4·u·√n·Σ|q|` left 1 — and that last warning was on a *small* molecule at 21·u·Σ|q|, which is not a summation error at all. It is the fp32 **EEM solve** returning its own Lagrange constraint, and that residual follows the system's conditioning, not `Σ|q|` of the answer. The tolerance is now a precision-scaled floor (1e-5 e at float32) or the size-scaled term, whichever is larger. Both scale with the working dtype, so float64 still reports a 5e-6 e miss, and 0.01 e is always reported |

**The profile now profiles production.** It used to run ≤32-atom molecules at a
512-atom pack against a **zero** ε[Z] — an old spec even recorded the zero table
as known debt. It now takes the official `4m` role with no size filter, the
tier's own capacity, and a fitted ε[Z] (`atoms 2…348`, 83 live elements).

**The bottleneck is the input path, measured.** GH200, medium tier, 131072-atom
pack:

| | per batch |
|---|---|
| compiled train step | **154 ms** (the 09-15 HBM sweep's number, reproduced) |
| eager step | 2771 ms — `torch.compile` is worth **18×** here |
| loader at 4 workers | **~2000 ms** |

The accelerator idles ~93 % of the time. Per-sample prepare is 2.1 ms, of which
**57 % is molpy Frame column writes** and 31 % is `schema.column` alone — 53
column writes per molecule, because `label_sample` wrote the valence blocks into
the frame and `_with_meta` then rebuilt the frame and copied every column again.
Each column is now written once. Worker scaling is near linear
(432 / 709 / 1439 molecules per second at 0 / 2 / 4), so production now sizes
workers from the Slurm allocation instead of hard-coding 4.

With workers sized from the allocation the loader reaches **5907 molecules per
second (500 ms per batch)** at 15 workers — the accelerator goes from ~7 % busy
to ~31 %. Still open: the prepare is recomputed every epoch for every molecule
although bonds and the valence enumeration are pure topology. Precomputing them
into the store is the structural fix and belongs to MolHub, not here.

**Gate results on current code.** `align` passes (`ok=1.0`, all three size
buckets). `overfit` passes (energy MSE 28.4 → 4.6e-4, force 13.9 → 7.6) — and
re-reading its own evidence found a fourteenth defect: the force-scale medians
were taken over the **padded** tensor, so `ref_med` read exactly 0.0, and
`_force_ok`'s `if ref_med <= 0: return pred_med <= 0` branch reported the scale
check as satisfied while checking nothing. The medians now run on live rows and
a missing reference scale is a failure, not a pass.

## Optimizing the compiled step, 2026-09-16

**First, a methodological correction.** `profile_class_i.torch_profile` calls
`model(batch)` directly — the **eager** path. Its hotspot table said 77 % of
device time went to `index_put_` and `indexing_backward_kernel`, and acting on
that removed a real no-op (`table[arange(n)]`, whose backward is an atomic
scatter) but moved the production number **not at all**: 198 ms → 202 ms, noise.
Inductor had already elided it. The eager step is 2.7 s against a compiled 88 ms,
so that table describes a computation production never performs. The profile now
also profiles the **compiled** callable `ClassIFit.run` trains, and that table
is the one to read.

**The real hotspot was the charge solve.** On the compiled step it was
`linalg_lu_factor_ex` at **22 %** — three times the next operator. The EEM
system is solved as a dense bordered matrix, one `(n_max+1)²` block per
molecule: at the operating point a batched LU over `513×513` blocks for
molecules averaging **55 atoms**, so ~99 % of every block is identity-pinned
padding.

It did not need factorizing at all. Class-I uses `DiagonalHardnessKernel`,
`A = diag(η)`, so row *i* reads `η_i q_i + λ = −χ_i` and the constraint fixes
`λ = −(Q + Σ χ_i/η_i) / Σ 1/η_i` — closed form, O(n). The kernel now solves its
own system (the structure is the kernel's knowledge); kernels that couple atoms
keep the dense path, and the over-capacity fail-closed applies to both.

| | before | after |
|---|---|---|
| compiled step (large) | 201.8 ms | **88.6 / 87.4 ms** (two runs) |
| speedup over eager | 14.4× | **31×** |
| graph count / breaks | 1 / 0 | 1 / 0 |

Verified as algebra, not a heuristic: the closed form matches `linalg.solve` to
**1e-10** on random χ/η over four molecules of different sizes with non-zero
totals, conserves the declared charge exactly, stays finite on empty molecules,
and still carries gradient to χ and η. The remaining table is a healthy ML
profile — `mm`/`addmm`/gemm ≈ 50 % (the encoder's own arithmetic),
`scatter_add` ≈ 11 % (graph aggregation).

**Two measurement traps found while doing this**, both of which would have
misled the next person:

- `loader_workers()` capped at 16, so requesting 64 cores changed nothing. The
  ceiling is now 32 and documented as a host-RAM trade (15 workers already peak
  at 39.8 GiB).
- The throughput probe buffered `W × prefetch` batches into a split that held
  fewer, so the timed window drained a queue rather than measuring production —
  it reported **202,356 molecules/s**, two orders above anything the hardware
  can do. The split is now sized past the queue.

## Where the time goes now (large tier, GH200)

| | before | after |
|---|---|---|
| compiled step | 201.8 ms | **~88 ms** |
| input, 32 workers | — | **232 ms** (12,926 molecules/s) |
| effective step `max(compute, input)` | 535 ms | **232 ms** |
| GPU busy | ~7 % | **~38 %** |

Loader scaling is near linear and unsaturated at 32 workers
(436 / 737 / 1490 / 12,926 molecules per second at 0 / 2 / 4 / 32), all measured
past the prefetch queue. Extrapolating, matching the 88 ms compute would need
~85 workers — beyond the 72 cores a single GPU's share of the node carries. So
**adding workers is finished as a lever**: the remaining input cost is the
per-sample prepare, and the structural fix is precomputing bonds and the valence
enumeration into the store, which is MolHub's to make.

## The input path: where its bytes and its time go (2026-09-16)

A packed Class-I batch is **92.5 % neighbour list**:

| | per batch | share |
|---|---|---|
| `pairs` (index + diff + dist) | 302 MB | **92.5 %** |
| valence (bonds/angles/propers/impropers) | 19.5 MB | 6.0 % |
| atoms (pos/Z/forces/mask/batch) | 5.1 MB | 1.6 % |

With 32 workers prefetching 4 batches each that is **40.9 GiB in flight**, against
3.1 GiB if the list were not carried — which is why the job asked for 320 GB of
host memory. The request was sized by extrapolating a 15-worker RSS rather than
measuring, and an over-large `--mem` costs queue time; size it from a measured
`MaxRSS`.

**The list is built in the worker, on CPU.** `_CollateFn.__call__` runs
prepare → collate → **batch tasks (NeighborList)** → pad, all inside the
DataLoader worker, and the 302 MB result then crosses shared memory to the main
process before the H2D copy. `CLAUDE.md` says the opposite: stores carry no
neighbour graph because "the cutoff graph is a function of positions and a
cutoff, so it is built downstream by `molix.data.NeighborList`" — which the MD
path does and this one does not.

**And it was built one molecule at a time.** `NeighborList.execute` looped
`for graph in range(n_graphs)`, taking a boolean mask and launching the C++
kernel per molecule: 2388 Python iterations and 2388 launches for one packed
batch — precisely the "avoidable per-molecule serial execution" the checklist
forbids on the input path, and the reason moving the task to the GPU *as it
stood* would have made things worse rather than better.

`execute` now has a whole-batch path: every molecule's candidate pairs are
decoded from one flat index (`searchsorted` for the molecule, the upper-triangle
closed form for `i < j`) and filtered in a handful of tensor ops, so the work
follows the pair count rather than the molecule count, on whatever device the
batch is on. It is used for open boundaries with `filter_padding=True` from 32
graphs up; PBC (needs a per-molecule box) and `filter_padding=False` (exists to
expose the kernel's own NaN rows) keep the kernel loop, so all 16 pinned
behaviours are untouched.

Writing the equivalence test caught a real defect: the kernel's half-pair
convention is `source > target`, and the new decode produced `i < j`. With
`symmetry=True` both directions are emitted and the mistake is invisible; only
the `symmetry=False` comparison exposed it.

Still the dominant input cost, and not addressed here: the **per-sample
prepare**, 2.1 ms × 2388 molecules ÷ 32 workers ≈ 157 ms of the measured
232 ms per batch. Its fix is precomputing bonds and the valence enumeration
into the store, which is MolHub's to make.

## What the profiler learned to report (2026-09-16)

Every wrong turn above was a missing feature, so each became one. Operator
ruling: a measurement records **which mode it ran under as a fact** — it does
not editorialize about whether that mode is the one production uses.

| feature | the failure it catches |
|---|---|
| `mode` (`eager`/`compiled`) on every measurement | a whole optimization aimed at an eager table while production runs compiled |
| `with_stack` + source frame per hotspot | `_index_put_impl_` read as the parameter gather when it was the `pos` gather |
| `summarize_device` raises on an empty window | a compiled hotspot table printed empty and read as "nothing is hot" |
| `resolved_settings()` — effective workers, whether the ceiling binds, cores, mem, peak RSS | 64 cores requested, 16 used, silently |
| `compute X / input Y → GPU busy N%` headline | computed by hand every round |
| `baseline_diff()` vs the last run of the same tier, 5 % noise band | medium-before nearly compared with large-after |
| `occupancy_warnings()` under 50 % live | the 513×513-per-55-atom solve was invisible until the source was read |
| kernel count + mean duration → launch vs compute/memory bound, with GFLOP | "gemm ≈ 50 %, so compute is done" was inference, not report |
| refuses to run when `production_calibration_drift()` is non-empty | the profile had been running ≤32-atom molecules at a 512 pack against a zero ε[Z] |
| throughput raises when the window is shorter than the prefetch queue | **202,356 molecules/s**, two orders above the hardware |

`record_function` inside the model was considered and **not** added: labelling
inside the compiled region risks a graph break, which costs more than the
readability buys. `with_stack` attribution reaches the same place from outside.

## Open

- **`n_terms` capacity**: one project-wide capacity (6 covers AMBER/CHARMM)
  versus a per-field capacity resolved at `to_potentials()`. The mask makes
  either safe.
- **molrs multi-term storage.** `Params` is `HashMap<String, f64>`, so a
  multi-term torsion is flattened to `k{m}`/`n{m}`/`d{m}` — which collides with
  the OPLS quartet's `k1..k4` (different meaning: LAMMPS `K_n` carries the ½).
  A guard now refuses the confusable direction, but the structural fix is for
  `Params` to hold per-type-per-term arrays as molnex does. Not attempted:
  it changes the type every molrs kernel reads.
- **molpy suite: 10 failures that predate this work and are not force-field
  vocabulary.** Confirmed by inspection, not assumed: the LAMMPS reader's
  `special_bonds` requirement (4 tests) and the GROMACS `[ defaults ]`
  requirement (3 tests) both exist at molrs `HEAD`, and the prmtop
  `gb_radius`/`gb_screen` column test plus the two `mrec` version tests live in
  molpy files this session never opened. They belong to the uncommitted work in
  `molrs/src/io/`, not here.
- **A stale extension binary hid two real defects for a day.** molpy's tests
  import the compiled `molrs-python/python/molrs/_lib.abi3.so`, built on
  2026-09-14, so nothing in this session's Rust reached them until a
  `maturin develop --release`. Once rebuilt, two molpy writer fixtures failed
  loudly: one still spelled the OPLS quartet `c1..c4` (the kernel reads
  `k1..k4`, and the old `f→c` fallback that covered it is gone), the other
  spelled a fourier torsion `k1`/`n1`/`d1`. Both now use the canonical names.
  **Any Rust force-field change must be followed by a maturin rebuild before a
  molpy or molnex test result means anything.**
- **GROMACS nonbonded is IO-only.** The reader never reads σ/ε: `[ atomtypes ]`
  contributes mass and charge, and the only LJ data it keeps is the cross-pair
  `c6`/`c12` of `[ pairtypes ]`. `pair:lj/cut` wants a per-atom-type
  `epsilon`/`sigma` self table, so a GROMACS-loaded field cannot build a LJ
  kernel at all, whatever the keys are called. Renaming `c6`/`c12` would only
  paint over that, so they keep their GROMACS spelling (units are converted)
  until the self-table gap is closed.
- **GROMACS coverage is partial.** The reader now normalises names and units
  for `periodic` / `rb` / `harmonic` dihedrals and for `lj12-6` / `buckingham`
  pairs, but its dihedral funct table still calls funct 3 `harmonic` with
  `psi0` / `k`, which is GROMACS' *improper* harmonic, not a proper torsion.
  Untouched here because it is a style-mapping question, not a naming one.
