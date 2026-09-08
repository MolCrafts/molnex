<!-- mol:note:topic:composable-potentials -->
# Composable interatomic potentials — spec ↔ molnex mapping and roadmap (2026-09-07)

Source: the user's design specification "Composable Interatomic Potentials"
(requirements **S1–S8**; profiles **A0/A1/A2** learned analytical parameters,
**B0/B1/B2** physics-augmented neural potentials; development order
A0 → A1 → B1 → B2, A2 after ablations). It describes a *proposed* framework.
This note records what molnex already has, what is missing, the shared type
designs both families will use, and the chain of executable specs. Only the
`composable-a0` chain is active in `.claude/specs/`; later chains are opened
with `/mol:spec` when reached and must re-read this note first.

## Decision

1. **Deliverable of this pass = mapping + roadmap, no code.** The A0 chain
   (six merge units) is the first executable work; A1 / iface / B1 / B2 / A2
   are roadmap rows below.
2. **First response module = variational induced dipoles** (Thole-damped,
   converged, diagnostics returned). The two existing non-variational
   responses (`EwaldMultipoleEnergy` one-shot, `MacePolar` finite recursion)
   stay as *distinct types* with their own energy definition; the manifest
   names the scheme (`"variational" | "direct" | "finite-iteration"`).
3. **F1 (periodic Coulomb exclusion bug) is fixed inside `composable-a0-03`**,
   not hot-fixed first — the parity gate that would have caught it lands in the
   same chain.
4. **No `lammps` / `openmm` Python imports anywhere under `src/`, `tests/`,
   `scripts/`. Anything that drives LAMMPS (subprocess `lmp` included) lives
   only under `interface/`** (`interface/tests/`, ctest, fixtures generated
   via `PYTHONPATH` to `src/`). LAMMPS is the production truth for classical
   parity; OpenMM export is text-golden only.
5. **A1 analytical export target = the compiled reference via `pair_style
   molnex`** (one permanent scale, d ≡ p). AMOEBA-style `d ≠ p` exports
   (OpenMM `AmoebaMultipoleForce`, LAMMPS `amoeba`) are not emitted.
6. **Family A parameter generation is once per chemical topology** —
   `Parameterizer` is topology-only, so `dθ/dx = 0` by construction
   (`src/molpot/forcefield/predictions.py:99-112`) and S5's "coordinate
   dependence of learned parameters" applies to Family B only.

## Global rules (binding for every chain below)

| Rule | Statement |
|---|---|
| Placement | Family A physics + contract + manifest in `molpot`; export plumbing in `molix.ff_export` / `molix.engine` (the one sanctioned `molix → molpot` edge is `molix.ff_export`; `molix.engine` takes a `Mapping`, never a molpot type). |
| S-decomposition | `E = E_local + E_disp + E_perm + E_pol + E_CT` is a **grouping of `term_energies()` by registry labels**, never a second energy path. `calc_energy` keeps summing `term_energies`, so the partition is by construction and double counting is impossible (S3). LJ splits into `lj_repulsion → local`, `lj_dispersion → disp`. |
| Contract | Every style in `PotentialMeta._REGISTRY` carries an `InteractionContract`; `Potentials.__init__` runs `PotentialMeta.check_composition` and raises on unsupported combinations (replaces the `Polarization` `UserWarning`). Unregistered styles raise (closes F6). |
| Manifest | `ForceField.manifest(*, profile)` owns it; `MANIFEST_SCHEMA` is versioned independently of `LAMMPS_META_SCHEMA`; it is persisted as a **sibling** `manifest.json` (`ModelManifest.write`; `export_for_lammps(manifest=m.to_dict())` writes `{name}.manifest.json` beside the AOTI export — a0-05 dropped the planned `LammpsWriter.write_manifest` pass-through as a façade) and as `run_record.write_meta(extra={"manifest": …})`. The C++ meta reader validates only `doc["lammps"]`, so no v3 bump. |
| Compile-clean | Everything on the MD force path obeys `compile-no-graph-breaks` (`notes.md`): fixed-capacity lists + masks, fixed-iteration solvers, `torch.where` guards, k-space shapes frozen at `bind_cell`. The single `.item()` for "converged?" lives in an observer hook, never in an energy. |
| Exclusions | Declared **per interaction class**: `NonbondedScaling(q=…, lj=…)` for permanent terms; `InducedScaling(perm_*, mutual_*)` for the induced term (A1). One permanent scale for field and energy (d ≡ p) — required for a variational functional to exist. |
| Total charge | Fail-closed: a `Parameterizer` with a `charge_model` requires `graphs.total_charge`; `Potentials.bind_topology` checks `Σ_g q = Q_g` eagerly. |
| Units | `from molix.units import units`; no restated conversion floats. α in Å³ with `k_e` outside the linear system (pending confirmation at A1 spec time — changes `ElementAlphaTable` and the polarization tests). |
| Tests | Unit tests under `tests/` mirror source; compile explains are `slow`; scenario goldens are hard-coded literals inside the mirrored unit tests (the `regressions/` tier was deleted 2026-09-07 — nothing is finished enough to regress against; MD-rollout gates are `slow`-marked); LAMMPS numbers only under `interface/tests/`. |

## Spec ↔ molnex mapping

Legend: ✅ exists · ◐ partial · ✗ missing. Anchors are `file:line` at commit 97c36852.

### S1 — separation and module contracts

| Requirement | Status | Anchor |
|---|---|---|
| Representation / parameter generation / interaction evaluation / engine integration separated | ✅ | `molrep` / `Parameterizer` (`molpot/forcefield/parameterizer.py:92`) / `Potentials` (`forcefield/potentials.py`) / `molix.engine`, `molix.ff_export` |
| System record (Z, pos, cell, pbc, total charge, topology, spin, field) | ◐ | batch `atoms/edges/graphs/bonds` + `angles/propers/impropers` columns; total charge via `AtomicChargeModel(total_charge)`; no external-field slot |
| Per-module declaration (inputs, shapes, units, transformation law, cutoff/LR solver, exclusions, derivatives) | ◐ | `ParamSpec(name, shape, unit, description)`, `StyleSchema(category, name, index_arity, params, energy_formula)`, `PotentialMeta._REGISTRY` (`forcefield/schema.py:15-174`) — params + units only |
| Assembled model returns E, named contributions, F, virial | ◐ | A: `Potentials.term_energies()` (`potentials.py:567`; keys bonds/angles/propers/impropers/lj/coulomb), `calc_energy_forces_virial` (`:348`). B: `MacePolar.energy_core` returns one unlabeled scalar (`molzoo/mace/polar.py:384`) |
| Response solvers return convergence diagnostics | ✗ | `Polarization._cg_solve` returns `x` only (`molpot/potentials/polarization.py:237-278`); EEM returns bare `q` (`charge/solvers/eem.py:115`) |
| Unsupported combinations rejected explicitly | ◐ | `Sonata` refuses induced heads (`composition/sonata.py:238-246`); `Polarization` only warns (`polarization.py:70,153`) |

### S2 — versioned manifest

| Requirement | Status | Anchor |
|---|---|---|
| Single versioned model manifest | ✗ | pieces: pydantic `MACEPolarSpec` / `SonataSpec` / `EwaldMultipoleEnergySpec` / `PermMultipoleHeadSpec`; LAMMPS `meta.json` v3 (`molix/engine/export.py:18-72`, `LAMMPS_META_SCHEMA = 3` at `:123`, pinned in C++); `molix.io.run_record.write_meta(extra=)` (`:90`); `BlockManifest` (data only); `ForceSpec.to_dict` = `dataclasses.asdict` precedent |
| Fields recorded nowhere | ✗ | enabled-terms list, polarization scheme, damping, exclusions/scaling, solver tolerances, reference method, energy offsets ↔ checkpoint, data version |

### S3 — no double counting

| Requirement | Status | Anchor |
|---|---|---|
| LJ repulsive → local, attractive → dispersion | ✗ | `lj126_pair_energy` one scalar (`potentials/vdw/lj126.py:23`); single `"lj"` term key |
| Explicit-vs-residual declared per configuration | ✗ | → S1 contract |
| Self-interaction counted once (B) | ✅ | `potentials/gto/hartree.py:398-401` |

### S4 — electrostatics and response

| Requirement | Status | Anchor |
|---|---|---|
| One multipole hierarchy q/μ/Θ with all cross terms | ◐ | `MultipoleEwaldKernel` f_qq/f_qu/f_uu/f_Qu/f_QQ (`elec/kernels/multipole_ewald.py:95,257,319`) — reciprocal is a bare k-sum, dense O(N²); mesh `Ewald/PME/P3MCalculator` are charge-only; `CalculatorDipole` (`elec/calculators/calculator_dipole.py`) is a disjoint μ–μ path (sparse real + Ewald k, no Thole, no exclusions, no q→field); GTO caps at `l_max ≤ 1` (`hartree.py:294`) |
| Permanent order ⟂ polarization scheme | ◐ | `PermMultipoleHeadSpec` toggles orders (`heads/multipole.py:89`); `MACEPolarSpec` couples `multipole_lmax` to the recursion (`molzoo/mace/spec.py:229`) |
| Local-frame convention for directional params | ✗ | `PermMultipoleHead` predicts in the lab frame (`heads/multipole.py:340`); nothing repo-wide |
| Damped variational induced dipoles (reference) | ◐ | `Polarization`: CG on (α⁻¹ − T)μ = E, Thole (`polarization.py:84,201,237`); energy equals the minimum numerically but the functional is unexpressed, A ≻ 0 never checked, exclusions ignored, unrolled tape, data-dependent `break` |
| Direct / finite-iteration responses as distinct types | ✅ | `EwaldMultipoleEnergy` one-shot μ = αE (`elec/ewald_multipole.py:343-349`); `MacePolar` 2-step non-SCF (`polar.py:164-174`) |
| Polarizability positive-definite | ◐ | `PolarizabilityHead` softplus on α_iso only (`heads/electrostatics.py:191-194,289`) |

### S5 — derivatives and constraints

| Requirement | Status | Anchor |
|---|---|---|
| Forces + cell derivatives incl. learned-param coordinate dependence | ✅ (B) / n.a. (A) | autograd through heads; Family A topology-only (Decision 6) |
| Virial/stress consistently defined | ✅ | symmetric-strain Cauchy (`derivation/stress.py:25`, `force.py:141`, commit d362a948) |
| Total charge explicit | ✅ / ◐ | EEM bordered KKT, per-graph `(B,)` (`charge/solvers/eem.py:115`, `charge/problem.py:26`); `AtomicChargeModel.total_charge = 0.0` is a silent default; multipole head uses a mean shift (`multipole.py:520`) |
| Per-fragment constraints | ✗ | one constraint row per graph |
| Separate exclusions for permanent vs induced | ◐ (a0-02) | `NonbondedScaling(q=, lj=)` keyed per interaction class with `kinds()/triple()` and a once-per-topology `ExclusionTable` (`forcefield/exclusions.py`); the `induced` kind arrives in a1-01; `Polarization` still applies none |
| Non-convergence reported, model unchanged | ✗ | — |

### Family A

| Requirement | Status | Anchor |
|---|---|---|
| Graph encoder → separate heads | ✅ | `Parameterizer(encoder, template, *, bond_head, angle_head, proper_head, improper_head, atom_head, charge_model)` |
| Repulsion / dispersion / multipole / polarizability heads on the Parameterizer | ✗ | composer-only heads (`composition/heads.py:120,163`) never reach `ForceField` / IR — two parallel paths |
| Constrained positive outputs | ✅ | `composition/mm_heads.py:96`, `charge/chi_eta.py:53`, `heads/electrostatics.py:174,289` |
| Parameterize once per topology, cached in MD | ◐ | `model(batch)` → filled `ForceField` → `to_potentials()` (`forcefield/container.py:180`); no topology fingerprint / staleness check |
| A0 point charges under PBC (Python) | ✅ with F1 | `pair:coulomb` → `CoulombPeriodicPotential` (`forcefield/pair_terms.py:152`) → charge-only Ewald/PME/P3M |
| Export cached params → LAMMPS / OpenMM | ✅ (a0-05, 2026-09-08) | `LammpsAdapter` emits `pair_style lj/mdf` / `lj/cut/coul/long` / `hybrid/overlay … coul/long` + `kspace_style pppm` + `special_bonds` from `PotentialIR.nonbonded_method` + scaling; `LammpsWriter.write_pair_block` owns the pair section (explicit LB-mixed `pair_coeff i j`), `write_settings(skip_pair_style=True)` bonded-only include, box from `graphs.cell`, LAMMPS-legal zero-padded identity labels, `nonbonded_scaling` mirrored into molpy `special_bonds`; `OpenMMAdapter` PME / switching text records; `export_for_lammps(manifest=)` sibling. Verified against LAMMPS: butane 2.6e-12, periodic water+ions 9.9e-9 (`interface/tests/parity_classical.py`) |
| Smooth cutoff (continuity gate) | ✅ (a0-02) | quintic switch on `LJCutPotential(cutoff, switch_start)` (`pair_terms.py`), exported as LAMMPS `lj/mdf` (identical taper) / OpenMM `useSwitchingFunction` (a0-05) |

### Family B

| Requirement | Status | Anchor |
|---|---|---|
| MACE backbone; MACE-POLAR-1 with interaction preserved | ✅ | `MacePotential` (`molzoo/mace/potential.py:124`), `MacePolar` (`polar.py:56`); `_from_official` rejects unsupported configs (`polar.py:386`) |
| Fine-tuning bases | ✅ | `MaceMatpes` / `MaceOmol` (`potential.py:903,986`) |
| Allegro / reduced MACE production student | ◐ | `Allegro` encoder-only; fused `tp_method` not fullgraph (`molzoo/allegro.py:319,400-406`) |
| Distillation machinery | ✗ | none in `src/`, `scripts/` |
| PME for multipoles, per-order tested; Ewald↔PME parity | ✗ | `PMECalculator` charge-only (`elec/calculators/pme.py:14`); `tests/test_molpot/test_elec/calculators/test_calculator.py` is shape-only |
| Compile-clean long-range path | ◐ | GTO yes (`hartree.py:394`; one-graph explain test); `ewald_multipole.py:217,264` host-syncs |

### S6 — data (MolHub / molexp scope)

| Requirement | Status | Anchor |
|---|---|---|
| Group holdout splits | ✅ | `molix.batch.group_split_indices` (`molix/batch/group_split.py:11`) |
| Missing labels masked, not zeroed | ◐ | task-routed losses (`TaskSpec` / `TaskRouter`, `molix/core/steps/multitask.py:92,113`) — one task label per batch; no per-sample label mask inside a batch |
| Composition / provenance / splits | out of molnex | MolHub (`learnable-classical-ff.md` boundary table); calibration evidence is a molexp experiment (`production-training-checklist.md`) |

### S7 — deployment

| Requirement | Status | Anchor |
|---|---|---|
| Differentiable reference | ✅ | `molpot` |
| Family B compiled neural potential in LAMMPS | ✅ short-range only | `export_for_lammps` (`molix/engine/export.py:352`), `interface/lammps/`; meta v3 has **no cell / charge / long-range / global-constraint** field (`interface/lammps/README.md:417`) |
| No host sync in the force loop | ✅ (law) | `notes.md` topic `compile-no-graph-breaks` |
| Local vs long-range benchmarked independently | ◐ | ad hoc per-term fp64 splits (`notes.md` Polar parity); no harness |
| Polarization iteration counts | ✗ | `MDObservables` (`molix/md/types.py:47-80`) has no field |
| Strong / weak scaling | ✗ | no multi-rank runner |
| Reference-vs-production agreement | ✅ classical under `interface/tests/parity_classical.py` (a0-05); neural tool deleted with `regressions/` — F5 | export-time gates `engine/export.py:81-92`; recorded runs in `interface/lammps/README.md` |

### S8 — release gates

| Gate | Status | Anchor |
|---|---|---|
| Translation / rotation / permutation | ✅ | `tests/test_molzoo/test_symmetry.py`, `tests/test_molpot/test_composition/test_sonata_invariants.py`, `test_forcefield/test_wilson.py:229,251` |
| Reflection | ✗ | none repo-wide |
| Charge conservation | ✅ | `test_charge/test_eem.py:54,67,99`; sonata ac-005 |
| FD force + FD cell derivative | ✅ (Sonata / Ewald / MACE), ◐ (A0 `Potentials`) | `test_sonata_periodic.py` ac-008/009, `test_derivation/test_force.py:52`; A0 has analytic-vs-autograd (`test_forcefield/test_analytic_forces.py`) but no FD stress on the assembled model |
| Reference vs production | ✅ (F5) | parity scripts |
| Cutoff continuity | ◐ | unit-level `PolynomialCutoff` / ZBL only |
| NVE energy drift | ◐ | criterion `|slope·T| / |E_tot(0)| < 1e-3` in `benchmarks/verify_md_ljcut_nve.py:36-39` — manual, not a gate |
| Versioned release (weights + manifest + provenance + validation) | ◐ | molexp run identity + `production-training-checklist.md`; no manifest |

## Shared type designs

### S1 contract — `src/molpot/contract.py` (new, torch-free leaf) + `forcefield/schema.py`

```python
class Contribution(StrEnum): LOCAL; DISP; PERM; POL; CT
class CutoffKind(StrEnum): NONE; HARD; SWITCHED; EWALD; GTO_KSPACE
class ExclusionClass(StrEnum): NONE; PERMANENT; INDUCED
class Derivative(StrEnum): FORCES; VIRIAL; HESSIAN
class ParamFrame(StrEnum): SCALAR; VECTOR_LAB; VECTOR_LOCAL; RANK2   # not `Frame`: collides with molpy.Frame

@dataclass(frozen=True)
class TermSpec:
    key: str                      # term_energies() key, e.g. "lj_repulsion"
    contribution: Contribution

@dataclass(frozen=True)
class InteractionContract:
    inputs: tuple[str, ...]       # batch keys, e.g. ("atoms.pos", "atoms.q", "bonds.bond_index")
    cutoff: CutoffKind
    exclusion: ExclusionClass
    derivatives: frozenset[Derivative]
    terms: tuple[TermSpec, ...]
    response: str | None = None   # POL only: "variational" | "direct" | "finite-iteration"
```

- `ParamSpec` gains `frame: ParamFrame = ParamFrame.SCALAR`; `StyleSchema` gains
  `contract: InteractionContract`.
- `PotentialMeta.contract_for(key) -> InteractionContract` raises `KeyError`
  on an unregistered style (no synthesised entry — closes F6).
- `PotentialMeta.check_composition(style_keys) -> None` raises `ValueError`:
  more than one `POL` style; a `POL` style without at least one `PERM` style
  (the same-cutoff-family relation HARD ↔ HARD /
  EWALD ↔ EWALD was dropped in a0-01: `pair:coulomb` is labelled `EWALD` as a
  *policy* label while the runtime solver is recorded by the manifest, which made
  the family rule unsatisfiable — **a1-01 must declare `pair:induced/thole`
  against a solver-compatibility relation, not the label**); `INDUCED` exclusion
  without `ForceField.induced_scaling` (checked in `Potentials.__init__`).
- Registry labels: bonded → `LOCAL` / `NONE` / `{FORCES, VIRIAL, HESSIAN}`;
  `pair:lj/cut` → terms `(lj_repulsion: LOCAL, lj_dispersion: DISP)`,
  `PERMANENT`, cutoff `SWITCHED`; `pair:coulomb` → `(coulomb: PERM)`,
  `PERMANENT`, cutoff `HARD` (open) / `EWALD` (cell); A1 adds
  `pair:induced/thole` → `(induced: POL)`, `INDUCED`, `{FORCES, VIRIAL}`,
  `response="variational"`.
- `Potentials.contribution_energies(system, *, pos=None) -> dict[Contribution, Tensor]`
  groups `term_energies()` by `TermSpec`; `calc_energy` is unchanged.
- Family B declares the same records in B1 (`MacePolar.contract`,
  `Sonata` style list) — out of A0 scope.

### S2 manifest — `src/molpot/manifest.py` (new; top-level so molzoo imports it in B1)

`MANIFEST_SCHEMA = 1`. Small frozen records, JSON via `dataclasses.asdict`
(enums → `.value`, frozenset → sorted list):

| Record | Fields |
|---|---|
| `RepresentationRecord` | `family, encoder, checkpoint_id, spec: dict` |
| `InteractionRecord` | `style, contract: InteractionContract, params: tuple[ParamSpec, ...], options: dict` |
| `NonbondedMethodRecord` (`molpot.ir.nonbonded`, a leaf `PotentialIR` may import) | `lj_cutoff, lj_switch_start, coulomb_cutoff` (open-boundary cutoff or None), `coulomb_calculator` (`"ewald"\|"pme"\|"p3m"`, the configured periodic solver), `real_space_cutoff, smearing, lr_wavelength, mesh_spacing, capacity_factor` — effective assembly defaults when the style is declared, all-None fields when not; always a record, never `None` |
| `ElectrostaticsRecord` | `multipole_order, polarization_scheme, damping, charge_constraint, total_charge, permanent_scaling: NonbondedScaling, induced_scaling: InducedScaling \| None, solver: SolverRecord \| None` |
| `SolverRecord` | `kind, iterations, tolerance, preconditioner` |
| `ParamTableRecord` | `category, names, shapes, dtype, sha256` |
| `NumericsRecord` | `dtype` (= `str(molix.config["ftype"])`), `matmul_precision` (= `torch.get_float32_matmul_precision()` — there is no `config["matmul_precision"]` key; `set_precision` sets torch's policy directly), `energy_dtype` |
| `ProvenanceRecord` | `reference_method, energy_offsets_id, data_version, software: dict, checkpoint_id, run_id, created_at, git_sha` (`run_id` = molexp identity as a string, never an import) |
| `ModelManifest` | `schema_version, profile, units, interactions, nonbonded, electrostatics, tables, numerics, representation=None, provenance=None`; `to_dict / from_dict / write / read` |

Owner: **`ForceField.manifest(*, profile) -> ModelManifest`** — styles,
`style_options`, tables and scaling all live on the FF; `Potentials` is its
torch view. `Parameterizer.representation_record()` supplies the
representation; the caller composes representation + provenance with
`dataclasses.replace` (composition is the caller's job — no façade).
Persistence: `ModelManifest.write(path)` (a0-04; the planned `LammpsWriter.write_manifest` was dropped in a0-05 as a pass-through façade) and (ff_export may
import molpot); `export_for_lammps(..., manifest: Mapping | None)` writes
sibling `{name}.manifest.json` with `meta.json` byte-identical;
training runs `run_record.write_meta(extra={"manifest": manifest.to_dict()})`. `profile` is validated (`"A0"` only until an A1 builder exists). The FF reports only what it owns: its `ElectrostaticsRecord` carries `permanent_scaling` and `multipole_order = 0` with `charge_constraint = "none"` / `solver = None`; the charge model supplies its own — `AtomicChargeModel.electrostatics_record(*, permanent_scaling)` (`"total-charge-kkt"`, `SolverRecord("electronegativity-equalization", 1, 0.0, "none")`) — and the caller composes it with `dataclasses.replace`, exactly like `representation_record`. Pair-style options are resolved **once** (`PairStyleDefaults`, incl. `real_space_cutoff_factor`): the record is built first and the pair terms are constructed from it, so `to_potentials()` and `to_ir()` cannot disagree (seam test). `ParamSpec` lives in `molpot.contract` (re-exported by `forcefield.schema`) so `molpot.manifest` never imports the `forcefield` package.

### A1 variational polarization (design; implemented by `composable-a1`)

- Functional `F(u, x) = ½ uᵀA(x)u − uᵀe(x)`, `A = α⁻¹ − S ∘ T_Thole`.
  `A = Aᵀ` because pair scales are keyed on undirected pairs and Thole
  damping uses `a_ij = a_i a_j`. Energy `E_pol = k_e · F(u*, x)`.
- **Forces and virial by the envelope theorem**: solve for `u*`, `detach()`,
  evaluate `F` on the live `A(x, cell)`, `e(x, cell)`; autograd then gives
  `∂ₓF(u*, x)` which equals `dE/dx` exactly at convergence — no unrolled
  solver tape. Through `calc_energy_forces_virial`'s strained backward the
  virial is exact too. Unconverged after K steps: the energy is the upper
  bound `E + ½ r_KᵀA⁻¹r_K` and the gradient omits `r_Kᵀ ∂ₓu_K`; both are
  bounded by the reported `residual_norm`, so nothing is dropped silently
  (S5). Hessian is declared unsupported (`derivatives = {FORCES, VIRIAL}`);
  `HessianDerivation` on an induced FF is refused by the contract.
- Solver `DiagonalPCG(iterations=K)` in `molpot/potentials/elec/solvers/pcg.py`:
  fixed K, no `break`, `torch.where` step guards, per-graph scalars by
  `index_add` over `atom_batch` (removes the present cross-molecule coupling
  of CG step sizes), padded atoms masked (`rhs = 0`, identity rows).
  Returns `ResponseDiagnostics(residual_norm, delta_norm, curvature_min,
  gershgorin_min, iterations)` as tensors, `converged(tol) -> bool tensor`.
  `curvature_min ≤ 0` is a necessary "not PD" flag; `gershgorin_min > 0` is a
  sufficient PD certificate (the "no polarization catastrophe" condition).
  The one `.item()` lives in an observer (`molix.md` hook / eval hook) that
  raises or logs by policy — never inside `forward` / `term_energies`.
- `VariationalInducedDipole` replaces `Polarization` (name removed; the
  module-level LES flag and `UserWarning` go with it — `check_composition`
  is the guard). `EwaldMultipoleEnergy` (`response="direct"`) and `MacePolar`
  (`"finite-iteration"`, `delta_norm = ‖Δdensity‖` of the last update) gain
  diagnostics only.
- `InducedScaling(perm_12, perm_13, perm_14, mutual_12, mutual_13, mutual_14)`
  beside `NonbondedScaling`. A1 policy: d ≡ p (one permanent scale for field
  and energy), mutual 1/1/1 under Thole (Tinker `mutual-1x-scale`).
  Polarization groups are an A2 field (`groups = "none"`).
- Head: `Parameterizer(polarizability_head=)` → `atoms.alpha`, `atoms.thole`
  (softplus); `PolarizabilityBag(alpha, thole)`; registry `pair:induced/thole`
  with option `thole` default when the head omits it. Isotropic α suffices
  for A1; PD anisotropic α is A2. Verify ByteFF-Pol's intramolecular
  exclusion policy in `/mol:litrev` before `composable-a1-03`.
- Periodic: extend `CalculatorDipole` (Thole real-space correction, excluded
  pair subtraction, charge → field via `f_qu` on the same list) as the `T·u`
  matvec; `MultipoleEwaldKernel` is the dense test oracle (O(N²), mask
  indexing — never the MD path). PME for dipoles is `composable-b1-03`.
- Export: compiled reference via `pair_style molnex` (Decision 5).
  Open-boundary A1 exports on meta v3 today; **periodic A1 export waits on
  `composable-iface-01-cell`** (cell + charges inputs, global reductions).

## Roadmap

### `composable-a0` — active (six merge units)

**Test-tier decision (2026-09-08, a0-06).** A bounded MD rollout *is* permitted under `tests/` when it is `slow`-marked and asserts one gate (the 200-step NVE drift on the butane-pair box) — the CLAUDE.md "two tiers" paragraph says so ("… or rolls out MD is marked `slow`"), while the older "Unit tests only … no MD rollouts" sentence predates the deletion of `regressions/` and reads as the fast tier's rule. Reconcile that sentence in CLAUDE.md when the tree is quiet (the peer session holds CLAUDE.md hunks today); `tests/test_molix/test_md/test_integrators.py`'s docstring is corrected in a0-06.

| Step | Scope | Gate it closes |
|---|---|---|
| `composable-a0-01-contracts` | `molpot/contract.py`; `ParamSpec.frame`, `StyleSchema.contract`, `contract_for`, `check_composition`; LJ split `lj_repulsion` / `lj_dispersion`; `Potentials.contribution_energies`; absent table raises; unregistered / induced styles refused | S1, S3; F2, F6 |
| `composable-a0-02-exclusions-switching` | `ExclusionScales` + `NonbondedScaling(q=, lj=)`; `ExclusionTable` built once per topology, fixed-shape `lookup`; quintic switch `S = 1 − x³(10 − 15x + 6x²)` on `LJCutPotential`; capacity-shaped `NeighborListBuilder._from_placed`; fail-closed `graphs.total_charge` | S5 exclusions + total charge, S8 continuity; F3 (FF path) |
| `composable-a0-03-periodic-coulomb` | full-`1/r` excluded-pair subtraction under Ewald/PME (**F1**); `CoulombPeriodicPotential.bind_cell` (`CellBound`) with frozen k-space shape; `pair_mask` instead of branches; PME default under a cell; FD force + FD strain tests on the assembled `Potentials` | S5 derivatives + virial; F1, F3 (k-space) |
| `composable-a0-03b-neighbor-list-to-molix` | delete `molpot/forcefield/neighbor_list.py` (a third enumerator, in the physics layer); `Potentials._placed_pairs` consumes the molix-placed list and fails closed without one; fixtures rebuilt through `molix.batch.NeighborList`; half-list orientation checked at bind | layering (operator, 2026-09-07); closes F4 |
| `composable-a0-04-manifest` | `molpot/manifest.py` records; `ForceField.manifest(*, profile)`; `Parameterizer.representation_record()`; `PotentialIR.nonbonded_method` (F7) | S2; F7 |
| `composable-a0-05-export` | LAMMPS `hybrid/overlay lj/mdf coul/long` + `kspace_style pppm` + `special_bonds`; OpenMM PME + switching (text only); `LammpsWriter.write_pair_block` (+ `write_settings(skip_pair_style=True)`, box from `graphs.cell`, zero-padded identity labels, `nonbonded_scaling` → molpy `special_bonds`; `write_manifest` dropped — `ModelManifest.write`); `export_for_lammps(manifest=)`; classical parity moved to `interface/tests/parity_classical.py` with a `--case periodic` (run for real: butane 2.6e-12, periodic 9.9e-9); ruff `TID251` bans `lammps`/`openmm`-import check | S7; F5 (classical) |
| `composable-a0-06-gates` | reflection symmetry test (with a chiral negative control); `EnergyDriftHook(MDHook)` in `molix.md.runner`; gate tests in the mirrored unit modules (T/R/P/reflection, Σq, FD force + strain, continuity, 200-step NVE drift as `slow`, manifest round trip, contribution partition) | S8 |

Acceptance criteria live in `.claude/specs/composable-a0-0N-*.acceptance.md`.

### `composable-a1` — variational induced dipoles (opened after A0 closes)

| Step | Scope | Key acceptance |
|---|---|---|
| `01-scaling-ir` | `InducedScaling`, `PolarizabilityBag`, `PotentialIR.induced_scaling` / `.polarizability`, `pair:induced/thole` registry entry, `ForceField(induced_scaling=)` | `to_ir()` round-trips α / thole; `perm` triple reproduces the `q` masks when equal |
| `02-pcg` | `DiagonalPCG` + `ResponseDiagnostics` | random SPD residual < 1e-8 at K = 32 fp64; constructed indefinite A → `curvature_min ≤ 0`; dynamo explain 0 breaks (slow) |
| `03-variational-term` | `VariationalInducedDipole` (open boundary), envelope forces; deletes `Polarization` + LES flag; fixes `.item()` host sync | FD gradients w.r.t. pos / q / α / thole ≤ 1e-6 rel; equals `−½ μ·e` at convergence to 1e-10; K = 1 gradient error ≤ C · `residual_norm` |
| `04-periodic` | Ewald `e` / `T` via the `CalculatorDipole` extension; F3 elec host syncs fixed (`ewald_multipole.py:217,264`, `from_dist` masks); direct / finite-iteration diagnostics on `EwaldMultipoleEnergy` and `MacePolar` | dimer in a large box == open to 1e-6; dense-oracle agreement 1e-8 |
| `05-wire` | `Parameterizer(polarizability_head=)`, `induced` term in `Potentials`, `ElectrostaticsRecord`, MD observer hook, open-boundary `pair_style molnex` export, A1 gate tests in the mirrored unit modules | FD-strain virial ≤ 1e-6; unconverged step raises via the hook only; manifest names `polarization_scheme = "variational"` |

### `composable-iface` — engine bridge extension (shared by A1-periodic, B1, B2)

| Step | Scope | Key acceptance |
|---|---|---|
| `01-cell` | meta v4: `cell` + per-atom `charges` inputs; global reductions per `mace-polar-10k-review.md` §"Global operations that must remain exact" (Σq constraint once, `|Σ_r S_r(k)|²` not `Σ_r |S_r|²`); bumps `LAMMPS_META_SCHEMA` on both sides; the `LAMMPS_META_SCHEMA` pin in `tests/test_molix/test_engine/test_export.py` and `interface/tests` re-pinned | 1-rank / N-rank / in-process three-way agreement on a periodic charged box (pattern `interface/tests/` parity) |

### `composable-b1` — physics-augmented foundation potential

| Step | Scope | Key acceptance |
|---|---|---|
| `01-contributions` | `MacePolar.energy_core` → named E0 / local / electron / Hartree; `MacePolar.contract`; `manifest()` via the same records | `Σ contributions == energy_core` to 1e-10 fp64; `energy_core` still 0 breaks |
| `02-response` | finite-iteration `ResponseDiagnostics` on `MacePolar`; decouple `multipole_lmax` from `num_recursion_steps` in `MACEPolarSpec` (S4 independence) | official checkpoint loads unchanged; diagnostics tensor-only |
| `03-pme-multipole` | PME for q + μ (+ Θ) replacing the bare k-sum in `EwaldMultipoleEnergy`; `PMECalculator` multipole-aware; retire `CalculatorDipole` as a disjoint path; compile-clean | Ewald ↔ PME parity ≤ 1e-6 per order; explain 0 breaks |
| `04-adapt` | molexp calibration experiment for polymer-electrolyte adaptation (`production-training-checklist.md`-gated); molnex change limited to per-sample label masks in losses (S6) | checklist sign-off record; held-out backbone / salt split |
| `05-gates` | B1 chain regression: symmetry incl. reflection, Σq, FD force + stress, NVE drift with polarization iteration counts, manifest | as A0-06 |

### `composable-b2` — distilled production student

| Step | Scope |
|---|---|
| `01-allegro-potential` | `AllegroPotential` with fullgraph compile (`tp_method="naive"` or a `setup_context`-capable fused kernel) |
| `02-distill` | teacher-label task in `MultiTaskTrainStep`; independent QM labels retained (data in MolHub) |
| `03-scaling-harness` | strong / weak scaling report; local vs long-range timed separately; polarization iteration counts in `MDObservables` |

### `composable-a2` — after controlled ablations

| Step | Scope |
|---|---|
| `01-local-frame` | documented z-then-x / bisector frame for `PermMultipoleHead` (`Frame.VECTOR_LOCAL`) |
| `02-pd-alpha` | Cholesky-parameterized anisotropic α |
| `03-quadrupole` | Θ in GTO / PME; polarization groups |

## Findings ledger (iron law — found during the mapping; none left silent)

| # | Finding | Anchor | Route |
|---|---|---|---|
| F1 | **Physics bug.** Periodic Coulomb drops scale-0 pairs from the real-space list and rescales only the `erfc` part for 1-4 pairs; the `erf(r/σ√2)/r` (k-space) part of every 1-2 / 1-3 / 1-4 pair stays at full strength. LAMMPS `special_bonds` + `coul/long` and OpenMM exceptions subtract the full `(1 − s) q_i q_j / r`. The only test is "finite and negative". | `src/molpot/forcefield/pair_terms.py:203-249`; `tests/test_molpot/test_forcefield/test_coulomb.py:195` | `composable-a0-03` |
| F2 | Declared `pair:lj/cut` / `pair:coulomb` with absent tables silently return zero (S1 breach; distinct from empty topology). | `forcefield/potentials.py:728-731,787-789` | `composable-a0-01` |
| F3 | Graph breaks on the MD force path: `LJCutPotential.energy` boolean slicing (`pair_terms.py:49-64`); `_mask_pairs_in` `torch.unique` (`potentials.py:895-910`); Coulomb `numel()` / `allclose` branches (`pair_terms.py:220,243`); `EwaldCalculator._compute_kspace` live-cell `ns` → k-vector shape (`elec/calculators/ewald.py:64-71`); `NeighborListBuilder._from_placed` `[:n_live]` / `[keep]` (`forcefield/neighbor_list.py:89-107`). On the A1 surface: `Polarization.forward` `int(batch.max().item())`, `EwaldMultipoleEnergy` per-graph loop + `bool(det…)` (`ewald_multipole.py:217,264`), `MultipoleEwaldKernel.from_dist` mask assignment. | as listed | `a0-02` / `a0-03` (FF path); `a1-03` / `a1-04` (elec) |
| F4 | `NeighborListBuilder._periodic` kept same-`atom_batch` pairs only (wrong for packed periodic training batches). | (deleted) | **closed by deletion in a0-03b**: molpot no longer enumerates pairs; `Potentials._placed_pairs` consumes the list `molix.batch.NeighborList` / `molix.md.NeighborList` produce and fails closed without one |
| F5 | Two LAMMPS-driven parity scripts lived outside `interface/` (Decision 4). Neither imports `lammps` (subprocess only); placement was the violation. **Classical half closed by a0-05** (2026-09-08): `interface/tests/parity_classical.py` (`--case butane|periodic`, ctest `SKIP_RETURN_CODE 77`, recorded runs in `interface/lammps/README.md`). The neural `scripts/lammps_iface_parity.py` was deleted from the tree by the peer session's `regressions/` cleanup (b5fbe9f3) and its runbook rows in the README still cite it — named debt for `lammps-iface-post-chain`. | `interface/tests/parity_classical.py`; `interface/lammps/README.md` | neural → `lammps-iface-post-chain` |
| F6 | `PotentialMeta.schema_for` synthesises an "unregistered" `StyleSchema` instead of failing. | `forcefield/schema.py:177-192` | `composable-a0-01` |
| F7 | `ForceField.to_ir()` loses cutoffs / calculator options; the parity script hard-codes `pair_style lj/cut 10.0`. | `container.py:220-270`; the parity script's hard-coded line (now `interface/tests/parity_classical.py`, which renders the block from the IR) | `composable-a0-04` (closed) |
| F9 | A hand-built *parallel* `Sonata` + `Polarization` graph is unguarded: `Sonata` refuses `Polarization` only when nested in `short_range_head` (`composition/sonata.py:245`), and `Polarization` is not a registered style, so `check_composition` never sees it; the deleted ambient LES flag was the only (non-compile-safe) guard. | `potentials/polarization.py` class docstring states the real coverage | closes in `composable-a1-03` when `VariationalInducedDipole` (registered `pair:induced/thole`) replaces `Polarization` |
| F10 | Deferred nits from the a0-01/a0-02/a0-03 reviews: `_require_bonded_params` re-checks a construction-time fact at 8 per-call sites; `calc_energy_forces` (132 lines) / `term_energies` (109) / `CoulombPeriodicPotential.energy` (144) exceed the 80-line budget; `pair_terms.py` duplicates the per-graph scatter block; (the six mypy-syntax pragmas of the deleted `neighbor_list.py` went with it in a0-03b); `InteractionContract.response` is stringly typed; `KSpaceKernel.kernel_from_k_sq` names its `|k|²` argument `kvectors` (`elec/lib/kspace_filter.py:31`); `tests/test_molpot/test_charge/test_eem.py` mirrors `charge/solvers/eem.py` and should live at `test_charge/test_solvers/test_eem.py` (kernel tests likewise under `test_charge/test_kernels/`); `tests/test_molzoo/test_imports.py:105-106` carries `@pytest.mark.slow` twice; `AtomicChargeModel.electrostatics_record` hard-codes `kind="electronegativity-equalization"` instead of deriving it from the injected solver (a1-01 adds solvers); `src/molix/engine/adapter.py:166-282` — `read_outputs` / `read_energy_triple` never narrow their `object` / `TensorCollection | Tensor` inputs (7 `ty` warnings, found by a0-05); `src/molpot/forcefield/container.py:realize` — `get_style` returns `Style | None` and `astyle.def_type` is called unnarrowed (pre-existing `ty` warning on a line a0-05 touched); `LammpsAdapter._pair` is ~98 lines (LJ sub-style, Coulomb sub-style, style fusion, kspace records — split into private steps); `export_for_lammps` crossed 80 lines by the manifest block. | `forcefield/potentials.py`, `contract.py` | `/mol:refactor` on `potentials.py` (queued with its function-length / duplication handoffs from the a0-01 simplify pass) / a1-01 (`Response(StrEnum)` with its first producer) |
| F11 | Latent exclusion-precedence bug: `Potentials._pair_scales` assigned the 1-3 scale after the 1-2 scale **unguarded** (`potentials.py` pre-a0-02), so on a 3-membered ring a directly bonded pair read `s13`; only the 1-4 write was guarded. Unobservable on acyclic topologies (butane golden unaffected). | `forcefield/potentials.py` (deleted in a0-02) | fixed in a0-02: `ExclusionTable` implements 1-2 ≻ 1-3 ≻ 1-4 (LAMMPS `special_bonds` semantics) with a cyclopropane test |
| F12 | `scripts/mm_param_learning/espaloma_mm_protocol_train.py:439-456` builds `Parameterizer(..., charge_model=AtomicChargeModel(...))` on batches carrying `graphs.mm_energy` / `graphs.task` but never `graphs.total_charge`; since a0-02 the Parameterizer is fail-closed on total charge (S5), so this workflow raises at its first forward until its collate emits an explicit `total_charge` column (zeros for neutral molecules). | `scripts/mm_param_learning/espaloma_mm_protocol_train.py:439-456` | user experiment script (molexp side) — not a molnex library change; fix the dataset/collate column before the next run |
| F13 | **Physics bug (periodic virial).** `NeighborListBuilder._from_placed` rebuilt pair vectors as `pos[t] − pos[s] + (edge_diff − Δpos).detach()`, i.e. with the lattice shift *detached*, so every pair that reaches its partner through a lattice vector contributed no shift-strain term: on a wrapped 4-ion box `tr W_auto = 254.35` vs central-FD / virial-theorem `268.70` kcal/mol; an unwrapped fixture of the same system was exact. Found by the a0-03 tester while pinning ac-003. | then `forcefield/neighbor_list.py:148-149` (now `Potentials._placed_pairs`), `potentials.py` `calc_energy_forces_virial` | fixed in a0-03: `_from_placed` prefers the non-detached `edges.shifts` branch and `calc_energy_forces_virial` strains the shifts it evaluates; the `edge_diff`-only fallback is documented as unwrapped-only |
| F14 | The 8-ion rocksalt cube is ill-posed as a Madelung golden under minimum image (nearest neighbours sit at exactly `L/2`, the half-list keeps one of two equidistant images and the real-space sum halves); the a0-03 golden uses a 2×2×2 supercell (64 ions, edge 11.28 Å) with `E = −(N/2)·M·k_e/d`, `M = 1.74756459463`. | `tests/test_molpot/test_forcefield/test_coulomb.py` | ac-002 wording corrected at close |
| F15 | Convention boundary of the a0-03 exclusion correction: each excluded pair is corrected at its **minimum image** only. In a box smaller than `2·r_c` the same pair can enter the real-space list at more than one image and only the minimum-image copy is corrected, whereas LAMMPS `special_bonds` excludes every image by atom tag. All a0-03 pins sit inside `L/2`; boxes below `2·r_c` are out of contract until a per-image correction is specified. | `forcefield/pair_terms.py` (`CoulombPeriodicPotential.energy` docstring) | documented boundary; revisit if a1-04 needs small-box support |
| F16 | Pre-existing hot-path break: `Potentials` energy paths called `ForceField._first_style` (a walk of the molrs style registry, untraceable by dynamo) once per bonded branch on every energy call (`forcefield/container.py:152`); exposed by the a0-03 step-level explain once the k-space breaks were gone. | `forcefield/container.py:152`, `forcefield/potentials.py` | fixed in a0-03: declared bonded categories resolved once at assembly (`Potentials._bonded_styles`) |
| F17 | Bound PME still host-synced inside `elec/lib/mesh_interpolator.py` (`MeshInterpolator.update`) and `elec/lib/kspace_filter.py` (`KSpaceFilter._prep_kvectors`), which held `ns_mesh` as a tensor and read its elements — so after a0-03 made PME the default under a cell, only the Ewald route was fullgraph-clean. | `elec/lib/mesh_interpolator.py`, `elec/lib/kspace_filter.py` | fixed in a0-03 (integer `ns_mesh` on the bound path; PME-parametrized explain test) |
| F18 | `ExclusionTable.__init__` builds its keys with boolean-mask indexing (`exclusions.py:145`), a data-dependent shape — by design a **bind-time** operation (`Potentials.bind_topology`, like the eager Σq check), so a first-ever energy call made *inside* a compiled region would break there. Contract: bind topology (and cell) eagerly before entering `torch.compile`; `ForceFieldPredictions` and `MD` do. | `forcefield/exclusions.py:145` | documented boundary (bind before compile); no force-path change |
| F19 | Two owners of the frozen-reciprocal-shape policy: `gto.kspace.KSpaceGrid` (half-box integer grid, `capacity_factor` 1.2, `max_axis_index` ceiling, `(1,3,3)` accepted) and a0-03's `CoulombPeriodicPotential.bind_cell` / `_reciprocal_shape` (fftfreq full grid / PME mesh, `capacity_factor` 1.0, no axis ceiling). Same freeze / capacity-check / in-graph-NaN pattern, two implementations — the spec deferred the unification on purpose (different grids). | `forcefield/pair_terms.py:~501`, `gto/kspace.py` | `/mol:refactor`: one owner of bind + check + NaN guard (generalise `KSpaceGrid`), both k-space users size through it; also gives PME/Ewald the axis ceiling |
| F20 | Periodic Coulomb under a cell with `num_graphs > 1` computed one scalar with graph 0's cell that `calc_energy` broadcast to every graph — silently wrong per-graph energies on packed periodic batches. | `forcefield/potentials.py:~1210,~1025` | a0-03 fails closed (`ValueError`); real per-graph periodic support is F4's spec |
| F21 | `scripts/mm_param_learning/{class_i_fit.py:260-263, espaloma_mm_protocol_train.py:492, _probe_nan_term.py:58-64}` (the former `parity_lammps_classical_md.py` is gone — a0-05 moved it under `interface/tests/` with placed edges) build `OPLSAAForceField` potentials on `molix.batch.DataModule` batches that carry no `NeighborList` task, i.e. no `edges` namespace; they relied on `NeighborListBuilder`'s silent all-pairs enumeration, which a0-03b deletes. | those scripts | user experiment scripts (molexp side): add `molix.batch.NeighborList(cutoff=<pair cutoff>, symmetry=True)` to the batch task chain before the next run |
| F22 | Two placed-list producer quirks recorded by the a0-03b sweep: `molix.batch.NeighborList(symmetry=False)` emits `source > target` rows, which read entirely dead under the `source < target` half-list rule (caught at `bind_topology` since a0-03b); and no in-tree producer writes `edges.num_edges` (`molix.md.NeighborList` pads with dead self-loops instead), so the `row < num_edges` liveness leg is optional. | `molix/batch/neighbor.py`, `molix/md/neighbors.py` | documented on `Potentials._placed_pairs`; publishing `num_edges` from `molix.md` is a molix follow-up |
| F23 | `scripts/mm_param_learning/{class_i_fit.py:34,40,57,154, align.py:29-528, espaloma_mm_protocol_train.py:88,306}` import `TargetSchema` / `FrameSample`, both deleted when `collate_molecules` went Frame-native (peer session `molnex-5b`, 2026-09-08); the three scripts fail at import; `_probe_nan_term.py:64-66` also calls the private `Potentials._lj_energy_forces` / `_coulomb_energy_forces` with a stale arity (broken at HEAD before a0-06). Porting them means rewriting their sample-construction path to emit molpy `Frame`s and cannot be verified without their datasets. | those scripts | user experiment scripts (molexp side) — port before the next run; joins F12 / F21 on the same files |
| F24 | **Scientific.** Every LJ pair in molnex is mixed by Lorentz–Berthelot (`lj126.lorentz_berthelot:125`: σ arithmetic, ε geometric) — including `OPLSAAForceField`, whose literature rule is geometric in both (Jorgensen 1996). The mixing rule is model identity yet appears in neither `PotentialIR` nor `NonbondedMethodRecord` / `ModelManifest`. a0-05 exports explicit `pair_coeff i j` lines computed with molnex's own function, so the deck matches molnex as computed and a rule change flows through automatically. | `src/molpot/potentials/vdw/lj126.py:85-125`, `forcefield/pair_terms.py:226,308`, `ir/nonbonded.py` | own spec after a0: `mixing` as a `pair:lj/cut` style option recorded in `NonbondedMethodRecord` + manifest, OPLS template → geometric; re-pin the a0-01/02 LJ goldens |
| F25 | `LammpsWriter` uses identity types (one atom type per atom); with explicit `pair_coeff i j` lines (a0-05) the pair block scales as N²/2 lines and the data file carries N type labels. | `src/molix/ff_export/lammps_writer.py` (`to_frame`, `write_pair_block`), `forcefield/container.py:realize` | type deduplication by parameter equality (same ε/σ/q/mass → one type; bonded types by parameter tuple) — own `/mol:refactor` after a0; not blocking |
| F26 | `k_e` differs between molnex (`332.0637132991921`, from pint) and LAMMPS `real` (`qqrd2e = 332.06371`) by 9.9e-9 relative — a hard floor on any molnex↔LAMMPS Coulomb comparison; the periodic parity gate (1e-6) sits well above it. | `interface/lammps/README.md` runbook | nothing to fix; documented |
| F27 | **Physics bug (fixed in a0-06, 2026-09-08).** `LJCutPotential.energy_forces` built the force direction from the raw unwrapped `pos[target] − pos[source]` while `dist` was the minimum-image distance from `Potentials._placed_pairs`, so every pair whose minimum image crosses the cell got a non-unit, mis-directed `r̂`: on a 256-atom argon box the analytic force was 1.6e-3 where a finite difference of the (correct) energy gave 3.8e-5, and NVE pumped 9 % of the energy in ten steps. Only the analytic MD branch was affected (autograd forces differentiate the correct energy). Fix: `_placed_pairs` returns the minimum-image pair vector it already forms, `energy_forces` takes it (`r̂ = vector / dist`); gate: `test_potentials.py` boundary FD test + the a0-06 NVE drift test. | `src/molpot/forcefield/pair_terms.py:332`, `potentials.py:_placed_pairs` | a0-06 |
| F28 | `benchmarks/verify_md_ljcut_nve.py` (and any `molix.md.MD` user): `--no-compile` / `--cuda-graphs` are dead — `MD` has no compile seam since d9043cee swapped the compiled `LennardJonesCutForceField` for the molpy `ForceField` path; the split-precision arm (`--potential-precision`) cannot place parameter buffers off the MD dtype and now fails closed instead of silently running at `--precision`. The benchmark also crashed at its summary print (`NameError: compiled`) and double-converted ε — both fixed in a0-06. | `benchmarks/verify_md_ljcut_nve.py`, `src/molix/md/driver.py` | `/mol:refactor` on `MD`: one `compile_core`-style seam (the `Potentials.compile_core` pattern) + a precision policy; `benchmarks/verify_md_lj_nve.py:125-127` also reaches into `MD._wire()` / `MD._adapter` (private) — same seam; not in this chain |
| F29 | **Contract gap (found by the a0-06 translation gate).** `Potentials` bonded terms read plain `pos` displacements while the pair terms read minimum-image ones, so `calc_energy` on a periodic frame whose molecules are wrapped across a box face returns a wrong bonded energy silently — the normal state of a frame read back from a wrapped trajectory. a0-06 states the *unwrapped-molecule* precondition on the public entry points; the decision (reject wrapped frames at `bind_topology` by checking every bonded displacement against half the box, or minimum-image the bonded terms) is open. | `src/molpot/forcefield/potentials.py` (`calc_energy`, `_rebuilds_pairs`), `tests/test_molpot/test_forcefield/test_potentials.py` (`_translated_whole_molecules`) | own `/mol:spec` after a0 (candidate: fail-closed check at bind, since MD never wraps in place) |
| F8 | (unverified, outside the touched surface) `tests/test_molzoo/test_mace/test_variants.py` and `src/molzoo/specs/mace_omol.md:3` reference the folded-away `mace/variants.py`. | — | note only; check when `molzoo/mace` is next touched |

## Decisions pending (resolve at each chain's spec time)

- α unit convention: Å³ with `k_e` outside the linear system (A1-01).
- Default `iterations` / `tolerance` for `DiagonalPCG`; observer policy
  raise vs log on non-convergence (A1-05).
- F4 periodic training batches (own spec).
- Whether LAMMPS `lj/mdf` is exactly the quintic `(1 − x)³(1 + 3x + 6x²)`
  (algebraically equal to OpenMM's `1 − x³(10 − 15x + 6x²)`) — verify against
  the LAMMPS docs in A0-05 before emitting it.

## Relations

Builds on `learnable-classical-ff.md` (ForceField path, EEM charges) and
`mace-polar-10k-review.md` (Polar rollout, global reductions). Supersedes
nothing; the `Polarization` type and its LES flag are retired by
`composable-a1-03`.
