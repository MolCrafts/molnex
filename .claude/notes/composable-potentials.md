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
| Manifest | `ForceField.manifest(*, profile)` owns it; `MANIFEST_SCHEMA` is versioned independently of `LAMMPS_META_SCHEMA`; it is persisted as a **sibling** `manifest.json` (`LammpsWriter.write_manifest`, `export_for_lammps(manifest=)`) and as `run_record.write_meta(extra={"manifest": …})`. The C++ meta reader validates only `doc["lammps"]`, so no v3 bump. |
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
| Separate exclusions for permanent vs induced | ✗ | one `NonbondedScaling` (`ir/scaling.py:24-41`); `Polarization` applies none |
| Non-convergence reported, model unchanged | ✗ | — |

### Family A

| Requirement | Status | Anchor |
|---|---|---|
| Graph encoder → separate heads | ✅ | `Parameterizer(encoder, template, *, bond_head, angle_head, proper_head, improper_head, atom_head, charge_model)` |
| Repulsion / dispersion / multipole / polarizability heads on the Parameterizer | ✗ | composer-only heads (`composition/heads.py:120,163`) never reach `ForceField` / IR — two parallel paths |
| Constrained positive outputs | ✅ | `composition/mm_heads.py:96`, `charge/chi_eta.py:53`, `heads/electrostatics.py:174,289` |
| Parameterize once per topology, cached in MD | ◐ | `model(batch)` → filled `ForceField` → `to_potentials()` (`forcefield/container.py:180`); no topology fingerprint / staleness check |
| A0 point charges under PBC (Python) | ✅ with F1 | `pair:coulomb` → `CoulombPeriodicPotential` (`forcefield/pair_terms.py:152`) → charge-only Ewald/PME/P3M |
| Export cached params → LAMMPS / OpenMM | ◐ | bonded + `pair_style lj/cut` + per-atom charges (`molix/ff_export/lammps_adapter.py:250-275`, `openmm_adapter.py:198`); **no `kspace_style` / `coul/long` / PME, no `special_bonds`** |
| Smooth cutoff (continuity gate) | ✗ | `LJCutPotential` hard truncation (`pair_terms.py:49-64`) |

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
| Reference-vs-production agreement | ✅ misplaced (F5) | `scripts/lammps_iface_parity.py`, `scripts/mm_param_learning/parity_lammps_classical_md.py`; export-time gates `engine/export.py:81-92` |

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
| `NonbondedMethodRecord` | `lj_cutoff, lj_switch_start, coulomb: "open"\|"ewald"\|"pme"\|"p3m", real_space_cutoff, smearing, mesh_spacing` (shared by `PotentialIR.nonbonded_method`) |
| `ElectrostaticsRecord` | `multipole_order, polarization_scheme, damping, charge_constraint, total_charge, permanent_scaling: NonbondedScaling, induced_scaling: InducedScaling \| None, solver: SolverRecord \| None` |
| `SolverRecord` | `kind, iterations, tolerance, preconditioner` |
| `ParamTableRecord` | `category, names, shapes, dtype, sha256` |
| `NumericsRecord` | `dtype, matmul_precision, energy_dtype` |
| `ProvenanceRecord` | `reference_method, energy_offsets_id, data_version, software: dict, checkpoint_id, run_id, created_at, git_sha` (`run_id` = molexp identity as a string, never an import) |
| `ModelManifest` | `schema_version, profile, units, interactions, nonbonded, electrostatics, tables, numerics, representation=None, provenance=None`; `to_dict / from_dict / write / read` |

Owner: **`ForceField.manifest(*, profile) -> ModelManifest`** — styles,
`style_options`, tables and scaling all live on the FF; `Potentials` is its
torch view. `Parameterizer.representation_record()` supplies the
representation; the caller composes representation + provenance with
`dataclasses.replace` (composition is the caller's job — no façade).
Persistence: `LammpsWriter.write_manifest(path, manifest)` (ff_export may
import molpot); `export_for_lammps(..., manifest: Mapping | None)` writes
sibling `{name}.manifest.json` with `meta.json` byte-identical;
training runs `run_record.write_meta(extra={"manifest": manifest.to_dict()})`.

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

| Step | Scope | Gate it closes |
|---|---|---|
| `composable-a0-01-contracts` | `molpot/contract.py`; `ParamSpec.frame`, `StyleSchema.contract`, `contract_for`, `check_composition`; LJ split `lj_repulsion` / `lj_dispersion`; `Potentials.contribution_energies`; absent table raises; unregistered / induced styles refused | S1, S3; F2, F6 |
| `composable-a0-02-exclusions-switching` | `ExclusionScales` + `NonbondedScaling(q=, lj=)`; `ExclusionTable` built once per topology, fixed-shape `lookup`; quintic switch `S = 1 − x³(10 − 15x + 6x²)` on `LJCutPotential`; capacity-shaped `NeighborListBuilder._from_placed`; fail-closed `graphs.total_charge` | S5 exclusions + total charge, S8 continuity; F3 (FF path) |
| `composable-a0-03-periodic-coulomb` | full-`1/r` excluded-pair subtraction under Ewald/PME (**F1**); `CoulombPeriodicPotential.bind_cell` (`CellBound`) with frozen k-space shape; `pair_mask` instead of branches; PME default under a cell; FD force + FD strain tests on the assembled `Potentials` | S5 derivatives + virial; F1, F3 (k-space) |
| `composable-a0-04-manifest` | `molpot/manifest.py` records; `ForceField.manifest(*, profile)`; `Parameterizer.representation_record()`; `PotentialIR.nonbonded_method` (F7) | S2; F7 |
| `composable-a0-05-export` | LAMMPS `hybrid/overlay lj/mdf coul/long` + `kspace_style pppm` + `special_bonds`; OpenMM PME + switching (text only); `LammpsWriter.write_pair_block` / `write_manifest`; `export_for_lammps(manifest=)`; classical parity moved to `interface/tests/parity_classical.py` with a `--periodic` case; static no-`lammps`/`openmm`-import check | S7; F5 (classical) |
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
| F4 | `NeighborListBuilder._periodic` keeps same-`atom_batch` pairs only — right for one-box MD, wrong for packed periodic training batches. | `neighbor_list.py:161-163` | recorded; not A0-blocking; own spec when periodic training batches are needed |
| F5 | Two LAMMPS-driven parity scripts live outside `interface/` (Decision 4). Neither imports `lammps` (subprocess only); placement is the violation. | `scripts/mm_param_learning/parity_lammps_classical_md.py`, `scripts/lammps_iface_parity.py` | classical → `interface/tests/` in `a0-05`; neural → named debt for `lammps-iface-post-chain` |
| F6 | `PotentialMeta.schema_for` synthesises an "unregistered" `StyleSchema` instead of failing. | `forcefield/schema.py:177-192` | `composable-a0-01` |
| F7 | `ForceField.to_ir()` loses cutoffs / calculator options; the parity script hard-codes `pair_style lj/cut 10.0`. | `container.py:220-270`; `parity_lammps_classical_md.py:184` | `composable-a0-04` |
| F9 | A hand-built *parallel* `Sonata` + `Polarization` graph is unguarded: `Sonata` refuses `Polarization` only when nested in `short_range_head` (`composition/sonata.py:245`), and `Polarization` is not a registered style, so `check_composition` never sees it; the deleted ambient LES flag was the only (non-compile-safe) guard. | `potentials/polarization.py` class docstring states the real coverage | closes in `composable-a1-03` when `VariationalInducedDipole` (registered `pair:induced/thole`) replaces `Polarization` |
| F10 | Deferred nits from the a0-01 architect review: `_require_bonded_params` re-checks a construction-time fact at 8 per-call sites; `InteractionContract.response` is stringly typed. | `forcefield/potentials.py`, `contract.py` | a0-02 (one construction-time bonded check) / a1-01 (`Response(StrEnum)` with its first producer) |
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
