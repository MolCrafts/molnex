---
title: Class-I prediction adapter (ClassicalIMM → per-task prediction dict)
status: approved
created: 2026-08-14
grilled: true
---

# Class-I prediction adapter (ClassicalIMM → per-task prediction dict)

## Summary

The four Class-I task losses (`energy_force_mse`, `ProjectedInternalHessianLoss`, `TorsionProfileLoss`, `ChargeMSELoss` in `src/molix/core/losses/molecular.py`) all consume a plain prediction mapping keyed `energy` / `forces` / `hessian` / `q`, but the Class-I model path produces a `ForceField` (`ClassicalIMM.forward` → `ff.to_potentials().calc_energy(batch)`). Today that last hop is hand-written inline in each training script (`scripts/mm_param_learning/espaloma_mm_protocol_train.py:386-399` builds the model, `:425` evaluates energy). This spec adds one primitive library type, `ClassIPredictions`, in `molpot.forcefield`: constructed with a `ClassicalIMM` plus the *declared* prediction keys for one task, it returns exactly those predictions and nothing else — energy and forces from `Potentials` in a single autograd pass, the Cartesian Hessian from the injected `molpot.derivation` Hessian derivation (spec `universal-class-i-06`), and `q` read from the topology-only EEM charges that the model already wrote onto the ForceField. It lives in molpot because it needs `to_potentials()` and molix must not import molpot; the *caller* (experiment script, or the multi-task step of spec 11) constructs one adapter per task, so no molix module gains a molpot dependency.

## Domain basis

Class-I energy is evaluated by the existing `Potentials` container (harmonic bond/angle, multi-term cosine propers, impropers, LJ, Coulomb); this spec derives nothing new about the functional form. What it fixes is the *derivative and charge contract* of the mapping:

- **Forces.** `F = −∂E(x; θ(G))/∂x`, kcal mol⁻¹ Å⁻¹. Class-I parameters `θ` are functions of the covalent graph `G` only, so the total derivative equals the partial: `dE/dx = ∂E/∂x + (∂E/∂θ)(dθ/dx)` with `dθ/dx = 0`. This is also the force the exported/deployed force field produces, so the trained and deployed forces agree by construction. Obtained through the canonical autograd backend (`src/molpot/derivation/force.py:99` `autograd_forces_from_energy`), never a hand-rolled third path.
- **Hessian.** `H_ij = ∂²E/∂x_i∂x_j`, shape `(3N, 3N)`, kcal mol⁻¹ Å⁻². Consumed by `ProjectedInternalHessianLoss`, which projects with the Wilson `B` matrix, `H_q = (B⁺)ᵀ H_x B⁺` (kcal mol⁻¹ Å⁻² on bond rows, kcal mol⁻¹ rad⁻² on angle rows). Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941). https://doi.org/10.1063/1.1750829
- **Charges.** EEM/QEq KKT system `[A 1; 1ᵀ 0][q; λ] = [−χ; Q]` with `A = diag(η)` (`DiagonalHardnessKernel`), solved by `ElectronegativityEqualization`; `q` in elementary charge `e`. With a diagonal hardness kernel the solve reads no positions, so `dq/dx = 0` and the force above carries no charge-response term — the **MEDIUM science constraint** for force consistency. Mortier, Van Genechten, Gasteiger, *J. Am. Chem. Soc.* **107**, 829 (1985) / Mortier, Ghosh, Shankar, *J. Am. Chem. Soc.* **108**, 4315 (1986), https://doi.org/10.1021/ja00275a013; Rappé & Goddard, *J. Phys. Chem.* **95**, 3358 (1991), https://doi.org/10.1021/j100161a070.
- **Multi-task Class-I fitting protocol** (energies + forces + charges + torsion profiles on one parameterization): Wang, Fass, Chodera, *Chem. Sci.* **13**, 12016 (2022), https://doi.org/10.1039/D2SC02739A; Takaba et al., arXiv:2307.07085 (2023).
- **Units.** Inherited from the ForceField parameter tables — Class-I convention here is kcal/mol, Å, kcal mol⁻¹ Å⁻¹, kcal mol⁻¹ Å⁻², `e`. The adapter restates no conversion floats (`molix.units` if any are ever needed).
- **Analytic validation case (hard-coded goldens).** Two-atom harmonic bond, `E = ½k(r−r0)²` (`src/molpot/potentials/bonds/harmonic.py:12`), `k = 100` kcal mol⁻¹ Å⁻², `r0 = 1.0` Å, `r = 1.5` Å along `x̂`:
  `E = 12.5` kcal/mol; `F₀ = (+50, 0, 0)`, `F₁ = (−50, 0, 0)` kcal mol⁻¹ Å⁻¹;
  `H = k ûûᵀ ⊕ (k(r−r0)/r)(I − ûûᵀ)` blocks → `H[0,0] = +100`, `H[0,3] = −100`, `H[1,1] = 100·0.5/1.5 = 33.333333333333336` kcal mol⁻¹ Å⁻².
  EEM with `χ = (0, 1)`, `η = (2, 2)`, `Q = 0` → `λ = −0.5`, `q = (+0.25, −0.25) e` (the more electronegative atom is negative).

## Design

**Owner type.** New plain class `ClassIPredictions` in `src/molpot/forcefield/predictions.py`. Shape check rule 1 (natural owning type) resolves *against* the two candidates already in tree: `ClassicalIMM` is forbidden from owning energy by the binding note (`.claude/notes/learnable-classical-ff.md`: "No energy method on `ClassicalIMM` — energy is always on `Potentials`"), and `Potentials` knows neither the model/charge path nor the per-task key set (it is constructed *after* the model forward, from one already-filled ForceField). The mapping "model + batch + declared task keys → prediction dict" is therefore its own single responsibility.

**Closest pattern (followed).** `molpot.derivation.energy_readout.EnergyReadout` — a plain (non-`nn.Module`) class, `model` as first positional argument, requested behaviour **bound at construction** so `__call__` is one monomorphic path, keyword-only options, fail-loud `ValueError` on an unknown mode. Being a plain class (not `nn.Module`) also keeps the trainer's parameter set unambiguous when several per-task adapters wrap one shared model. Method names are the prediction keys (`energy` / `forces` / `charges`), mirroring `AtomicChargeModel.charges`; the fused pass borrows the `Potentials.calc_energy_forces` naming as `energy_forces`.

**Public surface.**

```python
class ClassIPredictions:
    SUPPORTED_KEYS = ("energy", "forces", "hessian", "q")

    def __init__(self, model: ClassicalIMM, *, keys: Sequence[str],
                 hessian_derivation: Any | None = None) -> None: ...
    def __call__(self, batch) -> dict[str, torch.Tensor]: ...   # exactly the declared keys
    def energy(self, batch) -> torch.Tensor                     # (B,) — () for a single graph
    def forces(self, batch) -> torch.Tensor                     # (N, 3)
    def energy_forces(self, batch) -> tuple[torch.Tensor, torch.Tensor]   # one pass
    def hessian(self, batch) -> torch.Tensor                    # (3N, 3N)
    def charges(self, batch) -> torch.Tensor                    # (N,)
```

**Laziness is structural, not conditional.** `keys` is fixed at construction and `__init__` builds the call plan: `("energy",)` → energy only; `("energy", "forces")` → one `energy_forces` pass; `("q",)` → model forward + read `atoms.q`, with **no** `Potentials` assembly and no energy evaluation at all; `("energy", "hessian")` → energy plus one injected-derivation call. `__call__` walks that fixed plan, so a prediction the task did not declare is never computed and never appears in the dict. There is no `keys="all"`, no default key set, and no method that computes everything — this is the anti-façade requirement made structural.

**One model forward per call.** Each public method (and each `__call__`) performs exactly one `self.model(batch)` and, when a derivative is needed, one `ff.to_potentials()`. Forces come from `Potentials.calc_energy_forces` (single energy evaluation, `create_graph` following ambient grad state, so the force loss back-propagates into the heads). Parameters `θ` are produced outside the position graph, which is exactly the fixed-parameter force of the Domain basis; the geometry-independence precondition on `q` is what makes this exact, and it is enforced empirically by a unit test (identical `q` across two conformers of one topology), not assumed silently.

**Hessian by injection, not by re-derivation.** The Hessian derivation delivered by `universal-class-i-06` is passed in (`hessian_derivation=`), following the injection style of `AtomicChargeModel(head, solver)` / `ClassicalIMM(encoder, heads…)`. `"hessian" in keys` without an injected derivation is a construction-time `ValueError`. The adapter calls it with the same `(energy_fn, pos)` shape `ForceDerivation.forward` uses, where `energy_fn = lambda p: pots.calc_energy(batch, pos=p)`; if spec 06 lands a different call shape, that one line is the only seam that binds to it. `predictions.py` contains no `torch.autograd.grad`, no `torch.func.*`, no `torch.autograd.functional.hessian`.

**Composition decides what is evaluated — the adapter never forces a term.** Energy always goes through `ff.to_potentials()`, which assembles only the styles the ForceField declares (`Potentials._assemble_pair_potentials`, `potentials.py:78`). A bonded-only ForceField therefore evaluates no LJ and no Coulomb even when an `atom_head` (`LJParameterHead`, `src/molpot/composition/heads.py:12`, frozen in v0) has written `epsilon` / `sigma` into `atoms` params. The adapter adds no LJ-aware branch of its own.

**Fail loud.** `ValueError` at construction for: empty `keys`, duplicate keys, an unsupported key (message lists `SUPPORTED_KEYS`), `"q"` declared while `model.charge_model is None`, `"hessian"` declared without a derivation. `RuntimeError` at call time if `"q"` was declared but the filled ForceField carries no `atoms["q"]` — never a silent zero vector.

**Iron law — debt found in the touched surface.** `Potentials.calc_forces` / `Potentials.calc_energy_forces` (`potentials.py:185-205`) hand-roll `torch.autograd.grad`, duplicating the canonical `molpot.derivation.autograd_forces_from_energy` and contradicting the standing rule "models must not invent a third force path" (`.claude/notes/notes.md`, 2026-07-29). Since this adapter's forces flow through exactly those two methods, they are generalized here (numerics unchanged: same `create_graph = torch.is_grad_enabled()` default) rather than built around. Also: `__all__` in `src/molpot/forcefield/__init__.py` and `src/molpot/__init__.py` is unsorted, which the notes rule requires fixing when a name is inserted — done in the export task.

### Reuse decision

No `librarian_report` was supplied with this task; the survey below was performed directly (`Read`/`Grep` over `src/molpot`, `src/molix/core/losses`, `scripts/mm_param_learning`) and every candidate is resolved.

- `reuse ClassicalIMM.forward` (`src/molpot/forcefield/classical_imm.py:69`) — the adapter calls it; no encoder/head/charge wiring is reimplemented.
- `reuse ForceField.to_potentials` / `Potentials.calc_energy` / `Potentials.calc_energy_forces` — energy and the fused single-pass force.
- `generalize autograd_forces_from_energy` (`src/molpot/derivation/force.py:99`) — promote `Potentials.calc_forces` / `calc_energy_forces` to delegate to it, so the adapter, MD bind path and export share one force path (spawns the "Generalize" task below instead of new force code in the adapter).
- `reuse AtomicChargeModel.write_forcefield` (`src/molpot/charge/assignment.py:65`, invoked inside `ClassicalIMM.forward`) — `q` is read from `ff.params("atoms")["q"]`; the EEM KKT solve is never re-run by the adapter.
- `reuse` the Hessian derivation exported by `molpot.derivation` (spec `universal-class-i-06`) — injected, never re-derived.
- `pattern EnergyReadout` (`src/molpot/derivation/energy_readout.py:23`) — construction, init-time path binding, error style, docstring shape.
- `reuse` prediction-key vocabulary `energy` / `forces` / `hessian` / `q` from `PesBundle` targets (`src/molix/data/pes_bundle.py:23`) and the loss defaults — declared as a class-level tuple in molpot, no molix private import.
- `new — ClassIPredictions`: nothing in tree maps a model path onto a task-keyed prediction dict; the only two plausible owners are excluded by the binding note (`ClassicalIMM`) and by construction order / missing knowledge (`Potentials`).

## Files to create or modify

- src/molpot/forcefield/predictions.py (new)
- src/molpot/forcefield/potentials.py
- src/molpot/forcefield/__init__.py
- src/molpot/__init__.py
- tests/test_molpot/test_forcefield/test_predictions.py (new)
- tests/test_molpot/test_forcefield/test_potentials.py (new)
- regressions/universal-class-i-08-adapter.py (new)

## Tasks

- [ ] Write failing unit tests for ClassIPredictions (tests/test_molpot/test_forcefield/test_predictions.py → TestClassIPredictions): declared-keys-only dict, energy shape for B=1 and B=2, energy/force parity vs the direct ClassicalIMM+Potentials+ForceDerivation path, one-pass energy_forces equivalence, hessian present only when declared, q parity vs AtomicChargeModel and geometry independence, bonded-only composition evaluating no LJ/Coulomb, and all fail-loud error cases
- [ ] Write failing unit test for the canonical force delegation (tests/test_molpot/test_forcefield/test_potentials.py → TestPotentials): calc_forces / calc_energy_forces equal autograd_forces_from_energy on the same energy and keep force→parameter double backward
- [ ] Generalize Potentials.calc_forces and Potentials.calc_energy_forces in src/molpot/forcefield/potentials.py to delegate to molpot.derivation.autograd_forces_from_energy (no third force path; numerics unchanged)
- [ ] Implement ClassIPredictions in src/molpot/forcefield/predictions.py (init-time key plan; energy / forces / energy_forces / charges / injected hessian_derivation; construction- and call-time fail-loud errors)
- [ ] Export ClassIPredictions from src/molpot/forcefield/__init__.py and src/molpot/__init__.py, re-sorting both `__all__` lists alphabetically per the notes rule
- [ ] Add Google-style docstrings per doc.style with units and tensor shapes (kcal/mol, kcal mol⁻¹ Å⁻¹, kcal mol⁻¹ Å⁻², `e`; `(B,)` / `(N, 3)` / `(3N, 3N)` / `(N,)`) to every public symbol in src/molpot/forcefield/predictions.py
- [ ] Verify against the analytic harmonic-bond and EEM case in tests/test_molpot/test_forcefield/test_predictions.py: E=12.5 kcal/mol, F=(±50,0,0) kcal mol⁻¹ Å⁻¹, H[0,0]=100 / H[0,3]=−100 / H[1,1]=33.333333333333336 kcal mol⁻¹ Å⁻², q=(+0.25,−0.25) e, fp64 atol 1e-10
- [ ] Add regression example regressions/universal-class-i-08-adapter.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

Unit tests mirror source: `src/molpot/forcefield/predictions.py` → `tests/test_molpot/test_forcefield/test_predictions.py` (`ClassIPredictions` → `TestClassIPredictions`), `src/molpot/forcefield/potentials.py` → `tests/test_molpot/test_forcefield/test_potentials.py` (`Potentials` → `TestPotentials`). Each test targets a single method. Fixtures follow the existing `test_classical_imm.py` pattern: a deterministic stub encoder plus constant-output heads in fp64, so every expected value is analytic.

- **Happy path.** Energy parity: adapter `energy(batch)` equals `model(batch).to_potentials().calc_energy(batch)` exactly. Force parity: adapter `forces(batch)` equals `ForceDerivation(method="autograd")(lambda p: pots.calc_energy(batch, pos=p), pos)`. Fused parity: `energy_forces` equals the two separate calls (atol 1e-10). Charge parity: adapter `q` equals `AtomicChargeModel.charges(features, batch)` built from the same head and solver.
- **Declared-keys contract.** `keys=("energy",)` → dict keys exactly `{"energy"}`; `"hessian"` and `"q"` absent. With a call-counting stub Hessian derivation and a call-counting charge head, counts are 0 when those keys are not declared. `keys=("energy","hessian")` → `hessian` shape `(6, 6)` and symmetric within 1e-10.
- **Edge cases.** Energy shape `()` for a single-graph batch and `(2,)` for a two-molecule batch; bonded-only ForceField with an `LJParameterHead` `atom_head` → energy equals the bond golden exactly and `Potentials.term_energies` keys ⊆ `{bonds, angles, propers, impropers}`; `pytest.raises` for empty `keys`, duplicate keys, unsupported key, `"q"` without `charge_model`, `"hessian"` without a derivation, and a missing `atoms["q"]` after a forward — each message naming the offending condition.
- **Domain validation (hard-coded).** The two-atom harmonic-bond case from Domain basis (`k=100`, `r0=1.0`, `r=1.5`): `E=12.5`, `F=(±50,0,0)`, `H[0,0]=100`, `H[0,3]=−100`, `H[1,1]=33.333333333333336`, fp64 atol 1e-10, with the `E=½k(r−r0)²` derivation and the Wilson citation in the test docstring. EEM case `χ=(0,1)`, `η=(2,2)`, `Q=0` → `q=(+0.25,−0.25)`. Geometry independence: `q` for two different conformers of the same topology is bit-identical (`torch.equal`), pinning `dq/dx = 0`.
- **Regression example.** `regressions/universal-class-i-08-adapter.py` — minimal public-API script: build a `ClassicalIMM` with constant heads over a two-atom bonded topology plus a constant `ChiEtaHead`-shaped stub, construct `ClassIPredictions` twice (energy+forces; q-only), and assert the hard-coded analytic goldens above plus "q-only produced no energy key". All goldens are literals derived analytically in-file (no third-party oracle, no `subprocess`, no network); header records `capture: PYTHONPATH=src python regressions/universal-class-i-08-adapter.py` and the date.

## Out of scope

- Task routing, per-task batch selection (which batches are equilibrium), loss weighting and the multi-task step — spec 11. The adapter never decides which keys a batch gets; the caller declares them at construction.
- The Hessian derivation itself (spec `universal-class-i-06`) and the Wilson `B` builder (spec `universal-class-i-07`) — consumed here, implemented there.
- Per-molecule Hessian block extraction / padding for multi-molecule batches. `hessian(batch)` returns the batch-level Cartesian `(3N, 3N)`; the Hessian task uses single-molecule batches, and block slicing belongs to the routing spec.
- Geometry-dependent charge kernels. `ShieldedCoulombKernel` makes `q` a function of positions, which would add a `dq/dx` term the fixed-parameter force path does not carry; forces/Hessian tasks are v0-supported only on the topology-only `DiagonalHardnessKernel` path. A later spec introducing a geometry-dependent kernel must either add that term or reject the force/Hessian tasks — named here rather than left implicit.
- LJ parameter learning: v0 freezes `LJParameterHead`; unfreezing it, and any change to which styles a ForceField declares, is the training-config spec's business.
- Migrating `scripts/mm_param_learning/espaloma_mm_protocol_train.py` (and its siblings) onto the adapter — the generalized call shape lands here; the script switchover happens with the consuming spec so mid-flight experiments are not disturbed.
- Writing predictions in place onto the batch (`graphs.energy` / `atoms.forces`). The adapter returns a plain dict for the loss contract; the in-place `EnergyReadout` / `ForceReadout` path is untouched.
