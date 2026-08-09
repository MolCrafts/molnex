---
slug: md-neighborlist-skin-07-wire
criteria:
  - id: ac-001
    summary: exactly one rebuild-policy owner remains in the tree
    type: code
    evaluator_hint: "grep + pytest tests/test_molix/test_md/"
    pass_when: |
      `grep -rn "rebuild_every\|_force_eval_count\|NeighborListHook" src/molix/
      tests/ benchmarks/ scripts/ docs/ regressions/` returns no matches (the
      names may survive only in CHANGELOG.md, .claude/specs/ and the append-only
      run-log rows of src/molzoo/specs/mace_matpes.md); hasattr(molix.md,
      "NeighborListHook") is False and it is absent from molix.md.__all__ (which
      stays alphabetised); MD(HarmonicForceField(k=1.0), mass=1.0, dt=0.01,
      rebuild_every=1) raises TypeError.
    status: pending
  - id: ac-002
    summary: the integrator's rebuild switch is derived, static, and overridable
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_integrators.py tests/test_molix/test_md/test_forcefield.py tests/test_molix/test_md/test_driver.py"
    pass_when: |
      Integrator(force).rebuild is False for ForceField / HarmonicForceField /
      LennardJonesForceField / PotentialForceField / CallableForceField(neighbors=None)
      and True for LennardJonesCutForceField / PeriodicPotentialForceField /
      CallableForceField(neighbors=<list>), matching each type's
      rebuilds_neighbors property; explicit rebuild=False and rebuild=True
      override the derivation; md.integrator.rebuild is still True when the force
      field is wrapped by MD(autocast_dtype=torch.bfloat16) and when it is wrapped
      by torch.compile(ff, backend="eager"); no getattr(force/neighbors, ...,
      default) duck-read appears in src/molix/md/.
    status: pending
  - id: ac-003
    summary: the policy runs once per force evaluation at the evaluated positions
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_integrators.py::TestIntegratorRebuildSwitch tests/test_molix/test_md/test_forcefield.py"
    pass_when: |
      With rebuild=True, eval_force calls force.rebuild_neighbors exactly once per
      force evaluation and with the tensor passed to it — initial() at pos0 and
      step_nve() at the new end-of-step positions, never the step-start positions;
      with rebuild=False it is never called; ForceField.rebuild_neighbors delegates
      to neighbors.update(pos) only (a stub recording both calls sees zero
      rebuild() calls), the base ForceField stays a no-op, and a direct
      neighbors.rebuild(pos) still forces a build (rebuild_count += 1, ago reset).
    status: pending
  - id: ac-004
    summary: frozen-list integrators still compile fullgraph; live ones step eagerly
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_compile.py"
    pass_when: |
      The whole tests/test_molix/test_md/test_compile.py suite passes; a
      LangevinVerletIntegrator over LennardJonesCutForceField constructed with
      rebuild=False compiles both step and rollout under
      torch.compile(fullgraph=True, backend="aot_eager"), matches eager to 1e-12 in
      float64, and leaves the list's rebuild_count at 0; the same force field with
      rebuild=True advances eagerly through advance_n and raises rebuild_count
      above 0.
    status: pending
  - id: ac-005
    summary: skin-gated NVE drift stays within 3x of the rebuild-every-step baseline
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_driver.py -k drift"
    pass_when: |
      On the 64-atom cubic-lattice argon harness (eps=0.0103 eV/EV_PER_AMU_A2_FS2,
      sigma=2.5 A, cutoff=3.5 A, mass 39.95 amu, MaxwellBoltzmann seed=0 at 300 K,
      dt=4 fs, 100 steps, chunk=1, float64 CPU) driven through MD, the relative
      drift max_t |E_tot(t)-E_tot(0)|/|E_tot(0)| satisfies drift(skin=1.0) <= 3 *
      drift(skin=0.0), and drift(skin=0.0) > 0 (a nonzero baseline, so the ratio is
      not vacuous).
    status: pending
  - id: ac-006
    summary: driver-path rebuild counters and energies follow the skin as specified
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_driver.py -k observables"
    pass_when: |
      Over the same 100-step harness at skin in {0.0, 0.5, 1.0}: rebuild_count is
      non-increasing in skin with rebuild_count(1.0) < rebuild_count(0.0);
      rebuild_count(skin=0.0) == 100 exactly (one policy call per force evaluation;
      the entry evaluation at the build positions does not rebuild); ndanger == 0
      at skin 0.5 and 1.0 while ndanger(skin=0.0) == 100 (link 04's documented
      degenerate limit); the final MDState.energy of all three arms agree to
      atol=1e-10, rtol=0.
    status: pending
  - id: ac-007
    summary: benchmark and script consumers expose the list-owned policy flags
    type: code
    evaluator_hint: "python benchmarks/verify_md_ljcut_nve.py --help; ruff check scripts/"
    pass_when: |
      `python benchmarks/verify_md_ljcut_nve.py --help` and `python
      scripts/matpes_port/run_nve.py --help` each list --skin (with its Angstrom
      unit and default in the help text), --every, --delay and --no-check, and
      neither mentions --rebuild-every; `PYTHONPATH=src:. python
      benchmarks/verify_md_ljcut_nve.py --ps 0.4 --n 4 --device cpu --no-compile
      --no-save` prints the policy line (skin/every/delay/check), rebuild_count
      with its fraction of n_steps, ndanger, and `RESULT: PASS`; run_nve.py no
      longer prints an expected-rebuild heuristic and its --compile help no longer
      claims a "frozen" list; `ruff check scripts/ regressions/ && ruff format
      --check scripts/ regressions/` is clean.
    status: pending
  - id: ac-008
    summary: GH200 melt rerun conserves energy with few rebuilds and zero danger
    type: scientific
    evaluator_hint: "sbatch benchmarks/run_gh200_ljcut_nve.sbatch; read job stdout"
    pass_when: |
      `sbatch benchmarks/run_gh200_ljcut_nve.sbatch` at defaults (100 ps, N=500
      FCC argon, rho*=0.8442, T0*=1.44, r_c=2.5 sigma, dt=4 fs, n_steps=25000,
      skin=1.02 A, every=1, delay=0, check=True) prints, for BOTH stdout blocks
      (inductor fullgraph and the cuda-graphs preset): `RESULT: PASS` with rel
      energy drift < 1e-3, ndanger == 0, and rebuild_count <= n_steps / 5
      (expected ~n_steps/20-40). The job id and the four numbers (drift, rms,
      rebuild_count, steps/s) are quoted in the delivery summary.
    status: pending
  - id: ac-009
    summary: docs and CHANGELOG state the list-owned policy and the breaking removal
    type: docs
    pass_when: |
      docs/molix/user-guide/md.md contains no MD(rebuild_every=) or
      NeighborListHook; it documents that NeighborList(skin=, every=, delay=,
      check=) owns the cadence and Integrator.eval_force merely asks once per force
      evaluation at the evaluated positions, names skin=0/every=1/delay=0/check=True
      as the accurate no-skin limit and skin>0 as the production setting, carries a
      migration line for the removed kwarg plus the frozen-list route
      (MD(integrator=LangevinVerletIntegrator(ff, ..., rebuild=False))), and its
      pure-GPU example builds the list with a skin; CHANGELOG.md [Unreleased] gains
      a bullet recording both removals as breaking with the migration; the
      google-style docstrings of ForceField.rebuild_neighbors / rebuilds_neighbors,
      Integrator.rebuild and MD state units (A for skin, steps for every/delay),
      the semantic shift of rebuild_neighbors, the forced neighbors.rebuild(pos)
      route, and the per-force-eval host-sync cost.
    status: pending
  - id: ac-010
    summary: regression script reproduces the wiring and rebuild-accounting goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-neighborlist-skin-07-wire.py` prints OK
      and exits 0, reproducing its hard-coded literals on the 64-atom argon
      lattice: md.integrator.rebuild is True for the lj/cut force field while a
      rebuild=False twin ends 100 steps with rebuild_count == 0; skin=0.0 gives
      rebuild_count == 100 and ndanger == 100; skin=1.0 gives its recorded
      rebuild_count literal with 0 < rebuild_count < 100 and ndanger == 0; the two
      final total energies agree to 1e-10 and E_tot(0) matches its recorded float64
      literal to rtol=1e-9; MD(..., rebuild_every=1) raises TypeError and
      molix.md exposes no NeighborListHook. No third-party oracle is imported or
      subprocessed at runtime.
    status: pending
---

# Acceptance criteria

- **ac-001 — the point of the link.** Three competing owners of one decision is
  the defect; the grep is the cheapest proof that only the list is left. The
  exclusions matter: `CHANGELOG.md` must still name what was removed, and
  `src/molzoo/specs/mace_matpes.md` §7.4 is append-only run history that records
  what past runs actually used.
- **ac-002 / ac-003 — the seam.** ac-002 pins that the on/off is *derived* from a
  declared capability (no duck-reads, no driver poke) and survives both wrappers
  the production path uses; ac-003 pins the two things the deleted code got right
  and the deleted hook got wrong — once per force evaluation, at the positions
  being evaluated — plus the new semantics (`update`, not `rebuild`) and the
  escape hatch that keeps a forced build reachable.
- **ac-004 — the compiled-path invariant.** The reason the switch is a
  construction-time Python bool and not a counter or a tensor: with
  `rebuild=False` the branch is dead at trace time and the whole rollout still
  traces to one graph. If this regresses, the exported / CUDA-graph paths regress
  with it.
- **ac-005 / ac-006 — the physics and the anti-vacuity guard.** ac-005 is the
  falsifiable statement that moving the policy did not buy speed with energy: a
  wiring that lets pairs go missing injects `O(1)` force errors (the LJ force is
  discontinuous at `r_cut` even with the shifted energy) and blows the ratio.
  ac-006's `rebuild_count(skin=0.0) == 100` is deliberately an *equality*: every
  other assertion in this spec passes trivially if the policy is never called at
  all, and only this literal catches that. The `ndanger(skin=0.0) == 100` clause
  pins link 04's documented degenerate alarm through the driver, so the benchmark
  can safely gate on `ndanger == 0` for `skin > 0` only.
- **ac-007 / ac-008 — the consumers and the close-out.** ac-007 keeps the
  out-of-suite scripts honest (they are the only executable documentation of the
  new flags); ac-008 is the chain's scientific pay-off, and is read from benchmark
  stdout rather than a test because it needs a GH200 and a 100 ps trajectory. Its
  three numbers together say what the skin was for: conservation kept,
  `rebuild_count` collapsed, no rebuild ever overdue.
- **ac-009 / ac-010 — documentation and reproducibility.** A breaking removal is
  only complete when the migration is written down; the regression script is the
  oracle-free record that the wiring and the rebuild accounting behave as
  specified on any machine.
