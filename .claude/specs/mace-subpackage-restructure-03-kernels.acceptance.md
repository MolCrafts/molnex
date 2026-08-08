---
slug: mace-subpackage-restructure-03-kernels
criteria:
  - id: ac-001
    summary: kernels.py exposes the two batch-level force passes with the agreed signatures
    type: code
    pass_when: |
      src/molpot/derivation/kernels.py defines module-level
      grad_force_pass(energy_core, batch, *, create_graph=None, detach_energy=None)
      and func_force_pass(energy_core, batch), both returning the same TensorDict
      they were given; both are importable as
      `from molpot.derivation import grad_force_pass, func_force_pass`
      and listed in molpot/derivation/__init__.py __all__.
    status: pending
  - id: ac-002
    summary: kernels compose the existing force primitives instead of re-deriving
    type: code
    pass_when: |
      grad_force_pass calls molpot.derivation.force.autograd_forces_from_energy and
      func_force_pass calls molpot.derivation.force.functorch_forces_with_aux;
      kernels.py contains no direct `torch.autograd.grad(` or `torch.func.grad(`
      call, and reuses protocol.{call_energy, absorb_model_output, has_energy,
      write_forces} rather than inlining equivalents.
    status: pending
  - id: ac-003
    summary: no duplicated force-pass body remains at any of the three call sites
    type: code
    pass_when: |
      `rg -n "torch\.autograd\.grad\(|torch\.func\.grad\(" src/molpot/derivation
      src/molzoo/pinet` matches only src/molpot/derivation/force.py; modes/grad.py,
      modes/func.py and molzoo/pinet/potential.py each reach the force pass solely
      through grad_force_pass / func_force_pass.
    status: pending
  - id: ac-004
    summary: kernels.py imports nothing from molzoo or molix.md
    type: code
    pass_when: |
      src/molpot/derivation/kernels.py imports only from torch, tensordict,
      molpot.derivation.force and molpot.derivation.protocol; it does not read or
      write molpot.derivation.protocol._SESSIONS.
    status: pending
  - id: ac-005
    summary: kernel unit tests cover both passes incl. detach_energy tri-state
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molpot/test_derivation/test_kernels.py -v` passes
      with classes TestGradForcePass and TestFuncForcePass, including tests for
      energy_core=None (model n_forward == 0), missing graphs.energy (RuntimeError
      mentioning graphs.energy), detach_energy in {False, True, None} with both
      leaf-owned and caller-supplied-leaf inputs, create_graph True/False, and the
      grad-vs-func agreement at atol=1e-12 in float64.
    status: pending
  - id: ac-006
    summary: forces equal minus the finite-difference energy gradient
    type: scientific
    pass_when: |
      In float64, both kernels' forces on the toy potential and on a small
      PiNetPotential match central finite differences of graphs.energy with
      h = 1e-5 Ang to max|dF| < 1e-6 eV/Ang.
    status: pending
  - id: ac-007
    summary: existing derivation and PiNet suites stay green after rebinding
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molpot/test_derivation tests/test_molzoo/test_pinet -v`
      passes with no test modified, including
      TestGradMode::test_energy_then_force_is_single_forward (n_forward == 1),
      TestFuncMode::test_energy_then_force_is_single_forward (lazy: n_forward == 0
      after EnergyReadout) and test_func_forces_match_autograd_reference.
    status: pending
  - id: ac-008
    summary: PiNet fullgraph compile contract survives the rebinding
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molix/test_md/test_compile.py -v` passes,
      including test_pinet_step_fullgraph_compiles_and_matches_eager and
      test_pinet_rollout_fullgraph_nve for both float32 and float64 — these run
      torch.compile(..., fullgraph=True, backend="aot_eager"), so any graph break
      introduced by the kernel indirection raises instead of falling back.
    status: pending
  - id: ac-009
    summary: regression example reproduces the pre-rebinding PiNet goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-03-kernels.py` exits 0 and
      prints OK: PiNetPotential (seed 0, float64, 4 atoms / 8 edges) with
      method="func" and method="grad" reproduces the hard-coded golden
      graphs.energy (eV) and atoms.forces (eV/Ang) literals within atol=1e-12,
      rtol=0. The literals were captured from commit 82c3091 (pre-rebinding) and
      the script imports no third-party oracle (no ASE / e3nn / mace-torch) at
      runtime.
    status: pending
  - id: ac-010
    summary: kernels documented with units, shapes and the detach_energy contract
    type: docs
    pass_when: |
      Every public callable in src/molpot/derivation/kernels.py has a Google-style
      docstring with tensor shapes and units (pos Ang, energy eV, forces eV/Ang),
      documents the detach_energy tri-state (False / True / None = detach iff the
      call created the position leaf) and the fullgraph constraints; and
      docs/molpot/user-guide/gradients.md gains a section separating the
      tensor-level force.py API from the batch-level kernels.py passes.
      `ruff check src/ && ruff format --check src/` is clean.
    status: pending
---
