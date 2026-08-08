---
slug: mace-subpackage-restructure-04-potential
criteria:
  - id: ac-001
    summary: one MACEPotential covers both foundation variants, spec-constructed
    type: code
    pass_when: |
      src/molzoo/mace/potential.py defines class MACEPotential(nn.Module) whose
      __init__ takes a MACEMatpesSpec or MACEOMolSpec (plus compute_forces /
      use_fallback keywords only), and `from molzoo.mace import MACEPotential`
      succeeds. No second variant-specific potential class is added.
    status: pending
  - id: ac-002
    summary: pipeline is monomorphic — no compute_forces on the public surface
    type: code
    pass_when: |
      inspect.signature() of MACEPotential.forward, .energy_core and
      ._write_energy contains no `compute_forces` parameter; the only occurrence
      of `compute_forces` in src/molzoo/mace/potential.py is the __init__
      signature/docstring plus the single __init__ branch that binds
      self._pipeline; MACEPotential.forward's body is one dispatch statement.
    status: pending
  - id: ac-003
    summary: both seams exist — public flat energy_core and _write_energy hook
    type: code
    pass_when: |
      MACEPotential.energy_core is public (no leading underscore) with signature
      (positions, Z, edge_index, batch, num_graphs, shifts=None, ...) -> Tensor
      of shape (B,), takes no TensorDict and makes no .item() call; and
      molpot.derivation.protocol.call_energy(MACEPotential_instance, batch)
      dispatches to MACEPotential._write_energy (returns a batch carrying
      graphs.energy and no atoms.forces).
    status: pending
  - id: ac-004
    summary: no hand-rolled force pass — 03 kernels + protocol write-back only
    type: code
    pass_when: |
      src/molzoo/mace/potential.py contains no `torch.autograd.grad`,
      `torch.func.grad` or `.backward(` call; forces come from the 03 shared
      force kernel (invoked with detach_energy driven by needs_leaf) and the
      write-back uses molpot.derivation.protocol.write_energy / write_forces.
    status: pending
  - id: ac-005
    summary: forcefield contract preserved incl. graphs batch_size=[num_graphs]
    type: runtime
    pass_when: |
      pytest tests/test_molzoo/test_mace/test_potential.py passes, including:
      forward on a batch WITHOUT a graphs namespace yields
      td["graphs"].batch_size == (B,) (never []); graphs.energy is (B,) and
      atoms.forces is (N,3); edges.shifts is consumed when present; edge_index
      stays (E,2) with no transpose; a grad-leaf pos leaves graphs.energy
      attached (requires_grad True) while a non-leaf pos yields a detached
      energy; an energy-only instance writes no atoms.forces.
    status: pending
  - id: ac-006
    summary: energy_core traces fullgraph with zero dynamo breaks
    type: runtime
    evaluator_hint: "marker: slow"
    pass_when: |
      torch._dynamo.explain(potential.energy_core)(positions, Z, edge_index,
      batch, num_graphs, shifts) on the CPU-sized fp64 test model reports
      graph_count == 1 and graph_break_count == 0.
    status: pending
  - id: ac-007
    summary: state_dict key set and shapes identical to the flat variants
    type: code
    pass_when: |
      For both variants, {k: v.shape for k, v in MACEPotential(spec).state_dict()
      .items()} equals the same mapping from the flat MACEMatpes / MACEOMol built
      with the matching hyper-parameters, so flat_model.state_dict() loads into
      MACEPotential via load_state_dict(strict=True) with empty missing and
      unexpected key lists.
    status: pending
  - id: ac-008
    summary: energy and forces match the flat variants under state_dict transfer
    type: scientific
    pass_when: |
      On CPU/fp64, seed 0, use_fallback=True, for BOTH MACEMatpes-shaped and
      MACEOMol-shaped (charge/spin conditioned) small models after direct
      load_state_dict: max|dE| <= 1e-12 eV and max|dF| <= 1e-12 eV/Ang between
      MACEPotential and the flat module on the same batch (with and without
      edges.shifts).
    status: pending
  - id: ac-009
    summary: physical invariants hold for the merged potential
    type: scientific
    pass_when: |
      On the CPU/fp64 test system: |sum_i F_i| <= 1e-10 eV/Ang; rigid translation
      by 1.0 Ang changes the energy by <= 1e-12 eV; central-difference forces
      (h = 1e-4 Ang) match the analytic forces to max|dF| <= 1e-6 eV/Ang.
    status: pending
  - id: ac-010
    summary: regression example reproduces its embedded goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-04-potential.py` exits 0
      using only the public API (MACEMatpesSpec -> MACEPotential -> forward /
      energy_core), reproducing the fp64 energy and per-atom force literals
      embedded in the file to <= 1e-12 (eV, eV/Ang) and asserting sum F = 0; the
      file imports no third-party oracle and records how the goldens were
      generated (tool, commit, date).
    status: pending
  - id: ac-011
    summary: test mirror is a package with no shadowing module
    type: code
    pass_when: |
      tests/test_molzoo/test_mace/ is a package containing test_potential.py with
      class TestMACEPotential, and tests/test_molzoo/test_mace.py no longer
      exists (no module/package name collision hiding collected tests).
    status: pending
  - id: ac-012
    summary: both seams and the use_fallback default are documented
    type: docs
    pass_when: |
      MACEPotential's class and method docstrings are Google style with tensor
      shapes and units (eV, eV/Ang), explicitly distinguish energy_core (flat
      compile target) from _write_energy (protocol hook), state why the graphs
      namespace is built with batch_size=[num_graphs] instead of
      protocol.ensure_graphs, justify use_fallback=False (35.7x note), and cite
      arXiv:2206.07697 plus the variant papers.
    status: pending
---
