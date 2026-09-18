---
slug: derivation-head-01-unify
status: active
criteria:
  - id: ac-001
    summary: one head, one grad call
    type: code
    pass_when: |
      `molpot.derivation.head.DerivativeHead(energy, *, virial=False)` is the
      only batch-level derivative type; `kernels.py`, `virial.py`,
      `geometry.py`, `derivative.py`, `modes/` and the session helpers in
      `protocol.py` are gone; `torch.autograd.grad` appears under
      `src/molpot/derivation/` only in `force.py` (and the Hessian rows in
      `hessian.py`); no `torch.func.grad` force path remains in `molpot`.
    status: pending
  - id: ac-002
    summary: forces and virial are right, and the frame comes back as handed in
    type: code
    pass_when: |
      `tests/test_molpot/test_derivation/test_head.py::TestDerivativeHead`
      pins the analytic and central-difference forces, the hard-coded and
      central-strain virial on a periodic pair with explicit shifts, the
      exact restoration of `atoms.pos` / `graphs.cell` / `edges.*` objects,
      the detach policy, and the create_graph=True force loss reaching
      parameters. All pass on the compute node.
    status: pending
  - id: ac-003
    summary: the head is the compile boundary
    type: performance
    pass_when: |
      Slow explain tests: inference `Compiler.explain(head, frame)` is
      1 graph / 0 breaks for virial=False and virial=True; the training-mode
      trace has graph_break_count <= 1 with the break at the leaf in
      `force.py`; `MacePotential.forward` traces to one graph in inference.
    status: pending
  - id: ac-004
    summary: every call site goes through the head
    type: code
    pass_when: |
      `MacePotential`, `PiNetPotential` (no `method` kwarg), `ForceReadout`,
      and `Potentials.calc_forces` / `calc_energy_forces` (mlip branch) /
      `calc_energy_forces_virial` route through a `DerivativeHead`; the
      mirrored tests of those modules and `tests/test_molix/test_md/test_driver.py`
      pass on the compute node.
    status: pending
---
