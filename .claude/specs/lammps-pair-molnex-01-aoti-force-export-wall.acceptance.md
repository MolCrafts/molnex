---
spec: lammps-pair-molnex-01-aoti-force-export-wall
status: investigation-complete
---

# Acceptance — pair_style molnex AOTI force-export wall

This spec is **investigation-of-record + scope**, not a build. Acceptance is
that the wall is documented reproducibly and the route forward is scoped.

## Criteria

- [x] **a. Wall chain recorded.** The five sequential AOTI failures (1
  trace_autograd_ops, 2 requires_grad_ intermediate, 3 functorch switch, 4
  einsum-scalar Inductor crash, 5 cuet-no-exportable-backward → forces=0) are
  documented with symptom, cause, and action, each tied to a gate job id.

- [x] **b. Root cause pinned.** The hard wall is attributed to cuequivariance
  fused ops lacking an export-traceable backward, evidenced by energy bit-exact
  (`-128066.5225`) while AOTI forces are `0.0000` vs eager `≈9.56e5`.

- [x] **c. Not-our-bug established.** Eager forces finite+correct, energy graph
  exports bit-exact → forward port sound; zero arises at the autodiff∘cuet∘AOTI
  boundary.

- [x] **d. Route forward scoped.** TorchScript+libtorch `pair_style` (run
  `backward()` in C++) specified with the C++↔model tensor protocol incl.
  charge/spin pass-through, plus CSVR + per-species MSD plan.

- [x] **e. Code-touched decision.** `export.py` autograd-export enablement kept
  (correct, inert for energy-only); einsum monkeypatch confined to gate script,
  not upstreamed.

- [ ] **f. (route-forward, deferred)** TorchScript export round-trips energy
  bit-exact and C++ `backward()` forces match eager `-dE/dx` ≤1e-4 eV·Å.
