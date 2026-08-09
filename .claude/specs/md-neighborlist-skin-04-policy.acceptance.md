---
slug: md-neighborlist-skin-04-policy
criteria:
  - id: ac-001
    summary: build radius r_build = cutoff + skin sizes the list and its buffers
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      On the 64-atom simple-cubic fixture (spacing 3.0 A, 12 A cube),
      NeighborList(cutoff=3.5, skin=1.5) reports r_build == 5.0 and
      num_edges == 1152, while skin=0.0 reports num_edges == 384; capacity >=
      ceil(1.35 * 1152) and the buffers keep shapes (capacity, 2) / (capacity, 3).
    status: pending
  - id: ac-002
    summary: construction refuses r_build beyond the half-width and beyond the dead edge
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      cutoff=3.5 with skin=3.0 on the 12 A cube (min perpendicular half-width
      6.0 A) raises ValueError naming r_build even though cutoff alone passes;
      skin >= (DEAD_EDGE_CUTOFF_FACTOR - 1) * cutoff raises ValueError; skin < 0,
      every < 1 and delay < 0 each raise ValueError; every=4 with delay=10
      raises ValueError matching "multiple" (delay % every == 0, LAMMPS
      Neighbor::init parity) while every=4/delay=8 and any delay=0 construct
      cleanly.
    status: pending
  - id: ac-003
    summary: update() implements the conjunctive every/delay gate exactly
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      With every=4, delay=8 and a displacement far beyond skin/2, update()
      returns False for ago 1..7 and True first at ago == 8; every=4/delay=0
      first permits ago == 4; every=1/delay=10 first permits ago == 10; a forced
      rebuild() sets ago == 0 and re-phases the schedule from that build.
    status: pending
  - id: ac-004
    summary: half-skin criterion is strict > and degenerates correctly at skin=0
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      A max displacement of exactly skin/2 leaves update() False while
      skin/2 + 1e-12 returns True; with skin=0.0, every=1, delay=0, check=True an
      unchanged position returns False and any nonzero displacement returns True;
      with check=False update() rebuilds on every permitted opportunity regardless
      of displacement.
    status: pending
  - id: ac-005
    summary: ndanger counts builds fired at the first permitted opportunity
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      every=1/delay=0/skin=1.0 with a 1.0 A displacement per update gives
      ndanger == number of rebuilds; every=2/delay=6/skin=1.0 crossing the
      half-skin only after ago 6 rebuilds at ago 8 with ndanger == 0; the same
      configuration with the jump already present at the first permitted
      opportunity fires at ago == 6 == max(every, delay) and increments ndanger
      (the verbatim LAMMPS threshold, always reachable now that
      delay % every == 0 is enforced at construction).
    status: pending
  - id: ac-006
    summary: unwrapped-positions invariant fails loud inside the displacement check
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      Displacing one atom by a full 12 A cell vector between builds makes
      update() raise RuntimeError matching "unwrapped"; the same jump with
      check=False does not raise (documented consequence).
    status: pending
  - id: ac-007
    summary: live list is a superset of the exact cutoff pair set at every step
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      Over the 100-step float64 CPU NVE run (64-atom lattice, LJ eps=0.0103 eV
      /EV_PER_AMU_A2_FS2, sigma=2.5 A, cutoff=3.5 A, skin=0.5 A, dt=4 fs, mass
      39.95 amu, MaxwellBoltzmann seed=0 at 300 K, chunk=1), at every step the
      exact minimum-image O(N^2) pair set within 3.5 A is a subset of
      edge_index[:num_edges], and each such pair's list-reconstructed distance
      matches the reference to 1e-9 A.
    status: pending
  - id: ac-008
    summary: skinned+gated trajectory matches rebuild-every-step to 1e-10
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      Over the same 100-step run, per-step total energy and forces from
      skin=0.5/every=1/delay=0/check=True agree with skin=0.0 plus rebuild() on
      every force evaluation to atol=1e-10, rtol=0 in float64 on CPU.
    status: pending
  - id: ac-009
    summary: ndanger stays 0 and rebuild_count is non-increasing in skin
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListPolicy"
    pass_when: |
      The standard 100-step run at skin=0.5 ends with ndanger == 0; repeating it
      at skin in {0.0, 0.25, 0.5, 1.0} yields a non-increasing rebuild_count with
      rebuild_count(skin=1.0) < rebuild_count(skin=0.0).
    status: pending
  - id: ac-010
    summary: NeighborStrategy declares cutoff/skin/update; no getattr duck-read remains
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py tests/test_molix/test_md/test_forcefield.py"
    pass_when: |
      isinstance(NeighborList(...), NeighborStrategy) is True with the widened
      protocol; src/molix/md/forcefield.py contains no getattr(neighbors,
      "cutoff", ...) and LennardJonesCutForceField still raises ValueError when
      the requested cutoff exceeds neighbors.cutoff (not r_build).
    status: pending
  - id: ac-011
    summary: module docstring states the skin derivation, the trade, and the refs
    type: docs
    pass_when: |
      The src/molix/md/neighbors.py module docstring documents r_build = cutoff +
      skin, the half-skin criterion with its worst-case derivation, the raw
      (non-minimum-image) displacement test and the unwrapped-positions
      invariant, the ndanger meaning including the skin=0 caveat, the O(1)
      energy-injection cost of check=False / a coarse delay, and cites the LAMMPS
      neigh_modify docs, the Nordlund lecture-3 URL, and Verlet 1967
      (10.1103/PhysRev.159.98) / Quentrec & Brot 1973 (10.1016/0021-9991(73)90046-6)
      with the paywall caveat; new constructor args, update() and r_build carry
      google-style docstrings with units.
    status: pending
  - id: ac-012
    summary: regression script reproduces the hand-derived decision goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-neighborlist-skin-04-policy.py` prints
      OK and exits 0, reproducing its hard-coded literals: num_edges == 1152 at
      r_build 5.0 A and 384 edges within cutoff 3.5 A; scenario 1 (every=2,
      delay=4, 0.1 A/update) returns True exactly at updates {8, 16, 24, 32, 40}
      with rebuild_count == 5 and ndanger == 0; scenario 2 (every=1, delay=0,
      1.0 A/update) gives rebuild_count == 5 and ndanger == 5; scenario 3
      (every=5, check=False, no motion) gives rebuild_count == 4 and ndanger == 0.
      No third-party oracle is imported or subprocessed at runtime.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 — sizing and refusal.** The whole point of the skin is that
  the *build* radius exceeds the *interaction* radius; these pin that both the
  edge population and the fixed buffers follow `r_build`, and that the two ways
  `r_build` can silently break the list (crossing the minimum-image half-width,
  colliding with the dead-edge shift) are refused at construction rather than
  discovered as missing periodic images.
- **ac-003 / ac-004 / ac-005 — LAMMPS-exact policy.** The gate is conjunctive,
  the criterion is strict `>` on the raw half-skin displacement, `ago` resets at
  every build, and `ndanger` fires exactly on first-permitted-opportunity builds
  at the verbatim threshold `max(every, delay)` — always reachable because the
  constructor enforces `delay % every == 0`, mirroring LAMMPS `Neighbor::init`
  (operator decision at the 2026-08-09 audit grill; the generalised-threshold
  alternative was drafted and rejected).
- **ac-006 — the invariant.** Frozen shifts plus raw displacements are correct
  only on unwrapped positions; the guard converts a silent wrong-energy mode into
  a loud `RuntimeError`.
- **ac-007 — the primary falsification.** Everything else can pass while the list
  is quietly incomplete; only the per-step exact O(N²) subset check (plus the
  distance cross-check that catches a stale shift on a surviving index pair) can
  falsify the skin criterion. It is the criterion to look at first when this link
  regresses.
- **ac-008 / ac-009 — physics equivalence and gate sanity.** Equivalence to
  rebuild-every-step is the statement that the skin changed the *cost*, not the
  *PES*; the monotonicity and `ndanger == 0` checks catch an inverted or dead
  gate that equivalence alone would not (a gate that always rebuilds passes
  ac-008 trivially).
- **ac-010 — protocol.** Declared members are what let the force field stop
  guessing; the retained interaction-cutoff comparison is the anti-truncation
  guard and must not be relaxed to `r_build`.
- **ac-011 / ac-012 — documentation and reproducibility.** The docstring carries
  the derivation and the honest cost of `check=False`; the regression script is
  the oracle-free, integer-golden record of the rebuild decision sequence.
