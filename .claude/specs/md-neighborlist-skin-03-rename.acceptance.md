---
slug: md-neighborlist-skin-03-rename
criteria:
  - id: ac-001
    summary: Full unit suite green with no source-side behaviour edits
    type: code
    pass_when: |
      `PYTHONPATH=src python -m pytest tests/` exits 0 with zero errors and zero
      new skips versus the pre-change baseline, and `git diff` over tests/ shows
      only identifier substitutions, the TestPeriodicNeighborList ->
      TestNeighborList class rename, and the two added guard tests — no changed
      assertion, tolerance, or expected literal in any existing test.
    status: pending
  - id: ac-002
    summary: No PeriodicNeighborList reference survives outside CHANGELOG.md
    type: code
    pass_when: |
      `grep -rn "PeriodicNeighborList" src/ tests/ benchmarks/ scripts/ docs/ regressions/`
      returns no lines (exit 1). Repo-wide, `grep -rn "PeriodicNeighborList" .`
      matches only CHANGELOG.md: the historical line 90 and the new rename entry
      added by this spec. src/molzoo/specs/*.md is verified to contain no
      occurrences of the old name at all.
    status: pending
  - id: ac-003
    summary: molix.md exports NeighborList and no longer exports the old name
    type: code
    pass_when: |
      `PYTHONPATH=src python -c "import molix.md as m; assert m.NeighborList;
      assert not hasattr(m, 'PeriodicNeighborList');
      assert 'PeriodicNeighborList' not in m.__all__;
      assert 'NeighborList' in m.__all__; assert m.__all__ == sorted(m.__all__)"`
      exits 0.
    status: pending
  - id: ac-004
    summary: Kernel-task import is aliased so the renamed class cannot shadow it
    type: code
    pass_when: |
      src/molix/md/neighbors.py contains
      `from molix.data.tasks.neighbor import NeighborList as NeighborListTask`
      and no bare `import NeighborList` from that module; the construction site
      (was line 119) reads `NeighborListTask(...)`; and the guard test asserting
      `molix.md.NeighborList is not molix.data.tasks.neighbor.NeighborList`
      while `molix.md.neighbors.NeighborListTask is
      molix.data.tasks.neighbor.NeighborList` passes.
    status: pending
  - id: ac-005
    summary: MD module docstring names the two same-name types and their layers
    type: docs
    pass_when: |
      The module docstring of src/molix/md/neighbors.py names both
      `molix.md.NeighborList` (stateful fixed-capacity MD buffer owner) and
      `molix.data.tasks.neighbor.NeighborList` (stateless pipeline SampleTask),
      states that the collision is deliberate, and explains the NeighborListTask
      alias.
    status: pending
  - id: ac-006
    summary: Docs and CHANGELOG updated without rewriting history
    type: docs
    pass_when: |
      docs/molix/user-guide/md.md and
      docs/molix/explanation/throughput-and-compilation.md use `NeighborList`
      everywhere the old name appeared (table rows, example import, example
      construction, static-shapes paragraph); CHANGELOG.md gains one new bullet
      under `## [Unreleased]` -> `### Changed` naming the rename and the
      no-back-compat-alias decision; `git diff CHANGELOG.md` shows the line-90
      entry unmodified.
    status: pending
  - id: ac-007
    summary: Regression example reproduces the hard-coded lattice goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-neighborlist-skin-03-rename.py`
      prints OK and exits 0, using only public API (`from molix.md import
      NeighborList`) and asserting, for the 3x3x3 simple-cubic lattice
      (spacing 3.0 A, cell 9.0 A, cutoff 3.5 A, float64, CPU,
      capacity_factor=1.35): num_edges == 162, capacity == 219,
      edge_index.shape == (219, 2), shifts.shape == (219, 3), dead-edge shift
      norm == 35.0 A, and after a rigid translation rebuild num_edges == 162 with
      unchanged shapes and rebuild_count == 1. No network call, subprocess, or
      third-party oracle at runtime; the header records capture command, commit
      sha, torch version, date, device/precision.
    status: pending
  - id: ac-008
    summary: build.check gate clean over src/tests/scripts/regressions
    type: code
    pass_when: |
      `ruff check src/ tests/ scripts/ regressions/` and
      `ruff format --check src/ tests/ scripts/ regressions/` both exit 0, and
      `ty check src/ --exit-zero-on-warning` reports no new errors versus the
      pre-change baseline.
    status: pending
---

# Acceptance criteria

**ac-001 — suite green, behaviour untouched.** The binding proof of neutrality is
that every pre-existing test passes with its assertions byte-identical. A test
requiring a changed expectation means the rename changed behaviour; that is a
hard stop, not a diff to accept.

**ac-002 — no residue.** The rename is a hard rename: no alias, no shim, no
lingering docstring. The only permitted survivals of the string
`PeriodicNeighborList` in the repository are inside `CHANGELOG.md` — the
immutable line 90 record and the new entry this spec adds, which necessarily
names the old symbol to describe the change. Note that `src/molzoo/specs/*.md`
was verified to contain **no** occurrences of the old name (those specs reference
`molix.data.tasks.NeighborList`, the untouched pipeline task), so they are not an
allowed-residue location — they are simply out of scope.

**ac-003 — public surface.** Covers the export, the removal, and the alphabetized
`__all__` rule captured in `.claude/notes/notes.md` (2026-08-09).

**ac-004 — the shadow guard.** The one way this link can silently break: leaving
`from molix.data.tasks.neighbor import NeighborList` unaliased so the class
definition rebinds the name and the constructor recurses into itself. Both the
static form and the runtime identity are asserted.

**ac-005 — the naming decision is written down.** Two same-name types in
different layers is a deliberate choice; undocumented, it reads as a bug and the
next contributor "fixes" the wrong side.

**ac-006 — history stays history.** New entry, not an edited one.

**ac-007 — regression example.** Public-API smoke that the renamed symbol is
importable and functional, with analytically derived goldens (6 in-cutoff
neighbours per site on the lattice; next shell at 4.243 A is outside 3.5 A) so no
external oracle is involved. A mismatch is a defect report, never a golden edit.

**ac-008 — repo gate.** `benchmarks/` is deliberately outside this gate per the
build-check-scope note; the benchmark edits are covered by ac-002's grep plus a
manual import check, and the gate is not widened in this link.
