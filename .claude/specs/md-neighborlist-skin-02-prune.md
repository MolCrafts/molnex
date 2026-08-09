---
title: Prune dead neighbour-graph homonyms (molix.nn.locality, molpot.graph)
status: approved
created: 2026-08-09
grilled: true
chain: md-neighborlist-skin
---

# Prune dead neighbour-graph homonyms (molix.nn.locality, molpot.graph)

## Summary

Two dead neighbour-graph implementations are removed from the tree so that the
name `NeighborList` is unambiguous before link 03 renames the MD-side class.
`molix.nn.locality.NeighborList` (an `nn.Module` wrapper over
`molix.F.locality.get_neighbor_pairs`) and `molpot.graph.radius_graph` (the same
kernel plus a cross-molecule mask) each have zero call sites in `src/`, `tests/`,
`benchmarks/`, `scripts/` and `docs/`; the only traces are one re-export, one
stale docstring cross-reference, and one unchecked line in a completed spec.
This change deletes both, drops the re-export, retargets the docstring at the
neighbour-list class that actually exists, and annotates the fossil spec line so
no future reader treats the removed package as a live option. Nothing else
changes: no new public symbol is introduced, no behaviour is altered, and the
existing suite must stay green on a deletion-only diff.

## Design

**Entities removed.**

- `molix.nn.locality.NeighborList` — `nn.Module` with `(cutoff, pbc,
  max_num_pairs)` and a `forward(positions, cell)` that forwards to
  `molix.F.locality.get_neighbor_pairs`. It is a thin one-call wrapper with no
  in-tree user, and it collides by name with two live classes
  (`molix.data.tasks.neighbor.NeighborList`, `molix.md.PeriodicNeighborList`).
  The whole module file goes; `src/molix/F/locality.py` — the actual kernel
  binding, used by `molix/data/tasks/neighbor.py` and `molix/md/neighbors.py` —
  is untouched.
- `molpot.graph.radius_graph` — free function returning `(edge_index, edge_vec)`
  with `pos_j - pos_i` sign convention and a `batch[i] == batch[j]` mask. Zero
  call sites; `src/molpot/graph/` has no `__init__.py` (implicit namespace
  package, never a declared surface); lines 29–40 are unresolved "we currently
  don't support / for now, let's assume / a simple but potentially slow way"
  reasoning left in shipped code. It also predates the current Edge Convention
  (`edge_diff = pos[target] - pos[source]`) that CLAUDE.md fixes, so keeping it
  would be keeping a second, wrong-signed graph builder. The file and the
  directory both go.

**Public-surface impact.** `molix.nn.__all__` shrinks from five names to four
(`BatchAggregation`, `KeyedMLP`, `KeyedMLPSpec`, `ScatterSum`). This is a **hard
removal**: `$META.stage` is `experimental`, and the repo norm is no deprecation
alias, no shim module, no `__getattr__` fallback. `molpot` never exported
`radius_graph` at package level, so its removal is invisible from the package
API. Both deletions are import-time-only changes — no runtime path in the
training loop, the MD driver, or any encoder resolves either symbol.

**`__all__` ordering (discovered rot, fixed here).** `src/molix/nn/__init__.py`
currently lists `["KeyedMLP", "KeyedMLPSpec", "NeighborList", "ScatterSum",
"BatchAggregation"]` — not sorted, in violation of
`.claude/notes/notes.md:243` ("`__all__` stays alphabetized"). The deletion
rewrites that literal anyway, so it is re-sorted in the same edit and pinned by
the new test. Without this the guard test would fail on its own landing commit.

**Docstring retarget.** `src/molpot/heads/charge_bond.py:82` documents
`full_neighbor_list=True` as matching ":class:`molix.nn.locality.NeighborList`'s
``symmetry=True`` default". That is doubly wrong: the referenced class is being
deleted, and it never had a `symmetry` parameter at all. The live owner of that
default is `molix.data.tasks.neighbor.NeighborList`, whose `symmetry: bool =
True` produces the bidirectional edge list `BondChargeHead` assumes. The
reference is retargeted there; the surrounding prose is unchanged.

**Fossil-spec annotation.** `.claude/specs/mace-omol-port-02-pipeline-integration.md`
is a completed spec. Its line ~81 is an *unchecked optional* item, "Edge sourcing
via `molpot.graph` / `NeighborList` for standalone use". Only that one line gains
a parenthetical marking it obsolete. Checked `[x]` rows, the Summary, the
Testing section and the `.acceptance.md` sibling are historical record and are
**not** edited.

**Reuse decision** (resolving everything the librarian pass surfaced; no new
symbol is designed by this spec, so every verdict is a *keep the incumbent*):

- `reuse` `molix.data.tasks.neighbor.NeighborList` — the live `SampleTask` that
  builds the pipeline's edge tensors with the documented `symmetry` semantics.
  It is the retarget destination for the `charge_bond.py` docstring; nothing is
  reimplemented.
- `reuse` `molix.F.locality.get_neighbor_pairs` — the kernel both deleted
  symbols wrapped. It survives untouched and remains the single binding.
- `reuse` `molix.md.PeriodicNeighborList` / `NeighborListHook` — the MD-side
  rebuilding neighbour list, the standalone edge-sourcing story that made
  `molpot.graph` redundant, and the subject of link 03's rename.
- `new — none.` This spec adds zero public symbols. The only file created under
  `src/`-adjacent trees is a test and a regression script.

**Why this touches two packages without tripping `large-spec-split`.** The rule
targets scope size; here the diff is deletion-only, adds no symbol, and stays
inside one capability family (the neighbour-graph name). Splitting the
`molpot/heads/charge_bond.py` docstring fix out of the `molix.nn` deletion would
land a commit whose own docstring points at a module it just deleted — knowingly
shipping rot, which the iron law outranks. The two-package touch is deliberate
and scoped to the dangling reference.

## Files to create or modify

- `src/molix/nn/locality.py` (deleted)
- `src/molix/nn/__init__.py`
- `src/molpot/graph/radius.py` (deleted, together with the now-empty
  `src/molpot/graph/` directory)
- `src/molpot/heads/charge_bond.py`
- `.claude/specs/mace-omol-port-02-pipeline-integration.md`
- `tests/test_molix/test_nn/test_init.py` (new)
- `regressions/md-neighborlist-skin-02-prune.py` (new)

## Tasks

- [ ] Write failing package-surface tests for `molix.nn` (`tests/test_molix/test_nn/test_init.py` → `TestNnExports`) asserting `__all__` is alphabetized, resolves, excludes `NeighborList`, and that `molix.nn.locality` is unimportable
- [ ] Delete `src/molix/nn/locality.py` and drop the `NeighborList` import + `__all__` entry from `src/molix/nn/__init__.py`, re-sorting `__all__` alphabetically per `notes.md:243`
- [ ] Delete `src/molpot/graph/radius.py` and the now-empty `src/molpot/graph/` directory (including any stale `__pycache__`)
- [ ] Retarget the docstring cross-reference at `src/molpot/heads/charge_bond.py:82` to `molix.data.tasks.neighbor.NeighborList`
- [ ] Annotate the obsolete optional item at `.claude/specs/mace-omol-port-02-pipeline-integration.md` line ~81 with the parenthetical, leaving every checked row untouched
- [ ] Add regression example `regressions/md-neighborlist-skin-02-prune.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

**Unit — `tests/test_molix/test_nn/test_init.py` (`TestNnExports`).** Mirrors
`src/molix/nn/__init__.py`; one assertion family per test method, following the
pattern already used by `tests/test_molzoo/test_imports.py`.

- *Happy path* — `molix.nn.__all__ == ["BatchAggregation", "KeyedMLP",
  "KeyedMLPSpec", "ScatterSum"]`, exactly, as a hard-coded literal (this pins
  both the removal and the alphabetization rule in one value).
- *Every exported name resolves* — `getattr(molix.nn, name)` succeeds for each
  entry in `__all__` (guards a re-sorted list that lost a real import).
- *No stray public attribute* — the set of public, non-module attributes of
  `molix.nn` equals `set(molix.nn.__all__)`. Sub-module attributes (`mlp`,
  `scatter`, bound as a side effect of `from .x import y`) are filtered with
  `inspect.ismodule`; do **not** write a bare `set(dir(molix.nn)) ==
  set(__all__)` — `molix.nn` defines no `__dir__`, so that form fails on the
  submodule names and would be a false red.
- *Edge case, re-introduction guard* — `"NeighborList" not in molix.nn.__all__`
  **and** `not hasattr(molix.nn, "NeighborList")`. The two are distinct
  failures: a future `from .locality import NeighborList` without an `__all__`
  edit trips only the second.
- *Edge case, module really gone* —
  `importlib.util.find_spec("molix.nn.locality") is None`.

No domain-validation tier: this spec touches no physics, produces no numbers.

**Regression example — `regressions/md-neighborlist-skin-02-prune.py`.** One
minimal public-API script under the repo-root `regressions/`, run as
`PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py`; prints
`OK` and exits 0, exits 1 on drift. Header comment records commit, torch
version, date, device per `regressions/README.md`. Two sections, all
expectations hard-coded literals; no oracle, no network, no subprocess, no
third-party import.

1. *Negative — the deleted surface stays deleted.* `import molix.nn` and
   `import molpot` both succeed; `not hasattr(molix.nn, "NeighborList")`;
   `"NeighborList" not in molix.nn.__all__`;
   `importlib.util.find_spec("molix.nn.locality") is None`;
   `importlib.util.find_spec("molpot.graph") is None` (a surviving namespace
   directory — e.g. a leftover `__pycache__` — makes this non-`None`, which is
   itself the drift being caught); `import molpot.graph.radius` raises
   `ModuleNotFoundError`; `not hasattr(molpot, "radius_graph")`.
2. *Positive control — the surviving neighbour list still works.* Construct
   `molix.data.tasks.neighbor.NeighborList(cutoff=1.5, symmetry=True)` and run
   `execute` on a hard-coded 3-atom chain at `x = 0.0, 1.0, 2.0` Å. The goldens
   are arithmetic on the page rather than a captured measurement: within a
   1.5 Å cutoff the half-pairs are `(0,1)` and `(1,2)` (the `(0,2)` separation
   is 2.0 Å, outside), so `symmetry=True` gives `E = 2 x 2 = 4` rows in
   `edge_index` `(4, 2)` and `edge_dist` equal to `1.0` in all four entries
   (compared with `torch.allclose`, atol `1e-6`). This is the claim that makes
   the deletion safe — the capability was never lost, only the duplicate
   wrapper.

**Suite.** The change is deletion-only, so the full `python -m pytest tests/ -v`
result must be unchanged apart from the new file's tests; in particular
`tests/test_molix/test_nn/` and `tests/test_molpot/` must be green, and
`build.check` (`ruff check` + `ruff format --check` + `ty check`) must pass over
`src/ tests/ scripts/ regressions/`.

## Out of scope

- **Blueprint refresh.** `.claude/notes/architecture.md:65` lists
  `src/molpot/graph/radius.py`. Updating the architecture blueprint is
  `/mol:note` / `/mol:map`'s job, not a task here; the line will be stale
  between this spec landing and the next map run. Flagged deliberately, not
  forgotten.
- **The `NeighborList` rename itself.** Freeing the name is this link's only
  purpose; `molix.md`'s rename is link 03 of the chain and must not be started
  here.
- **Skin / Verlet-list behaviour.** No change to `molix.md.PeriodicNeighborList`,
  its rebuild cadence, or `NeighborListHook` — those are later links.
- **`molix.F.locality`.** The kernel binding stays exactly as is; only the two
  redundant wrappers over it are removed.
- **Other historical rows of the mace-omol fossil spec** and its
  `.acceptance.md` sibling (which also mentions `molpot.graph` at line 10).
  Completed specs are a record; only the single unchecked optional item is
  annotated, and no `verified` criterion is rewritten.
- **Deprecation shims.** No alias, no `DeprecationWarning`, no re-export stub.
  Stage is `experimental`; hard removal is the repo norm.
- **Domain basis section.** Omitted deliberately: `$META.science.required` is
  true, but this spec declares no physics — it deletes unused code and edits
  prose, computing nothing.
