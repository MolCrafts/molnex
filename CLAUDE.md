---
mol_project:
  name: molnex
  language: python
  build:
    install: "pip install -e '.[dev]'"
    check: "ruff check src/ && ruff format --check src/"
    test: "python -m pytest tests/ -v"
    test_single: "python -m pytest {path} -v"
    coverage: "python -m pytest tests/ --cov=src --cov-report=term-missing"
  arch:
    style: package-tree
    rules_section: "## Architecture"
  doc:
    style: google
  science:
    required: true
  notes_path: .claude/notes/notes.md
  specs_path: .claude/specs/
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MolNex** (v0.1.0) is a dict-first molecular ML framework for unified modeling of molecular potentials and properties with physics-aware ML. It is composed of four packages:

| Package | Role | Key Patterns |
|---------|------|-------------|
| **molix** | Training infrastructure | Trainer, TrainState (dict), Step protocol, Hook lifecycle |
| **molrep** | Representation learning | Embedding → Interaction → Readout pipeline, equivariance via cuEquivariance |
| **molpot** | Potential functions | BasePotential (nn.Module + ABC), functorch forces, PotentialComposer |
| **molzoo** | Pre-built encoders | Encoder-only (MACE, Allegro, PiNet, Sonata), no readout — downstream uses molpot |

## Build & Development

```bash
# Install (editable, with C++ extensions via scikit-build-core + CMake >=4.0)
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_molzoo/test_mace.py -v

# Run single test
python -m pytest tests/test_molzoo/test_mace.py::test_mace_forward -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Python >=3.10 required. Requires `torch>=2.10` (always use latest stable PyTorch).

## Architecture

### Two-tier data contract (raw sample vs collated batch)

The pipeline speaks **two** shapes. They are intentionally different — get
this wrong and `KeyError(('edges','edge_index'))` is the first thing you see.

| Stage | Container | Example access |
|---|---|---|
| `DataSource[i]`, pipeline `SampleTask` / `DatasetTask` I/O, `MmapDataset[i]` | **flat `dict`** | `sample["Z"]`, `sample["edge_index"]`, `sample["targets"]["U0"]` |
| Post-`collate_molecules` (what encoders / losses receive) | **nested `TensorDict`** (plain, no subclass) | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]`, `batch["graphs", "energy"]` |

The single conversion point is `molix.data.collate.collate_molecules`
(invoked by `DataModule._CollateFn`). Stats / diagnostics operating on
individual cached samples use **flat** keys; model / loss / metric code
operating on batches uses **nested tuple-keys**.

#### Post-collate batch schema

```
TensorDict (batch_size=[])
├── "atoms": TensorDict (batch_size=[N])
│   ├── Z: atomic numbers (N,)
│   ├── pos: positions (N, 3)
│   └── batch: graph membership (N,)
├── "edges": TensorDict (batch_size=[E])
│   ├── edge_index: source-target pairs (E, 2)   # [:,0]=source, [:,1]=target
│   ├── edge_diff: edge vectors (E, 3)            # pos[target] - pos[source]
│   └── edge_dist: edge distances (E,)
└── "graphs": TensorDict (batch_size=[B])  [optional]
    ├── num_atoms: (B,)
    └── <targets>
```

Access: `batch["atoms", "Z"]`, `batch["edges", "edge_dist"]`. Encoders
mutate the batch in place, writing `node_features` under `atoms` and
`edge_features` under `edges` — no subclass swap.

### Cache format

The on-disk cache is `molix.data.cache.PackedCache` — **one** `.pt` file
per cache with concatenated per-atom / per-edge / per-graph tensors and
`atom_ptr` / `edge_ptr` cumsum pointers. Loaded via `torch.load(mmap=True)`
for zero-copy reads; each DataLoader worker shares the OS page cache.

**Do not migrate to `TensorDict.memmap_()`.**
`LazyStackedTensorDict.memmap_(prefix)` writes one subdirectory per
sample (~1M files + ~260k dirs for QM9 130k molecules) — blows past HPC
shared-fs inode budgets and destroys `rsync` / `tar` / `rm` performance.
`PackedCache`'s single-file packed-bucket layout is strictly better for
fixed-schema molecular datasets. See `src/molix/data/cache.py` module
docstring for full rationale.

### Edge Convention (MUST follow everywhere)

```
edge_index[:, 0]  — source atom  (the "centre" in Allegro; the "sender" in MACE ConvTP)
edge_index[:, 1]  — target atom  (the "neighbour" in Allegro; the "receiver" in MACE ConvTP)
edge_diff         — pos[target] - pos[source]   (displacement vector, source → target)
edge_dist         — ‖edge_diff‖
```

`NeighborList` defaults to **full bidirectional** edges (`symmetry=True`, `E = 2 × n_pairs`).
Pass `symmetry=False` to get only the upper-triangle half-pairs (`E = n_pairs`) when you explicitly
want to exploit Newton's-3rd-law symmetry.  The two modes produce different `task_id`s so pipeline
caches are kept separate.

**Why edge_diff = pos[target] − pos[source]?**  This makes the displacement vector point in the
same direction as the edge (source → target), which is the convention expected by `SphericalHarmonics`
and all `cuEquivariance`-based tensor products in this repo.  The C++ `getNeighborPairs` kernel
returns `pos[rows] − pos[cols]` (opposite sign); `NeighborList.execute` negates it.

### `edge_index` (cutoff graph) vs `bond_index` (covalent topology)

Two **different** connectivity objects, with **deliberately different shapes**
so one can never be silently used as the other:

| field | shape | meaning | offset axis |
|---|---|---|---|
| `edge_index` | `(E, 2)` | geometric neighbour graph `G(r_c)` — recomputed from positions each step, bidirectional, includes through-space contacts | rows |
| `bond_index` | `(2, N)` COO | fixed covalent topology (from molpy `atomi`/`atomj`), paired with `bond_types`; consumed by `BondHarmonic` and other bonded terms | columns |

A within-cutoff neighbour is **not** a chemical bond. `BondHarmonic` rejects a
`(E, 2)` tensor passed as `bond_index` (raises, rather than computing a wrong
energy) — the transpose is a type-level anti-alias guard. The collated batch
exposes bonds under a `"bonds"` namespace (`batch["bonds", "bond_index"]`,
`batch["bonds", "bond_types"]`, `batch_size=[]`).

### torch_geometric collate/Batch parity

molnex's batch is a plain nested `TensorDict`, not a PyG `Data`/`Batch`. The
collate intentionally implements only the lazy subset of PyG's batching:

| PyG feature | molnex | why |
|---|---|---|
| `__inc__` / `__cat_dim__` (per-key offset on batching) | **implemented** as the declarative `INDEX_KEYS` registry + `rebase()` in `collate.py` | the one piece worth having — also fixes the latent un-offset `bond_index` bug |
| `follow_batch` (per-attribute batch vectors) | **skip** | no consumer |
| `ptr` (cumsum boundaries) | **skip** — derive on demand from `graphs.num_atoms` / `atom_ptr` | redundant to materialise |
| `to_data_list()` / unbatch | **skip** | `PackedCache` covers persistence/unpack |
| `exclude_keys` | **skip** | no consumer |
| arbitrary `HeteroData` node/edge types | **skip** | fixed `atoms`/`edges`/`graphs`/`bonds` namespaces are the contract |

The packed-mmap collate fast path (`collate_packed`) is a deliberate
divergence from PyG's per-sample Python `Collater` — faster and HPC-fs
friendly for fixed-schema molecular data.

### Module Dependency Graph

```
molix.config (global dtype singleton)
    ↓
molrep.embedding → molrep.interaction → molrep.readout
    │  └── molrep.heads (ScalarHead, …)        │
    │            ↑ (molpot.heads → molrep.embedding)
    ↓            │                              ↓
molzoo (MACE, Allegro, PiNet encoders) ──→ molpot.heads
    ↓                                          ↓
molpot.composition (PotentialComposer,    molpot.potentials
    Sonata, build_sonata)                      ↓
    ↓                              molpot.derivation (EnergyAggregation,
molix.core (Trainer, TrainState, Step, Hook)    ForceDerivation)
    ↓
molix.data (Dataset, collate, preprocess)
molix.datasets (QM9, RevMD17, ThreeBPA, WaterLES, MolRec)
molix.md (Langevin velocity-Verlet) ─→ molix.quant (T_eff / quantization scalars)
molix.export (AOT Inductor) · molix.compile (torch.compile / CUDA-graph capture)   [leaf execution utils]
```

Notes on cross-package edges (verified against imports):
- `molzoo` consumes `molrep.readout`/`molrep.interaction` and `molpot.derivation`
  (PiNet's energy/force head is co-located with the encoder by design).
- `molpot.heads` imports `molrep.embedding` (e.g. `heads/edge.py`) — the arrow
  runs heads→embedding, **not** readout→heads.
- `molrep.heads` (`ScalarHead`, …) is a distinct sub-tree from `molpot.heads`.
- `molix.quant` holds the T_eff / quantization-as-thermal-noise scalars
  (quantization infra in the `molix` base layer, operating on generic
  `nn.Module` / `state_dict`). The thermal-noise *study* layer (trajectory
  diagnostics, active learning, colored-noise thermostats) was migrated out to
  the `pinet-quant` project (`csmd` package); molnex keeps only the reusable MD
  engine (`molix.md`) and `molix.quant`.
- `molix.md` (in-process Langevin velocity-Verlet driver) produces paired
  trajectories; `molix.export` (AOT Inductor) and `molix.compile`
  (`torch.compile` / CUDA-graph capture) are leaf execution utilities that wrap a
  trained model — nothing in the core training loop imports them.

### State namespace contract

`TrainState` (`src/molix/core/state.py`) is a dict with a **fixed
top-level layout**. Every metric / scalar produced during training lives
inside one of four **namespace sub-dicts**, never at the top level with
a slash-prefix key.

| top-level key         | owner                    | kind        |
|-----------------------|--------------------------|-------------|
| `epoch`               | `Trainer`                | counter     |
| `global_step`         | `Trainer`                | counter     |
| `stage`               | `Trainer` (`Stage` enum) | counter     |
| `steps_since_last_eval` | `Trainer`              | counter     |
| `best_metric`         | `CheckpointHook`         | scalar      |
| `train`               | train-phase hooks        | **dict**    |
| `eval`                | eval-phase hooks         | **dict**    |
| `performance`         | throughput/timing hooks  | **dict**    |
| `gpu`                 | GPU telemetry hooks      | **dict**    |

**Writes — always nest.** Hooks must write into the namespace
sub-dict, never into the flat root with a slash:

```python
# ✅ correct
state["train"]["loss"] = loss.item()
state["eval"]["MAE"] = mae
state["performance"]["step_per_second"] = rate
state["gpu"]["peak_gib"] = peak

# ❌ rejected at __setitem__ — raises ValueError
state["train/loss"] = loss.item()
state[("train", "loss")] = loss.item()
```

This prevents two hooks from silently sharing a flat namespace and
colliding on keys (which is exactly how the train/MAE-exploded-to-203
bug happened: eval path wrote into an accumulator shared with the
train path).

**Reads — three equivalent forms, no helper required.**
:class:`TrainState` walks its nesting on read. All three of these
return the same value (or ``None`` / a default if any segment misses):

```python
state["eval"]["MAE"]            # nested dict access
state["eval/MAE"]               # slash-string path
state[("eval", "MAE")]          # tuple path

state.get("eval/MAE")           # all four: same fallback semantics
state.get(("eval", "MAE"))
state.get("epoch")              # plain top-level key still works
"eval/MAE" in state             # walk-aware __contains__
```

For arbitrary ``Mapping`` instances (e.g. plain ``dict`` snapshots),
:func:`molix.core.state.resolve` accepts the same shapes:

```python
from molix.core.state import resolve, display

resolve(state, ("train", "loss"))     # tuple path
resolve(state, "train/loss")          # slash string
resolve(state, "epoch", default=0)    # flat key with default
display(("train", "loss"))            # → "train/loss"  (for column headers)
```

The slash-string form is fine for *reads* — Log column headers,
TensorBoard tags, and external scripts addressing scalars all benefit
from a single string identifier. Only the *write* side stays
nested-only, which is what actually prevents namespace collisions.

**Phase-ownership rules for hooks.**

1. `on_train_batch_end`, `on_after_backward` — may write to
   `state["train"]`, `state["performance"]`, `state["gpu"]`.
2. `on_eval_batch_end`, `on_eval_step_complete` — may write to
   `state["eval"]`.
3. `on_epoch_start` / `on_epoch_end` — may write anywhere they own
   by phase, but should **prefer the per-phase callbacks** so readers
   have deterministic ordering.
4. **Hooks must not share mutable buffers between train and val.**
   If a hook needs the same metric on both sides (e.g. `MetricsHook`
   with `MAE`), it must hold two distinct instances — typically via
   `copy.deepcopy` in `__init__` — so a `.reset()` / `.update()` on
   the val side is invisible to the train side.

**Trainer guarantees for eval-phase hooks.**
`_run_eval_phase` fires `on_eval_step_complete` at the end of **every**
eval phase (both step-based `eval_every_n_steps` and epoch-end),
before the LR scheduler reads `best_metric_name`. Eval-publishing
hooks (`MetricsHook`, `TensorBoardHook`) should write their
`state["eval"]` scalars from `on_eval_step_complete`, not
`on_epoch_end`, to be available for the scheduler's read.

### Key Design Patterns

- **Encoder-only molzoo**: Encoders take a batch `TensorDict` and write per-layer features `(N, layers, features)` under `atoms.node_features`; readout/potentials handled by molpot
- **Pydantic configs**: All block configs use `BaseModel` with `ConfigDict(arbitrary_types_allowed=True)`
- **cuEquivariance**: Tensor products use `cuequivariance` / `cuequivariance_torch` for GPU-accelerated equivariant operations
- **Functorch forces**: `BasePotential.calc_forces()` computes `F = -dE/dx` via `torch.func.grad` (energy as a pure function of positions; compile-friendly, no double-backward)
- **Functional composition**: `PotentialComposer` chains pooling → parameter heads → potential terms → aggregation
- **Hook protocol**: Lifecycle callbacks (`on_train_start`, `on_epoch_end`, etc.) via `Hook` protocol
- **Step protocol**: `DefaultTrainStep` / `DefaultEvalStep` wrap forward → loss → backward → optimizer

### TensorDict Contract

The post-collate batch is a **plain `tensordict.TensorDict`** — no subclass.
The `AtomData / EdgeData / GraphData / GraphBatch` subclasses that existed
before 2026-05-18 have been removed. The `atoms / edges / graphs` namespace
structure and per-namespace `batch_size` are the only schema contract; they
are documented in the table above, not enforced by Python types.

**Rules for new code:**
- Use plain `TensorDict(..., batch_size=[N])`, never subclass it for batch data.
- Encoders: prefer inheriting `TensorDictModuleBase` for `in_keys`/`out_keys`
  validation. Plain `nn.Module` is accepted as long as `forward(td: TensorDict)
  -> TensorDict` is respected.
- Do not introduce `@tensorclass` dataclass wrappers for batch containers.

### Adding New Components

**New encoder** (molzoo): Implement `forward(td: TensorDict) -> TensorDict`, reading `td["atoms", "Z"]` / `td["edges", ...]` and writing per-layer features `(N, layers, features)` back under `atoms.node_features`. Use `molrep` building blocks. Add paper reference.

**New potential** (molpot): Inherit `BasePotential`, implement `forward() -> scalar energy Tensor`. Forces come from functorch (`torch.func.grad`) automatically — `forward` must read positions via `pos=`/`_get_positions` so the energy can be differentiated as a function of positions.

**New embedding/interaction** (molrep): Pure `nn.Module`, use `cuequivariance` for equivariant layers.

## Scientific Correctness Requirements

Every implementation of a physical model, potential, or operator MUST:
1. Reference the original publication (arXiv/DOI) in the module docstring
2. Match the equations in the paper — document any deviations with rationale
3. Include numerical validation tests against reference implementations or published values
4. Preserve physical symmetries (energy conservation, rotational/translational invariance, permutation equivariance)

## PyTorch Version Policy

This project tracks **latest stable PyTorch** (currently >=2.10). Use modern PyTorch APIs:
- `torch.compile` for performance-critical paths
- `torch.export` for model serialization
- `torch.nn.functional` over deprecated module alternatives
- Native `torch.nested` for variable-length sequences where appropriate

## Docstring Convention

Google-style docstrings with tensor shape annotations:

```python
def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Message passing step.

    Args:
        node_feats: Node features ``(n_nodes, hidden_dim)``.
        edge_index: Edge indices ``(n_edges, 2)``.

    Returns:
        Updated node features ``(n_nodes, hidden_dim)``.

    Reference:
        Author et al. "Paper Title" Venue Year
        https://arxiv.org/abs/XXXX.XXXXX
    """
```

## Generic workflow — use the `mol` plugin

The workflow verbs (plan → build → verify → document → ship) live in the
[`mol` plugin](../claude-plugin/). Install once from any cwd:

```
/plugin marketplace add /Users/roykid/work/molcrafts/claude-plugin
/plugin install mol@molcrafts
```

From the MolNex root, the `mol_project:` frontmatter at the top of this
file tells every skill which commands to run. Use:

| Was (deprecated)   | Use instead        | Notes                                   |
|--------------------|--------------------|-----------------------------------------|
| `/molnex-impl`     | `/mol:impl`        | Full scope → litrev → spec → TDD → impl |
| `/molnex-spec`     | `/mol:spec`        | Natural-language → `.claude/specs/`     |
| `/molnex-litrev`   | `/mol:litrev`      | Paper + reference-impl credibility      |
| `/molnex-arch`     | `/mol:arch`        | Reads `arch.rules_section` (this file)  |
| `/molnex-review`   | `/mol:review`      | Fans out architect ∥ optimizer ∥ documenter ∥ undergrad ∥ scientist |
| `/molnex-test`     | `/mol:test`        | Runs `build.test`                       |
| `/molnex-perf`     | `/mol:perf`        | Uses `perf.focus: pytorch`              |
| `/molnex-docs`     | `/mol:docs`        | Uses `doc.style: google`                |
| —                  | `/mol:note`        | Capture into `.claude/NOTES.md`         |
| —                  | `/mol:fix`         | Minimal-diff bug fix                    |
| —                  | `/mol:debug`       | Diagnose-only — never writes code       |
| —                  | `/mol:refactor`    | Restructure preserving invariants       |
| —                  | `/mol:ship commit` | Pre-commit gate (format + lint)         |
| —                  | `/mol:ship push`   | Pre-push gate (+ full test suite)       |
| —                  | `/mol:ship merge`  | Pre-merge CI-parity gate                |

MolNex-specific agents remain: `ml-expert` (training dynamics — ML
axis not covered by the generic plugin) and `molzoo-auditor` (the
molzoo paper↔code verifier — see below). The former
`molnex-compute-scientist` and `molnex-pm` have been promoted into
the plugin as generic `compute-scientist` and `pm` agents and are now
invoked via `/mol:review`.

## Molzoo spec workflow — one skill, one agent, one file

Each encoder in `src/molzoo/` owns **one** spec artifact at `src/molzoo/specs/<encoder>.md`, following the strict 10-section template at `.claude/skills/molzoo-spec/template.md` (Scope & Boundary → Paper↔Code Mapping → Reference Alignment → Adaptation Ledger → Mathematical Contract → Config Mapping → Benchmark Contract incl. embedded Run log → System Boundary → Version Pinning → Spec Drift Policy).

**Spec-first.** §2 / §3.1 / §5 of `<encoder>.md` MUST be filled from the paper + reference impl **before** any encoder code is written. The flow is `scaffold → fill → implement → audit`; reverse workflows ("write code first, retrofit the spec") are the failure mode this whole apparatus exists to prevent. See "Spec-First Principle" in `.claude/skills/molzoo-spec/SKILL.md` for the full rule.

There is **no** `_experiments.csv` and **no** `_walkthrough.md`. Run logs are §7.4 of `<encoder>.md` (markdown table, append-only). Audit verdicts from `molzoo-auditor` are **printed to the developer at chat time**, not tracked as files; on 📝/🆚 the agent patches `<encoder>.md` directly and that diff is the persistent trace.

One skill + one agent. Both repo-local under `.claude/skills/` and `.claude/agents/` — do not promote to `~/.claude/` without a second repo adopting the same pattern.

| Trigger | Command | Effect |
|---------|---------|--------|
| Introduce a new encoder | `/molzoo-spec <encoder> --paper <arxiv_url> [--ref <org>/<repo>@<sha>]` | **create mode** (auto-detected when spec is missing): copies the 10-section template to both spec copies; header + §1 + §9 only; status `draft`. Code authoring forbidden. |
| Fill placeholders or fix drift in an existing spec | `/molzoo-spec <encoder>` | **update mode** (auto-detected when spec exists): fills §2 / §3.1 / §5 from paper + reference, reconciles drift with code, refreshes stale anchors. Bumps status `draft` → `partial` once §2/§3/§5 are filled; required before `<encoder>.py` is written. |
| After a bench / training run | `/molzoo-spec <encoder> --log <k=v ...>` | appends one row to §7.4; on MAE regression > 10 % or dirty tree, warns + suggests `molzoo-auditor`. |
| Asking a question about spec content | (no command — answer inline) | The skill's §"Lookup Behavior" hard rule applies: always `Read` the spec first, quote matches verbatim, refuse on a miss and point at `molzoo-auditor`. |
| Verify code vs spec vs paper | `molzoo-auditor` agent | **prints** ≥ 1 verdict report (✅ ℹ️ ⚠️ 📝 🆚) to the developer; ⚠️ (code-drift) recommends a code change without editing spec; 📝/🆚 patch `<encoder>.md` directly; may backfill the §7.4 `note` cell. |

**Closed-loop contract.** Every operation either updates `<encoder>.md`, prints a verdict, or explicitly delegates — no operation closes silently:

1. `/molzoo-spec <encoder> --paper <url>` triggers create mode only when the spec file is absent; status starts `draft`.
2. `/molzoo-spec <encoder>` triggers update mode when the spec exists; it MUST be run before any code is written and bumps status to `partial`. Code authored against status `draft` is a process violation, not just drift (see §10.3 in the spec template).
3. `/molzoo-spec <encoder> --log <k=v ...>` is append-only on §7.4; it MUST check the previous row and warn (not auto-invoke) on anomalies.
4. Questions about spec content are answered inline under the skill's §"Lookup Behavior" hard rule — `Read` the spec, quote verbatim, refuse on a miss, refuse outright on a `draft`-status spec and point at update mode.
5. `molzoo-auditor` MUST print ≥ 1 verdict per invocation (even `✅ confirmed, no drift`), citing the triggering `run_id` or question + paper section + spec row + code file:line. ⚠️ (code-drift) is print-only; 📝/🆚 produce file diffs in `<encoder>.md`. The auditor never edits code — code changes are always the user's call.

The 10-section structure of `<encoder>.md` is **immutable** — adding / removing / renaming a section is a §10.2 breaking change. Invariants are enforced inside the skill, not via `settings.json` hooks, for now.
