# MolNex — Evolving Decisions

Capture non-obvious decisions, trade-offs, and provisional rules here via `/mol:note`.
When an entry stabilises, `/mol:note` promotes it into `CLAUDE.md` and deletes it here.

This file is read by every `mol:*` skill / agent before running checks.

---

## TensorDict — no subclass for batch data (2026-05-18)

**Context.** `AtomData / EdgeData / GraphData / GraphBatch` were four empty
`TensorDict` subclasses that added zero functionality (no new methods, no
overrides) — only docstring-schema and type tags. They blocked `torch.compile`,
increased coupling to unstable tensordict subclass APIs, and created confusion
for newcomers needing to understand four type names for what is really just a
nested dict of tensors.

**Decision.** Removed all four subclasses (spec `tensordict-cleanup`).
Post-collate batch is now a plain `tensordict.TensorDict` with three nested
namespaces (`atoms`, `edges`, `graphs`). Schema contracts (field names,
shapes, `batch_size`) are documented in CLAUDE.md, not enforced by Python types.
Encoders are encouraged to inherit `TensorDictModuleBase` for `in_keys/out_keys`
validation, but `nn.Module` with `forward(td: TensorDict) -> TensorDict` is
also valid.

**Status.** stable (promoted to CLAUDE.md).

---

## Industrial module layout + test mirror (2026-07-16)

**Context.** PiNet train/infer work was blocked by god-files (`molzoo/pinet.py`
756 LOC, `molrep/interaction/pinet.py` 11 classes), dual homes
(heads/pooling/export), and tests that crossed package boundaries
(`test_molzoo` owning `PadMolecularBatch` and molrep layer checks).

**Decision — package ownership (one-way):**

```
molix  → infra (data, trainer, compile, export, md)
molrep → pure representation (embedding / interaction / readout)
molpot → physics (heads / derivation / potentials)
molzoo → recipes that wire molrep blocks into encoders (+ temporary
         potential façades under molzoo.pinet.potential for import
         stability; long-term home is molpot)
```

Hard rules:
1. `molrep` must not import `molpot` or `molzoo`.
2. `molpot` must not import `molzoo` (accept encoder Protocol / tensors).
3. One physics concept → one owner package (no duplicate CosineCutoff, etc.).
4. Prefer explicit `nn.Linear(in, out)` over `LazyLinear` on any export /
   functorch / compile path.

**Decision — source ↔ test mirror:**

```
src/<pkg>/<area>/<module>.py
  → tests/test_<pkg>/test_<area>/test_<module>.py
```

- One source module ↔ one primary test module.
- One public class ↔ dedicated test class (and preferred: dedicated file when
  the source file still hosts multiple public types during migration).
- Cross-package integration tests go under `tests/regression/` or an explicit
  `test_*_integration.py` name — never as a substitute for unit mirrors.
- Gate: `python scripts/check_test_mirror.py --strict-pinet`.

**PiNet spine layout (landed):**

```
src/molrep/interaction/pinet/{ff,message,residual,blocks}.py
src/molzoo/pinet/{spec,geometry,encoder,potential,properties}.py
tests/test_molrep/test_interaction/test_pinet/...
tests/test_molzoo/test_pinet/...
tests/test_molix/test_batch/test_pad.py
```

**Status.** active (PiNet spine done; full-repo mirror is incremental).

---

## Force derivation — dual explicit backends (2026-07-29)

**Context.** CLAUDE.md and some docs still said “forces always via
`torch.func.grad`”. Reality is more nuanced: cuEquivariance fused kernels
register a legacy `autograd.Function` without `setup_context`, so
`torch.func.grad` rejects them (pytorch#170834). PiNet (pure torch) benefits
from functorch + `torch.compile(fullgraph)`; MACE / OMOL need autograd.

**Decision.** Canonical entry is `molpot.derivation.ForceDerivation`:

| `method` | Implementation | Use when |
|----------|----------------|----------|
| `"autograd"` (default) | `torch.autograd.grad` | cuEq / MACE / any model; always correct |
| `"functorch"` | `torch.func.grad` | pure-torch energy graphs (e.g. PiNet) for single-backward + fullgraph |

`BasePotential.calc_forces` uses `ForceDerivation(method="autograd")` (protocol
path must work for every potential). Models must not invent a third force path.

**Status.** active (promoted into CLAUDE.md Key Design Patterns; docs/gradients
aligned). Landed 2026-07-29: OMOL/`energy_forces`, Sonata forces, PiNet via
`molpot.derivation.protocol` helpers, `BasePotential.calc_forces` all go
through `ForceDerivation`.
`has_aux` supported on functorch backend for single-pass eval.

---

## Soft-optional native deps for datasets (2026-07-29)

**Context.** `MolRecSource` needs `molpy.MolRec` (optional public surface;
not every molpy build ships labeled-configuration records). Hard-importing
optional surfaces broke `from molix.datasets import …` for every source.

**Decision.**
1. Soft-import `MolRecSource` in `molix.datasets.__init__`; on failure export a
   stub class whose `__init__` raises a clear `ImportError` (never `None`).
2. All Element / Frame / Block / Box / Trajectory / UnitsError / MolRec access
   goes through **`molpy` only**.
3. QM9 / ThreeBPA / WaterLES / MolRec Element lookups use `molpy.Element`.

**Status.** active.

---

## Never import molrs from molnex Python (2026-07-29)

**Context.** molrs is the Rust core; molpy is the supported Python façade
(re-exports Element, Frame, Trajectory, UnitsError, …). Direct `import molrs`
couples molnex to native ABI details and bypasses molpy versioning.

**Decision.** **Hard rule:** no `import molrs` / `from molrs import …` in
`src/` or `tests/`. Use `from molpy import …` exclusively. Ecosystem README
may still *mention* molrs as a dependency of molpy.

**Status.** active.

---

## Native op lib — arch tag + torch_python (2026-07-29)

**Context.** Custom autograd Functions need `torch::autograd::_wrap_outputs` from
`libtorch_python`, which is not in `${TORCH_LIBRARIES}`. Also multi-arch HPC
nodes need both x86_64 and aarch64 builds in the same tree.

**Decision.**
- CMake `OUTPUT_NAME = molnex_opLib.${CMAKE_SYSTEM_PROCESSOR}`; loader loads only
  the arch-tagged path (no untagged fallback).
- Always link `torch_python`; on Linux wrap with GNU-ld `--no-as-needed` /
  `--as-needed` so the DSO is kept at load time. Non-Linux: plain link only.

**Status.** active.

---

<!-- mol:note:topic:cueq-use-fallback -->
## cuEq `use_fallback` split by force backend (2026-08-08)

**Context.** Hardcoding the pure-torch path abandoned fused kernels even when
forces used autograd; conversely `MACEMatpes` shipped `use_fallback=True` as
its default — a measured **35.7x** per-step regression with no correctness
upside (its forces are always autograd, so the functorch reason never applies).

**Rule**: Block constructors (`ConvTP`, `SymmetricContraction`,
`DensityInteraction`, `ProductHead`, …) default `use_fallback=True`
(functorch-safe). **Autograd-only full models default `False`**
(`MACEMatpes`, `MACEOMol`); CPU/test call sites pass `True` explicitly.
Composable encoders (`MACE`) expose the knob and inherit the safe default.

**Supersedes**: the 2026-07-29 entry (which left MACEMatpes on the slow
default and MACE with no knob at all).

**Status.** active.

---

<!-- mol:note:topic:cueq-ops-capability -->
## [2026-08-08] cuEq fused-kernel capability is probed, never assumed

Without the `cuequivariance-ops-torch` wheel, cuEq honours
`use_fallback=False` by silently degrading ~30x (one UserWarning).

**Rule**: Report fused-kernel status from an actual
`import cuequivariance_ops_torch` probe, never from the `use_fallback`
request flag. GPU installs use the `cueq-cu12` / `cueq-cu13` extras
(pyproject); a degraded run must warn, not self-report "fused".

---

<!-- mol:note:topic:bench-guard-encoders -->
## [2026-08-08] Headline-number encoders need a benchmark guard

The `use_fallback` regression shipped because MACE had no benchmark while
carrying the repo's headline GH200 compile numbers.

**Rule**: Every molzoo encoder whose docs/specs cite performance numbers has a
`benchmarks/bench_<encoder>.py` guard (see `bench_mace_matpes.py`, which
asserts the fused/fallback ratio).

---

<!-- mol:note:topic:scatter-onehot-optin -->
## [2026-08-08] scatter one-hot GEMM is explicit opt-in

The one-hot matmul (bit-exact under inductor) measured 2.4–3.3x slower than
`index_add_` at every profiled molecular-graph shape, including inside the
size window that used to auto-select it.

**Rule**: `scatter_sum_compile_safe` defaults to `index_add_`. The one-hot
GEMM is chosen only by `MOLNEX_SCATTER_ONEHOT=1` (bit-exactness as a
deliberate, global choice) — never by a size heuristic.

<!-- mol:note:topic:build-check-scope -->
## [2026-08-09] build.check covers tests/scripts/regressions + advisory ty

The gate used to lint `src/` only, which let `scripts/` and test-side debt
accumulate invisibly (found during the mace-restructure follow-up sweep).
`benchmarks/` is deliberately excluded while the PiNet bench scripts are
mid-experiment; fold it in once that work lands.

**Rule**: `mol_project.build.check` runs ruff (check + format) over
`src/ tests/ scripts/ regressions/` plus `ty check src/
--exit-zero-on-warning` (warnings stay advisory per `[tool.ty.rules]`;
real type errors block). Do not narrow it back to `src/` alone.

<!-- mol:note:topic:init-dtype-at-construction -->
## [2026-08-09] Construct at config.ftype; init consumes the global RNG

A 48-site sweep found `nn.Linear`/`nn.Embedding`/`cuet.Linear`/buffer
constructors omitting `dtype=config.ftype`, yielding silent fp32 params
under the fp64 config (hidden by post-hoc `.double()` casts in fixtures).
`cuequivariance_torch.Linear` honours `dtype=` — the old claim that it
ignores `config.ftype` is false.

**Rule**: every parameter/buffer constructor under `src/` passes
`dtype=config.ftype` explicitly (fp64-contract tests pin the pattern
per module). Weight inits draw from the **global** torch RNG (e3nn
convention; `_ScalarO3Linear` weight ~ N(0,1), bias zero) — never a
private Generator, never zero-init for trainable readout weights.

<!-- mol:note:topic:all-alphabetized -->
## [2026-08-09] __all__ stays alphabetized

ruff's isort rule does not cover `__all__` literals, so ordering drifts
silently (caught in molzoo/mace/__init__.py during review).

**Rule**: keep `__all__` alphabetically sorted in every package
`__init__.py`; re-sort when inserting a name.

<!-- mol:note:topic:ty-ignore-syntax -->
## [2026-08-09] Type-suppression pragmas: ty syntax only

The repo's checker is ty (no mypy/pyright config exists). mypy-style
`# type: ignore[code]` is inert under ty — 67 dead pragmas were swept
2026-08-09. Most suppressions are unnecessary anyway: the known
TensorDict/torch-stub false-positive classes are already downgraded to
"warn" via `[tool.ty.rules]` in pyproject.toml.

**Rule**: never write `# type: ignore[...]`. If a per-line suppression
is truly needed, use `# ty: ignore[rule]`; prefer relying on the
`[tool.ty.rules]` downgrades over per-line pragmas.

---

## Learnable classical FF placement (2026-08-10)

**Rule.** Reuse molpy≥0.13; no new `Foo(method=…)`; non-diff work sinks to molpy/molrs.
Full text: [learnable-classical-ff.md](learnable-classical-ff.md).

**Status.** active.

---

<!-- mol:note:topic:block-store-lustre-io -->
## [2026-08-14] Block reads stay on mmap; the cliff is locality, not the path

`scripts/store/bench_block_read.py` on `arrhenius1.hpc.arrhenius.naiss.se`
(login node, project Lustre `/nobackup/proj/disk`, PFL 1 stripe × 4 MiB,
torch 2.12.1+cpu, `OMP_NUM_THREADS=1`), median of 3 runs over a 1.03 GB
block (256 units × 8192 frames × 40 atoms, 17 columns). One pass gathers
65536 frames = 31.5 MB of `pos` payload in 256-frame batches; `cold`
is the first touch after `POSIX_FADV_DONTNEED` (`mincore` witnessed 0.000
residency), `warm` the repeat. MB/s is useful payload.

| order | path | cold MB/s | warm MB/s |
|---|---|---|---|
| sequential | mmap | 104 | 1114 |
| sequential | pread | 128 | 218 |
| shuffled | mmap | 2.2 | 602 |
| shuffled | pread | 1.6 | 178 |

Write side: 1000 small units (12 atoms, 4 frames → 18 fsyncs/unit) commit at
**153 ms/unit = 6.5 units/s**; the ~1 MB-per-unit read block at 185 ms/unit
= 21.8 MB/s. `seal()` streams sha256 at **1.4 GB/s** (spec assumed 1 GB/s).

**Rule**: `BlockReader._map_column` stays mmap. Warm, mmap wins 5.1× (seq) /
3.4× (shuffled); cold the paths are a wash (`pread`'s best reproducible edge
1.23×, under the spec's 1.5× follow-up threshold), so no `pread` column
reader. Budget the write path by **fsync latency, not bandwidth** — cost is
per *unit* (~18 fsyncs), so 200 k units ≈ 8.5 h regardless of unit size.
Cold shuffled frame access is ~2 MB/s (~3–5 k frames/s) on **both** paths,
~270× below warm mmap: a consumer pre-warms a block sequentially rather than
relying on random faults (conveyor concern, `-03`). Cold cells vary up to ~3×
run-to-run on a shared login node — re-measure on the compute node before
prefetch policy is frozen.

**Status.** active (spec `streaming-data-tiers-02-block-format`, task 8).

---

<!-- mol:note:topic:sampler-plan-table-scale -->
## [2026-08-14] Sampler plan tables stay in code form; the plan, not the table, is the cost

`scripts/batch/bench_plan_tables.py` on `arrhenius1.hpc.arrhenius.naiss.se`
(login node, torch 2.12.1+cpu, CPython 3.14.5, `OMP_NUM_THREADS=1`,
2026-08-14). One synthetic count-only split per cell (10 frames/unit, 4 tasks,
8–32 atoms/frame, `max_atoms=4096`, seed 20260814) planned through
`TaskMixtureBatchSampler` in both representations: **list** = per-frame
`task_labels: list[str]` + `group_ids: list[str]` via
`FrameLabelTable.from_labels`; **code** = `(F,) int32` `task_codes` /
`group_codes` + bounded `task_names`. Each cell runs in its own subprocess, so
`peak` is that cell's `/proc/self/status:VmHWM`; `net` drops the ~301 MB
interpreter + torch baseline. Both forms produced **byte-identical plans** at
every F (sha256 over the plan; the bench aborts loudly otherwise).

| F | form | table s | plan s | peak MB | net MB | net B/frame | batches |
|---|---|---|---|---|---|---|---|
| 1e6 | list | 0.04 | 1.56 | 485 | 185 | 185 | 4 895 |
| 1e6 | code | 0.00 | 1.45 | 450 | 149 | 149 | 4 895 |
| 1e7 | list | 0.32 | 13.75 | 2 067 | 1 766 | 177 | 48 930 |
| 1e7 | code | 0.04 | 12.74 | 1 776 | 1 475 | 148 | 48 930 |
| 1e8 | list | 3.20 | 147.42 | 17 534 | 17 234 | 172 | 489 355 |
| 1e8 | code | 0.50 | 144.11 | 14 876 | 14 576 | 146 | 489 355 |

list/code peak ratio **1.08× (1e6), 1.16× (1e7), 1.18× (1e8)**; table build is
6–9× cheaper in code form but is only ~2 % of plan time; plan build is within
1.08× either way. A repeat of the default run reproduced the plan digests
exactly and the peaks within 1 % (483 / 454 MB at 1e6, 2 082 / 1 765 MB at
1e7); wall times swing ±20 % on a shared login node, so treat the seconds as
orders of magnitude and the peaks as the measurement.

**Verdict** (default `--rss-budget-gb 6.0`): `plan-table verdict — no crossover
below F=1e8 in peak RSS (largest measured ratio 1.16x at F=1e7)`. That default
run skips both 1e8 cells on projection (21.9 / 14.3 GB > 6.0 GB); they were
measured in a bounded follow-up (`--frames 100000000 --rss-budget-gb 24`,
~5 min, sequential, single-threaded) and confirm the trend at 1.18×.

**Rule — chosen representation: code form** (`(F,) int32` columns + bounded
`task_names`) for `02-view` and `03-plan`. It wins every measured axis, needs
no decode step, and is the only part of the footprint a successor still gets to
choose.

**What the numbers actually say.** The `01-tables` hand table is confirmed in
shape and refuted in emphasis. The steady-state *table* difference is only
~27–29 B/frame (two list pointers, one 8-char id `str` per unit, `int64`
instead of `int32` columns); the transient `from_labels` code list (~36
B/frame) never survives to peak. The shared *planning* machinery — the
`tuple(indices)` parent-index materialization, the `(task, group)` bucketing
dict, the per-unit member lists and the plan's own lists — costs ~146 B/frame
in **both** forms, i.e. ~85 % of the footprint. Code form buys ~17 % (0.29 GB
at 1e7, 2.7 GB at 1e8); only unit-granularity planning (`03-plan`) can touch
the other 85 %. Corollary: the `01-tables` Bench paragraph sized the 6.0 GB
budget on an estimated "≈5 GB at 1e8 list-form" — measured **17.5 GB**, so
under the default budget 1e8 self-skips by design.

**Status.** active (spec `streaming-data-tiers-04-scheduler-integration-01-tables`, task 7).
