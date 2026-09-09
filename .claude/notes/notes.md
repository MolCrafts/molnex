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

<!-- mol:note:topic:native-op-rebuild -->
## Native op lib — arch tag + always rebuild (2026-08-31)

**Context.** Custom autograd Functions need `torch::autograd::_wrap_outputs` from
`libtorch_python`, which is not in `${TORCH_LIBRARIES}`. Multi-arch HPC needs
both `x86_64` and `aarch64` artifacts in the same tree. A stale
`libmolnex_opLib.aarch64.so` (wrong torch ABI) silently disabled native
neighbor ops and MD fell back to `bin=0.0`. Login-node `maturin` then overwrote
in-tree `molrs/_lib.abi3.so` with x86-64, so GH200 `import molix` died even
after the molnex op rebuilt. Probe without GCC 14 `libstdc++` on
`LD_LIBRARY_PATH` failed with `CXXABI_1.3.15`.

**Rule**: CMake `OUTPUT_NAME = molnex_opLib.${CMAKE_SYSTEM_PROCESSOR}`; loader
loads only the arch-tagged path (no untagged fallback). Always link
`torch_python` (`--no-as-needed` on Linux). **Every `.aarch64` env/torch
change rebuilds ALL native libs together on GH200 — never reuse a stale
`.so`, never rebuild molnex op without also re-asserting molrs aarch64.**

1. `molrs/_lib.abi3.so` via `maturin develop --release` with
   `RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu` and
   `CARGO_TARGET_DIR=molrs-python/target-aarch64`. The in-tree file is
   untagged: login rustc emits x86-64 into the same path. Keep
   `_lib.abi3.so.aarch64` / `_lib.abi3.so.x86_64` copies. Never run
   molrs maturin on the login node if GH200 still needs that tree.
2. `src/molix/op/libmolnex_opLib.aarch64.so` against **this** interpreter's
   torch. Export `TORCH_CUDA_ARCH_LIST=9.0` (PyTorch cmake ignores
   `CMAKE_CUDA_ARCHITECTURES`). GH200 CUDA arch is `90`, not the cache's `75`.
3. Runtime `LD_LIBRARY_PATH` = GCC 14.3 `lib64` + `torch/lib` + CUDA `lib64`
   **before** `python` starts (uv CPython is linked to system libstdc++).

Recipe: `.scratch/polar_npt/rebuild_op.sbatch` (molrs + op) and step 4b of
`upgrade_aarch64.sbatch`. Missing/ABI-mismatched `.so` is an environment
error for MD, not a reason to skip the rebuild.

**Supersedes**: "warn and continue with pure-Python NeighborList" as the GH200
workflow; "rebuild only molnex op, leave molrs".

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

<!-- mol:note:topic:test-tiers-orthogonal -->
## [2026-09-07] One on-disk format (molix.block/1); `regressions/` deleted (user rule)

**Decision (user).** molnex keeps **one** on-disk format: `molix.block/1`.
`molix.store.cache.PackedCache` — the single-`.pt`, schema-inferred, run-scoped
packed cache — is gone, along with `MmapDataset` / `CachedDataset` and the
`_PackedCollateFn` route in `DataModule`.

**Why it could go.** MolHub owns dataset production and already writes only
`molix.block/1` (`molhub/loader/materialize.py`: `_BLOCK_FORMAT`,
`_WRITER_MODULES = ("molix.store.block", "torch")`); it never imported
`PackedCache`. Inside molnex `src/`, the only production writer left was a
profiler helper that round-tripped samples through a temp file purely to
"exercise the same code path real workflows do". Everything else that wrote
one was a test or a `regressions/` script.

**What survived, and why.** The *packed payload* is the real contract and it
stays: `FrameStoreReader.frame_payload` produces it, `PackedView` wraps it, and
`collate_packed` turns it into a batch without materialising per-sample dicts.
`PackedCache` was only one on-disk spelling of that payload, and the worse one
(no manifest, no seal, no digests, no resume, no splits). The unpacker moved to
`molix.store.block_view._unpack_packed_sample`; `SubsetDataset.packed_view()`
is defined explicitly rather than inherited, because `__getattr__` forwarding
would silently hand back an **unmapped** view.

**Consequences that are features, not losses.** Blocks carry no neighbour
graph by design, so the cutoff graph is built downstream by
`molix.batch.NeighborList` — the cache's `edges` bucket, the sampler's
`max_edges` budget and `avg_num_neighbors` had no producer left. The
`TensorDict.memmap_` argument in the old docstring was answering the wrong
question: it argued against `LazyStackedTensorDict.memmap_` (one directory per
sample, ~1M files for QM9), which nobody proposed. Packing first and memmapping
the packed TensorDict would have been ~15 files. The real reason to keep one
format is that two formats for the same job is one too many.

**`regressions/` deleted (user rule, same day).** All 69 scripts, 1.1 MB. The
operator's reason: *nothing in this repo is finished enough to regress against*
— a golden captured from unfinished code pins today's behaviour, not a
contract. `tests/` is the only test tier now; CLAUDE.md's "Public-API scenarios
→ `regressions/`" routing and the `build.check` command were updated with it.

**Status.** active.

---

## [2026-09-07] Tests assert behaviour, not surface (user rule)

**Rule (operator, 2026-09-07).** `tests/` holds correctness tests. Beyond the
static source scans banned the same day, three more genres are out:

1. **Namespace / re-export pins** — `__all__` contents, `__module__` location,
   "this name is gone", "A is the same object as B", "the symbol is importable".
   An import that breaks fails every real test in the package already.
2. **Protocol-membership pins** — `isinstance(x, SomeRuntimeProtocol)` and its
   negation. A `runtime_checkable` Protocol check *is* a `hasattr` check.
3. **Constructor / default echoes** — build an object, assert the arguments or
   the field defaults come back. CLAUDE.md already forbade "tests of signature
   defaults"; molrep's `test_valid_config` / `test_initialization` pairs were
   nine copies of exactly that.

Deleted: 151 test functions + 4 whole files
(`test_hooks/test_reexports.py`, `test_nn/test_init.py`,
`test_md/test_forcefield.py`, `test_forcefield/test_exports_cut.py`),
~1230 lines. Collected tests 3173 → 3021.

**Kept, deliberately.** Anything asserting a *computed* value, a state
transition, an error path with real inputs, or a measured resource property:
`PackedCache` atomicity (no partial file after a failed save), the sampler's
"no `(F,)`-length table survives construction", `DEFAULT_ALPHA_NORM_FACTOR`
equals its Bohr³ → e·Å²·V⁻¹ formula, the class-I 1-2/1-3 scaling constants
(physics, not a default), and the MACE readout-width wiring relation.
The molzoo lazy-import probes survived and moved to `slow`: "importing molzoo
must not import cuEquivariance" is a real, load-bearing contract.

**This was a maintenance win, not a speed win — measured.** Deleting the 126
pins moved user CPU from 76.7 s to 75.3 s, i.e. nothing: a `hasattr` assertion
costs nothing to run. The speed win came only from the static scans, which
walked the filesystem. Judge test-suite bloat by what a test *pins*, not by
what it costs.

**Measuring on this box.** The login node has **4 cores**, 260+ users and a
routine load average above 100, so wall-clock suite timings are not comparable
between runs — the same tree measured 106 s and 418 s hours apart. Use
`/usr/bin/time` user+sys CPU, and treat any wall-clock delta under 2× as noise.

**Status.** active.

---

## [2026-09-07] No static source-scanning tests (user rule)

**Rule (operator, 2026-09-07).** `tests/` must not contain tests that assert on
the **text or AST of source code, docstrings, or docs files**. A test asserts
runtime behaviour or it does not exist. Deleted outright: every
"package X does not import Y" AST scan, every "this method delegates to that
helper" `inspect.getsource` / `ast.unparse` check, every banned-identifier grep
over `src/`, every "no `.item()` on this call chain" AST walk, and every
docstring-presence / docstring-content assertion
(`tests/test_molix/test_io/test_docstring.py` went as a whole file). 31 tests
across 22 files, ~800 lines.

**Why they were also slow.** Five of the fast tier's six slowest tests were
these scans, because `rglob("*")` over `src/` walks
`src/molix/op/build*/` — thousands of CMake artifacts on a shared HPC
filesystem. The single worst (`test_no_compiled_alias_survives_in_the_tree`)
cost 12 s to assert a grep result.

**What replaces them.** Nothing in `tests/`. Import-direction and
banned-dependency rules are stated in CLAUDE.md and enforced by review and by
`ruff` / `ty`; if a boundary is worth machine-checking it belongs in a lint
rule, not in a unit test. The two behavioural
survivors of the sweep kept their real assertions and lost only their static
tails (`Potentials.bind_cell` / `compile_core` guards; provenance
`decode_with_confidence`). `test_repository_boundaries.py` keeps exactly one
test — `find_spec("molix.data") is None` — which probes the interpreter, not a
file.

**Also moved to `slow`** (they matched the existing slow-tier definition and
were in the fast tier by oversight): the subprocess lazy-import probe
`test_reaching_a_spec_class_does_not_import_the_encoder` (10.6 s) and the one
fast-tier test that actually dynamo-traces,
`TestCompiler::test_compiled_module_produces_same_output` (6.5 s).

**Measurement note.** The fast tier is 3212 tests / 181 s on **4** cores
(`-n 4`, `OMP_NUM_THREADS=1`). The "couple of minutes on 12 workers" budget in
the 2026-09-03 entry assumed a 12-core box; on the 4-core login node the same
suite is ~3 min, and `-n 12` there is a 3× oversubscription that makes it
*slower*, not faster.

**Status.** active.

---

## [2026-09-07] Virial / stress convention: Cauchy `σ = W/V`, shear not doubled (user rule)

**Context.** Every strain path in the repo built a full 3×3 strain leaf, applied
`ε_sym = (s + sᵀ)/2` inside the graph, then post-processed the gradient with
`g + gᵀ − diag(g)` — the *engineering-shear* convention (`γ_αβ = 2ε_αβ`, the
`(α,β)`/`(β,α)` pair counted as one variable). That doubles every off-diagonal
component relative to the Cauchy stress. It reached the LAMMPS virial, the
MACE-Polar benchmark's "ours vs mace-torch" comparison (where it showed up as a
~2× shear disagreement I first misread as a sign error), and the FD tests, which
were self-consistently doubled on *both* sides and therefore never caught it.

**Decision (user).** molnex maintains **one** convention and everything adapts
to it: `W = ∂E/∂ε` over the symmetric strain tensor, so `dE = Σ_αβ W_αβ dε_αβ`
runs over all nine entries and a shear `W_αβ` is **not** doubled;
`σ = W / V`. This is the `Σ r ⊗ f` virial and what LAMMPS / mace-torch report.

**Where it lives.** `molpot.derivation.symmetrize_strain_grad` (defined in
`derivation/stress.py`) is the single definition — `0.5 * (g + gᵀ)`, which for a
strain-symmetric energy only removes finite-precision asymmetry. Consumers:
`autograd_forces_and_virial` (MACE/cuEq virial), `StressDerivation`,
`Sonata` (inline, same expression). Any new strain path routes through it.

**Consequences.**
- Off-diagonal virial/stress components are now half their pre-2026-09-07
  values; diagonals (and therefore `BerendsenBarostat`, which reads only
  `virial.trace()`) are unchanged.
- `regressions/polar-compile-clean-03-wire.py`'s `GOLDEN_VIRIAL` shear entries
  were recaptured before that script was deleted with the rest of
  `regressions/` on 2026-09-07; the ratio old/new was exactly 2.000
  off-diagonal, 1.000 on the diagonal — the cleanest possible confirmation
  that only the convention moved.
- FD tests that perturb `ε_ab` and `ε_ba` together must divide the off-diagonal
  difference by 2; that halving is now written out in `test_ewald.py`,
  `test_sonata_periodic.py` and the MACE FD-virial test.
- Debt found and fixed on the way: `StressDerivation` had no mirror test at all
  and divided by `volume.view(-1, 1, 1)`, silently adding a leading axis for an
  unbatched `(3, 3)` cell. Now `unsqueeze(-1).unsqueeze(-1)`, with
  `tests/test_molpot/test_derivation/test_stress.py`.

---

## [2026-09-03] Test suite: two tiers, orthogonal unit tests only (user rule)

**Rule (operator, 2026-09-03).** `tests/` holds unit tests only — one
function or module per test, each behaviour asserted once at the lowest
layer that owns it. No e2e runs (training loops, MD rollouts,
export→load→run round trips, spawned DataLoader workers), no
reference-value parity / regression suites, no benchmark or profiling
runs. Deleted on 2026-09-03: `tests/regression/` (torch-pme parity),
`tests/test_scripts/`, `tests/test_molpot/test_les_parity/`,
`test_composition/test_potential_parity.py`,
`test_elec/calculators/test_workflow.py`, `test_molix/test_export.py`,
`test_molix/test_engine/test_export.py`, `test_core/test_trainer_smoke.py`,
the two profiler e2e files, the spawn-worker DataModule cases, Ewald's
`TestTorchCompile`, `test_potentials_compile`, and the Allegro /
typing-probe overfit runs. Public-API scenarios with hard-coded goldens
lived in `regressions/` (plain scripts) until that tier was deleted on
2026-09-07. Known losses to carry: no unit
test now covers `molix/export.py` / `molix/engine/export.py`, the
LAMMPS meta-block schema (was inside the deleted export file), or an
explain-level compile guard for `EwaldMultipoleEnergy`; the "bounded
overfit test per head" clause of the training-readiness law has no
in-tree test any more.

**Tiers.** Default `pytest` = fast unit tier, `addopts = -m "not slow"
--durations=10`, measured 3210 tests / 96 s wall on 12 x86 workers
(was 3938 / 192 s uncontended, ~10 min when other pytest runs competed).
`-m slow` = on-demand tier: anything that compiles, dynamo-traces
(`Compiler.explain`), AOT-exports, spawns workers, loads external
weights (`TestOfficialOMolWeights`) or probes imports in a subprocess —
one dynamo trace per compile seam, nothing duplicated across layers.
`python scripts/affected_tests.py --run` runs only the tests mirrored
by the git diff (mirror rule of `scripts/check_test_mirror.py`) plus
`tests/test_repository_boundaries.py`. CI config is the operator's —
not edited.

**Orthogonality decisions applied.** The 2026-09-03 polar-compile-clean
tests were cut from 117 to ~80: the "isolated branch still breaks"
guard lives once in `test_hartree.py` (the 35 s MacePolar copy and the
features copy are gone); `_KSpaceGridHolder` facts (bind_cell, k_cutoff
delegation, capacity knob) are tested once in `test_hartree.py`, the
grid itself in `test_kspace.py`; the (1,3,3)-vs-(3,3) squeeze once in
`test_kspace.py` plus one consumer check; static grep / AST invariants
once (hartree); the byte-identity source snapshot and the
signature-default / `hasattr` tests were deleted; the SI/NOSI gradient
twins collapsed into one parametrized fact; `test_symmetry.py` runs one
seed (exact invariances need no sweep); the integrator compile test
uses `aot_eager` as its own comment intended (57 s → seconds).

**Status.** active.

---

<!-- mol:note:topic:cueq-conv-fusion -->
## [2026-09-02] cuEq conv fusion: gather/scatter live inside the TP kernel

**Context.** `ResidualInteraction` and `_DensityInteractionBase` gathered
`node_feats[source]`, ran `conv_tp` per edge and `scatter_sum`'d the
`(E, target.dim)` message. MACE-POLAR-1-M (512 channels, l_max=3, target
dim 8192) on the 4995-atom PEO box (E = 2.86·10⁵ directed edges at 6 Å)
makes that tensor 9.4 GB per layer in fp32 — plus its layout-transpose copy
and, in the backward, `∂E/∂m_ji` of the same size again. The bench OOM'd
at 93 GiB on a GH200 (`bench5k/slurm-molnex-1916374.err`, 2026-09-02).

**Rule.** Every cuEq channel-wise convolution uses the indexed
`cuet.ChannelWiseTensorProduct` path. The ordinary layout passes
`indices_1=source, indices_out=target, size_out=N`; the memory-bounded layout
interprets its configured size as a maximum, balances
`ceil(N / max_block_size)` receiver intervals, then passes one interval's
global sources, block-local targets and `size_out=actual_span`. In both cases
`uniform_1d` owns the fused gather, contraction and scatter on GPU (`naive`
keeps the CPU fallback working). Never
`conv_tp(x[source], …)` followed by a Python scatter, and never maintain a
local TP/CG kernel. This is the same cuEq execution primitive used by official
mace-torch with `conv_fusion=True`; receiver blocking changes only the fixed
work partition. Pinned by
`test_message_matches_explicit_gather_scatter` in
`tests/test_molrep/test_interaction/test_residual.py` and
`tests/test_molrep/test_interaction/test_mace/test_density.py`, plus the
blocked/full equivalence tests in those files and `test_conv.py`.

**Inference grad mode.** `calc_energy_forces_virial` / `calc_forces` /
`grad_force_pass` resolve `create_graph` from the *ambient* grad mode
(training needs it). Outside `torch.no_grad()` that is `retain_graph=True`:
the forward tape and every backward intermediate stay alive until the
gradient is dropped. `molix.md.MD.run` wraps in `no_grad`; any hand-rolled
eval loop (bench scripts, parity checks) must do the same.

**Measured (GH200, PEO 4995 atoms / 2.86·10⁵ edges, MACE-POLAR-1-M,
`calc_energy_forces_virial` under `no_grad`, 2026-09-02).** Before: OOM at
93 GiB. After: fp32 peak 28.3 GiB (forward-only 15.0), 0.38 s/step; fp64
peak 56.5 GiB, 0.59 s/step. Official mace-torch (cueq, conv fusion) on the
same box: 25.2 GiB / 0.5–1.0 s at fp32. Records:
`peo-tg/experiments/size-convergence-mace-polar/bench5k/verify_molnex*.json`.

**Receiver blocking — REMOVED 2026-09-09 (operator: slower at every size, memory saving lost under `torch.compile`; the paragraphs below are the record of why it existed).** Deleted: `MACEEncoder.configure_cueq_target_blocks`, `configure_target_blocks` on the interactions / `ConvTP`, the checkpointed block path in `conv.py`, `molix.md.NeighborList(target_block_size=)` with its interleaved padding and `live_edges_are_prefix`, the `--cueq-target-block-size` runner flag, and their tests.

**Receiver blocking saves memory only in eager (measured 2026-09-06).**
On a 21 400-atom cooled PEO melt, blocking cuts the eager peak from 86.8 to
19.1 GiB (−78 %) but the compiled peak only from 71.1 to 67.1 GiB (−6 %): the
compiled backward recomputes the blocks — AOT graphs show the block bodies
re-emitted in the backward — but does not serialise them, so several blocks'
`(E_block, weight_numel)` radial weights are live at once and the peak lands
back at the full convolution's. Two ways of asking the compiler to serialise
did not work: writing each block into its own rows of one buffer instead of
`cat` (Inductor proves the rows disjoint and reorders anyway; measured
identical 67.06 GiB, reverted) and `activation_memory_budget < 1` (crashes the
partitioner: `Node _assert_async didn't have 'val' metadata`). Consequence:
the only configuration that reaches 35 400 atoms on a 95 GiB GH200 is
**eager** blocked (1.95 s, 33.2 GiB); compiled blocked OOMs there. Do not
assume `torch.compile` preserves a checkpoint's memory intent.
Records: `lab-v3` experiment `peo-tg/polar-inference-bench`.

**Parity and cost against official mace-torch 0.3.16, both ports, -OCH3 PEO
melts (2026-09-08, `lab-v3` experiment `peo-tg/mace-scaling-bench`, molnex
c6be5652).** MatPES fp64: dE 8e-16 eV/atom, max|dF| 4e-14 eV/Å, max|dW| 5e-12 eV
— machine precision on 7 400 / 8 850 atoms. Polar fp64 (3 700-atom sub-cell,
the official model OOMs at 7 400 in fp64): dE −3e-9 eV/atom, max|dF| 2.9e-8
eV/Å, rms 6e-9 — not the fp64 floor; since the MACE body is bit-identical the
residual sits in the GTO / k-space part (k-vector set or summation order —
open, diagnose per term when machine precision is wanted). max|dW| 0.2 %: the
official virial strains the cell but reads `rcell`/`volume` unstrained. Virial
convention is `W_molnex = −W_official` element for element — the doubled
off-diagonals recorded by `polar-inference-bench` were wrong. fp32: molnex
fp32 sits 1.7e-4 eV/Å / 1e-7 eV/atom from its fp64 (fp64 energy accumulation);
the official fp32 Polar loses 8.6e-3 eV/atom in E at 7 400 (the earlier "0.3 %
disagreement" was theirs) and its forces blow up at 8 850 atoms (max|F| 75 eV/Å
vs 2.05). Root cause (2026-09-09, lab `mace-scaling-bench` plan-book, records
`.records/diag/`): not precision, not cuEq — the backward of the official
`SparseUvuTensorProduct` (Polar field blocks) forms d/dpair with
`torch.bmm((512, N, 1), (512, 1, 512))`, and that k = 1 batched GEMM is wrong on
GH200 / torch 2.14.0+cu130 from N = 8 209 atoms on (2³¹ / (511·512) = 8 208.03,
an int32 element offset; standalone it faults with `cudaErrorIllegalAddress`,
inside the model it reads garbage), fp32 and fp64 alike. Forward energies stay
right; the k-space electrostatics spreads the wrong gradient to every force.
molnex is unaffected because `molrep.interaction.polar.uvu` reassociates the
contraction and never builds the (N, 512, 512) tensor, so that autograd node
does not exist. The official e3nn path crashes (`CUDA_ERROR_ILLEGAL_ADDRESS`) at
7 080 atoms fp32 for its own reasons. Cost, fp32, 7 400→21 400 atoms: Polar molnex compiled 21→23 µs/atom
(α 1.06) vs official 75 µs/atom and OOM ≥ 17 700; MatPES molnex compiled 5.5
µs/atom vs official 7 µs/atom, official E+F+W 4.5× its own E+F.

**Compiled + receiver-blocked breaks above 21 400 atoms (2026-09-06, fixed).**
`Compiler(fullgraph=True, dynamic=False, donated_buffer=True)` on the blocked
Polar core fails in Inductor codegen at 2.1 M edges and above:
`ValueError: Scalar -2536860480 is out of range for type int32` inside
`triton_per_fused_add_cos_div_linalg_vector_norm_mul_neg_slice_slice_backward_...`
— a fused kernel over the GTO k-space path. The *same* sizes compile fine with
the full indexed convolution, so it is fusion-dependent, not a plain size wall.
Fixed by `d1e0b243`: the radial MLPs' first layer is evaluated from split
inputs, so the `(E, 1056)` concatenation that produced the index constant is
never built. Compiled blocked now builds at 21 400 atoms (0.64 s / 67.1 GiB).

**Receiver-block scale check (GH200, Polar fp32, 8850 atoms, skin graph,
2026-09-05).** The ordinary official cuEq convolution was 1.382 s/MD step at
63.90 GiB (`npt-2051047.out`). Checkpointed official receiver blocks reduced
that to 32.80 / 32.77 / 32.74 GiB for configured maxima 1024 / 2048 / 4096;
balanced spans were 984 / 1770 / 2950, fixed capacities 1.433 M / 1.390 M /
1.364 M edges, and one-step times 1.786 / 1.783 / 1.778 s
(`npt-2051210..12.out`). These include initialization and are **not** a
steady-state comparison with the historical Python-loop 1.83 s/step.
Maximum 4096 is the runner default. The radial MLP + official
TP block is non-reentrant-checkpointed; its derived MLP scalars are
non-persistent tensor buffers so the tiny CPU naive-cuEq trainable
forward/backward graphs pass `torch.compile(fullgraph=True)` (slow tests in
`test_conv.py` / `test_density.py`).

**10k re-review (2026-09-05).** See
[MACE-POLAR 10k review and multi-GPU design](mace-polar-10k-review.md).
The large observed `(N,512,512)` OOM was in Polar field `uvu.py`, not the
cuEq symmetric product. GEMM reassociation removes it without a new op.
The actually loaded conv `weight_numel` values are 2048 / 1280, not an
assumed 86-path width. Fixed-coordinate 8850-atom strict-fp32 E/F/virial:
full cuEq 0.410 s / 44.55 GiB, receiver-blocked 0.637 s / 16.49 GiB (before
fp64 energy pooling; jobs 2057105 / 2057021). Matmul `high` failed the
precision check; the runner now uses `highest`. Model graph-energy pooling
is float64 to avoid large-baseline fp32 scatter error. Single-system
multi-GPU must include global Fukui and GTO structure-factor reductions;
DDP alone does not split one periodic graph. The linked design is not an
implemented or validated distributed training backend.
After fp64 pooling, fullgraph GH200 E/F/virial is 0.258 s / 36.49 GiB full
cuEq (2057314) or 0.416 s / 28.45 GiB receiver-blocked (2057313); eager
blocked stays 0.636 s / 16.49 GiB (2057185). Full compiled versus fp64:
energy error 0.000369 eV, force RMSE 6.08e-5 eV/Å. These are warmed fixed-
coordinate measurements, not long-MD/force-training acceptance. At 0.5 fs,
the fastest kernel rate is only about 0.167 ns/day before MD overhead.

**Periodic Polar parity (found 2026-09-02, fixed 2026-09-03).** fp64 term
split on the same box (`bench5k/diag_components_float64.json`): E0 + local
interaction agreed to 2.7·10⁻⁵ eV with identical edge counts (285 546), but
k-space Hartree was +13.5 eV and the local electron energy +2.5 eV vs
official. A fixed-charge reproduction (`bench5k/repro_kspace.json`) showed
the reciprocal sum itself exact to 3·10⁻⁷ eV: both defects were the on-site
``i = j`` term, which the reciprocal sum already contains, being handled as
if the evaluator were the real-space one. (1) `GTOHartreeEnergy.forward`
added ``½Σωq²`` unconditionally after the `cell` branch — official adds it
only on the real-space path and *subtracts* it on the k-space path when the
flag is off. (2) `GTOFieldFeatures._kspace_features` never removed the
atom's own potential/field (official does when `field_si=False`, the dump's
setting). Fix: branch-aware self term in `hartree.py`; `GTOFieldFeatures(
include_self_interaction=False)` subtracting the closed-form ``r → 0``
limits `self_overlap_l0 = l0·eV_A/(σ_c√π)`, `self_overlap_l1 =
l1·δ·eV_A/(6√π σ_c³)` (dump: `[19.1995, 12.1428]`, `[2.3155, 1.1715]`);
`MACEPolarSpec.electrostatic_self_interaction=True` /
`field_self_interaction=False` wired through `MacePolar`. Pinned by
`tests/test_molpot/test_potentials/test_gto/test_{hartree,features}.py`
(large-cell k-space == isolated, lone-charge Madelung golden, flag
polarity) and `regressions/mace-polar-periodic-gto.py` (official fp64
goldens on the PEO box). Isolated path unchanged (water regression).

**Status.** active.

---

<!-- mol:note:topic:compile-no-graph-breaks -->
## [2026-09-03] Everything compiles, zero graph breaks (user rule)

**Rule.** Every hot path runs under `torch.compile` with
`torch._dynamo.explain(...).graph_break_count == 0`. Where a region truly
cannot compile, isolate it behind the narrowest seam and maximise the
compiled region; name the break here with reason + owner. Forbidden on a
hot path: `.item()` / `int(tensor)`, data-dependent shapes (boolean-mask
indexing, `nonzero`, tensor-derived `arange`), Python branches on tensor
values, per-call CPU→device tensor creation, pydantic reads. Dynamic sizes
= fixed capacity + mask (the `NeighborList` pattern). Every new hot-path
module ships an explain→0 test (pattern:
`tests/test_molzoo/test_mace/test_potential.py::test_energy_core_compiles_to_one_graph`).

**Baseline inventory (2026-09-03, CPU tiny Polar, naive cueq).** MACE local
backbone: 1 graph / 0 breaks. Polar MD step
(`molpot.forcefield.potentials.Potentials.calc_energy_forces_virial`, cell given): 51 graphs /
50 breaks from five sites — `gto/kspace.py:101-127 half_sphere_kvectors`
(`.item()` ×4 in a comprehension, 5 boolean-mask shapes),
`gto/features.py:204,214` and `gto/hartree.py:281` (`int(batch.max().item())`
for `num_graphs`), `molzoo/mace/encoder.py:303 validate_elements`
(`torch.unique` on a mask, traced on the first call),
`derivation/force.py:167-168` (leaf creation + `requires_grad=True` tensor
inside the traced region). Isolated path adds `gto/hartree.py:111-145`
(pair enumeration: `.item()` ×5, data-dependent mask). The fused cueq
kernels are `torch.library.custom_op` + `register_fake`, i.e. compile-safe.
Compile seam: `energy_core` (flat tensors) — `autograd.grad` stays outside
and gets its backward from AOTAutograd. Fix tracked by spec
`polar-compile-clean`.

**01-kgrid (2026-09-03) — fixed-capacity reciprocal grid.** Sites (1) and
(2) above are closed: `gto/kspace.py` now owns `KSpaceGrid` (plain class,
not a module, sized **once** on the first `bind`; later binds only
`check_capacity`, never resize — resize = build a new grid), and both GTO
modules take an explicit `num_graphs` (the `.item()` survives only on the
`None` branch, eager-only). The grid is an integer **half-box**
`n ∈ [-N_c, N_c]³`, first-nonzero > 0, with `N_c,i = ceil(f·‖a_i‖k_cut/2π)`;
`|k| < k_cut`, `k ≠ 0` are 0/2 multiplicative weights recomputed from the
live cell, guarded denominator before the division. Over capacity: eager
raises a `RuntimeError` naming `KSpaceGrid` + `capacity_factor`; in-graph
the weights become NaN (never truncation), both from the same
`floor(x) > N_c` predicate. Cost of the box vs the old data-dependent
half-sphere (`k_cut = 1.8467 Å⁻¹`): PEO box 46.36 Å cubic, M = 5318 →
**21437** at `capacity_factor=1.2` (N_c = 17), 12194 at 1.0 (N_c = 14);
10 Å cube 46 → 364. The half-box is still that size; the `(N, M)` plane-wave
matrix is **not** materialised. `structure_factor` and `kspace_phi_field`
use LAMMPS `ewald` / `eik_dot_r`: 1-D `exp(i n_d b_d · r)` tables of
length `2 N_c,d + 1` (the integer box already frozen on `KSpaceGrid`)
and a separable sum onto that Miller grid. Workspace is `O(N N_c²)`,
not `(N, M)` and not an arbitrary k-tile. Same SI V.6 sum; not PME.
Ordinary autograd through the einsums — no `custom_op`. MD compile warmup is `calc_forces`, not
virial (`calc_energy_forces_virial` strains `graphs.cell` and OOMs).
The Polar convolution stays on the official cuEq indexed
`ChannelWiseTensorProduct(method="uniform_1d")`; there is no local TP/CG
kernel. Large systems use fixed receiver blocks: the configured
`target_block_size` is an upper bound, and
`ceil(N / target_block_size)` receiver intervals are spread evenly to avoid a
tiny, heavily padded last block. The neighbour list packs equal-capacity edge
slices whose targets belong to one such interval, and cuEq runs each slice
with local `indices_out` / `size_out=actual_span`. This bounds
both cuEq's output workspace and the transient `(E_block, weight_numel)` radial
weights. The block's radial MLP + cuEq TP is checkpointed so backward
recomputes rather than retains every block's weight table. Configure model and
neighbour list with the same maximum `target_block_size` before compilation; `None`
keeps the ordinary full indexed convolution. `multipole_ewald.py` still builds a
dense `(N, M)` — owner **molpot elec**, not this seam. NVT callers may pass
`capacity_factor=1.0`; step time / peak memory are measured by 03-wire.
Same three defects (`.item()` sizing, boolean-mask slicing, half-sphere
mask) still live in `elec/kernels/multipole_ewald.py:221-245` /
`:352-370` — owner **molpot elec**, deliberately not migrated (its
FD-stress keep-set `enumerate_kvec_indices:247-255` is frozen to that
layout). **Two-way promotion trigger:** when `multipole_ewald` becomes a
consumer of the grid, *or* when `KSpaceGrid` gains any consumer **outside
`src/molpot/potentials/gto/`** first (its two in-package holders,
`GTOHartreeEnergy` / `GTOFieldFeatures`, do not count), `KSpaceGrid` moves
to `src/molpot/potentials/elec/lib/kvectors.py` and the keep-set is
re-derived — neither side may sit on it silently. Eager cost to measure in
03-wire, not assume: `check_capacity` reads three lattice norms back to the
host per call and eager `forward` calls it every step (≈ 5 calls per Polar
step at `num_recursion_steps=2`, ≈ 15 device→host syncs); the compiled
route pays none. The MacePolar-level `bind_cell` fan-out (one call instead
of `coulomb_energy.bind_cell` + `electric_potential_descriptor.bind_cell`)
is 02-seam's `CellBound` / 03-wire's wiring, not 01's.
**Post-chain inventory (03-wire, 2026-09-03, from `ExplainSummary.break_sites`).**
Cell-bound Polar `MacePolar.energy_core` (periodic, B = 1, first order):
**1 graph / 0 breaks** — the compiled core, installed on the mlip by
`MacePotential.compile_core` and dispatched by `_write_energy`; it survives
every `ForceField.to_potentials()` rebuild because it lives on the shared
`nn.Module`, not on the `Potentials` container. Whole MD step
(`Potentials.calc_energy_forces_virial`): breaks **only** at the named
autograd seam `src/molpot/derivation/force.py` — `autograd_forces_and_virial`
leaf creation at `:179-180` (`pos.detach().requires_grad_`, the strain
`zeros(..., requires_grad=True)`) and `torch.autograd.grad` at `:189-194`
(CPU and GH200 regression `regressions/polar-compile-clean-03-wire.py`
report three sites, all in that file: `:179`, `:180`, `:189`; the count
`graph_break_count` is `graph_count − 1`, not `len(break_sites)`). Retained
by Q3: the seam is the *caller* of the compiled core; AOTAutograd supplies
the backward. Owner: molpot derivation. Buffer donation is off process-wide
(`register_inductor_fallbacks` pins `torch._functorch.config.donated_buffer =
False`), so the compiled core also answers the `create_graph=True` backward a
force loss needs; `Compiler(donated_buffer=True)` opts back into donation and
with it `create_graph=False` (MD only). Before that pin, grad-enabled use
raised "compiled with non-empty donated buffers which requires
create_graph=False" (spec `cuet-force-doublebackward`). Baseline 51 graphs / 50 breaks → sites (1)
`kspace.py`, (2) `features.py` / `hartree.py` `num_graphs`, (4)
`encoder.py:303` element gate (now eager-only behind
`torch.compiler.is_compiling()`) are closed; (3) the isolated all-pairs
enumeration `gto/hartree.py:96-151` stays (owner `polar-compile-clean-04-pairs`),
`multipole_ewald.py:221-245` / `:352-370` stays (owner molpot elec), (5)
`force.py` is the design seam above. Fixed on the way: the eager NVT-MLIP
MD path was dead under `MD.run`'s `no_grad` (`autograd_forces_from_energy`
summed the energy outside `enable_grad`; `Potentials.calc_forces` never
re-enabled it) — only the NPT path had worked. Process note: `.claude/specs/INDEX.md`
marked `md-forcefield-molpy-reexport-05-wire` `[code-complete]` while its
ac-007 (no `PotentialForceField` / `energy_scale=` in the user docs) was
still pending and factually unmet; 03-wire paid that debt and flipped it —
**`[code-complete]` is not acceptance.**

**Inductor fused-scatter leak (found 2026-09-04, GH200, torch 2.12.0+cu130).**
The first GH200 parity run of the compiled Polar core (job 1971898, inductor,
fp64, PEO box) gave `dE = 14.238975 eV` with forces exact to 6e-14 and virial
to 9e-11; `aot_eager` was exact (dE = 0.0, jobs 1972031), every submodule
compiled alone with inductor was exact (job 1972127, 33 modules, worst 2e-14),
E0 alone was exact. Minimal reproducer (job 1973858): `out[0] += v*0.7 + 1.3`
over N rows via `index_add_` (= `scatter_sum_compile_safe`), inductor vs
eager — error = `(roundup(N, 32) − N) × 1.3` for both fp32 and fp64
(N=21 → 14.3, 129 → 40.3, 4097 → 40.3, 4995 → 37.7; N a multiple of 32 → 0).
Mechanism: inductor lowers `index_add_` → `index_put_(accumulate=True)` →
`ir.Scatter(atomic_add)` and inlines the pointwise producer into the scatter
loader; the load is masked (padded lanes read 0) but the fused epilogue
`f(0) = 1.3` is still atomically added for the padded lanes. In Polar the
producer is `scale·node_es + shift` (f(0) = shift): 29 padded lanes × shift
= 14.24 eV. Any per-graph / per-node scatter whose fused producer has
f(0) ≠ 0 is affected under `torch.compile(backend="inductor")` and under
AOTInductor (same lowering); `aot_eager`, eager and the CPU tests never
see it, which is why the chain's explain/aot_eager tests passed. Forces are
exact because AOTAutograd's backward is a separate gather kernel.
**Fix (validated, jobs 1974354 / 1974355):** realising the scatter source
before the atomic `index_put` does *not* help (job 1974354: unchanged
37.7 / 14.24); what works is keeping `aten.index_add` an **extern ATen
kernel** under inductor — `molix.compile.register_inductor_fallbacks()`
removes the `index_add` decompositions from `torch._inductor.decomposition`
and `torch._decomp` (clearing the lru caches so it also holds after an
earlier compile in the process) and registers `make_fallback` for
`index_add` / `index_add_`. With it: probe error at round-off (fp64 N=4995
9.8e-11, N=21 0.0; fp32 3.9e-3), PEO-box compiled core dE **0.0**, dF 5.7e-14,
dW 1.9e-10. It runs from `Compiler.__call__` and from the AOTI packager
(`Exporter._package`), is process-global and idempotent; a direct
`torch.compile` caller must call it first. Rule: **never lower a
molnex scatter to inductor's fused atomic Scatter** — every compiled
numerical parity check must run under `backend="inductor"` on the target
device, not only `aot_eager`. Upstream reproducer = the probe above
(torch 2.12.0+cu130, sm_90).
**03-wire GH200 (2026-09-04, job 1976727, n186, 4995-atom 46.36 Å PEO, fused cueq, `Compiler(dynamic=True)`).**
fp64 compiled vs eager: `dE = 0.0`, `max_dF = 4.97e-14`, `max_dW = 6.29e-11`, `n_edges = 285546` (job 1974822 had `max_dW = 1.087e-10` against the same 1e-9 virial contract; |W| ~ 5200 eV, ~2e-14 relative). fp32 compiled: mean step **0.190 s** (eager baseline 0.38 s), peak **25.87 GiB** (eager 28.3 GiB; the larger k-grid did not OOM), recompiles after warm-up **0** (live E 285546 → 285446). Official goldens unchanged: `regressions/mace-polar-periodic-gto.py` diffs ~1e-5 eV (`atol` 1e-3), `mace-polar-official-water.py` `dE = 1.3e-7 eV`.
**Same-job eager vs compiled (2026-09-04, job 1984979).** `regressions/polar-peo-compile-compare.py`: fp64 compiled vs eager `dE = 0.0`, `max_dF = 5.7e-14`, `max_dW = 1.08e-10`. fp32 molnex eager 0.395 s / 39.24 GiB, compiled **0.142 s / 25.87 GiB** (2.78×), 0 recompiles. Official fp64 goldens unchanged (`denergy = +3.0e-5` eV).
**Kernel-only official PolarMACE vs molnex E+F (2026-09-04, job 1987847, n412).** Neighbour graph built once; official `PolarMACE.forward` (autograd forces, no ASE rebuild) vs molnex `calc_energy_forces_virial`. fp32:

| path | mean E+F kernel | peak HBM | vs official |
|---|---|---|---|
| official PolarMACE | 0.429 s | 25.24 GiB | — |
| molnex eager | 0.415 s | 39.18 GiB | ~1.0× |
| molnex compiled | **0.175 s** | (see 1984979: 25.87 GiB) | **2.45× official** |

fp32 accuracy vs official on the same coordinates: `dE = −0.75` eV on `|E| ~ 2.83×10⁶` (relative 2.6×10⁻⁷), `max|dF| = 3.6×10⁻⁵` eV/Å (`Fmax` 27.4395 vs 27.4395). ASE `get_forces()` (~1.06 s) is **not** the model — it rebuilds the neighbour list every call.
Post-impl review adjustments (2026-09-03): the head-room knob is a
constructor keyword `kspace_capacity_factor` on both holders (the earlier
"swap `mod.kgrid`" advice was a leaked lifecycle; plumbing it up to
`MacePolar` / `MACEPolarSpec` is 03-wire's NPT surface); `k_cutoff` has one
home (`kgrid.k_cutoff`, the holders' properties delegate); the holder glue
(`bind_cell` + `_apply`) lives once in a private mixin in `hartree.py`
because it had exactly two call sites; holders import only public names
from `kspace.py`.
**Retained break (named, asserted):** the isolated-molecule batched
all-pairs enumeration `gto/hartree.py:96-151` (`_complete_directed_pairs`
/ `cached_complete_directed_pairs`: `.item()` ×5, `torch.equal`, boolean
masks) is untouched — a whole-batch mask would inflate the training-path
pair count from `16·N(N−1)` to `(4Bn)²` (B = 128, n = 20: 128×). Owner:
spec `polar-compile-clean-04-pairs`. Sketch: block-diagonal per-graph
enumeration at a construct-time `max_atoms_per_graph` capacity,
`O(B·n²·copies²)`, all indices from `arange` arithmetic, overflow raised at
bind — the `NeighborList` fixed-capacity idiom. Unit tests assert this
branch *still* breaks so it cannot silently become a contract.

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

**Rule**: `FrameStoreReader._map_column` stays mmap. Warm, mmap wins 5.1× (seq) /
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


## [2026-08-14] Streaming scheduler integration — sharp edges of record

<!-- mol:note:topic:scheduler-integration-edges -->

**Rule.** The `("graphs","task")` stamp on block-collated batches is the
**view's global** task code (never block-local `reader.task_codes` — a block's
local `(0,1)` may be global `(1,2)` after the dictionary merge). Each
`FrameStoreDataset` merges its OWN dictionary, so two views (e.g. train vs val)
can assign the same integer to different tasks: a `TaskRouter` shared across
loaders must be fed views whose dictionaries agree, or the caller must remap.
Pinned by `regressions/streaming-data-tiers-04-scheduler-integration-03-plan.py`
(discriminating local!=global fixture; the cross-view case documented in its
docstring as a deliberate sharp edge).

**Open routes** (named, not silent): `/mol:fix` scoped to `molix.batch.task`
(`BatchTask.execute` annotated `dict` while the contract is TensorDict — one
pre-existing ty warn per collate route); `FrameStoreDataset.__getstate__` should
drop the LRU pool + cached tables so a spawn worker re-opens by path (worker
pickling currently ships mmap'd column bytes by value — same shape as the
pre-existing `_PackedCollateFn` debt, worse for the streaming tier;
`num_workers=0` unaffected); molnex `streaming-data-tiers-06-ddp-plan`
(cross-rank plan slicing) and molhub `streaming-data-tiers-05-multinode`
(ledger writer election) are the two named successor specs.

**Status.** Chain closed 2026-08-14: 01-tables 57f2adc (FrameLabelTable +
sampler generalization + R1 bench: no crossover below 1e8, code-form chosen,
85% of plan footprint is representation-shared planning machinery →
unit-granularity planning is the real lever), digest-core revision 180332f
(task_names covered), 02-view bc75a9d (FrameStoreDataset), 03-plan eb39d8d
(FrameStorePlanSource + block collate + visit-order handoff).

## [2026-08-30] LAMMPS interface — fixed_graph kept, C++ toolchain fixed

<!-- mol:note:topic:lammps-iface-post-chain -->

**`Contract::fixed_graph` stays declared-but-unimplemented.** The 01-meta rule
was "implement or remove by schema bump before merge". Resolution: **KEEP**.
The binding consumer is the operator's espaloma-style learnable-classical-FF
roadmap (`.claude/notes/learnable-classical-ff.md`): a parameterized Class-I
force field runs on the **fixed covalent topology**, not a per-step cutoff
graph, so its LAMMPS impl consumes `bond_index`-shaped inputs with no neighbor
rebuild — a different runtime-contract family by construction. Until that impl
lands, `registry.cpp`'s exhaustive switch rejects `fixed_graph` loudly (tested:
`registry_rejects_fixed_graph`); removing the enum value would force a schema
round-trip for a consumer already on the roadmap. Implementation arrives as its
own spec chain (`lammps-iface-fixed-graph`), not as a side effect.

**C++ toolchain for `interface/`.** `interface/.clang-format` = Google base at
its native 80 columns, `SortIncludes: Never` (include order is deliberate:
LAMMPS protocol-role grouping, torch/ATen below molnex). Tree formatted
one-shot with clang-format **22.1.5** (the PyPI `clang-format` wheel; local
install is the same package); CI's interface job gates `--dry-run --Werror`
pinned to `22.1.*` because wrap decisions drift across clang-format majors.
Non-ASCII (em-dash) comment lines wrap by byte count — accept clang-format's
choice, do not hand-wrap. `mol_project.language` stays `python`: `build.check`
is the Python gate; C++ hygiene rides the interface CI job + this file.

---

<!-- mol:note:topic:tensorframe -->
## `TensorFrame` — the typed batch view, and the torch 2.14 bump it rode in on (2026-09-07)

**Two shapes, two names.** At rest a molecule is a molpy ``Frame`` (`mp.Frame`);
in the model it is `molix.batch.TensorFrame` (`mn.TensorFrame`) — a
`tensordict.TypedTensorDict` whose declared fields *are* the post-collate
schema. `atoms` is the only required namespace; `edges` / `graphs` / `bonds` /
`angles` / `propers` / `impropers` are `NotRequired`, because `edges` is built
downstream by `NeighborList` and the term namespaces depend on what the frames
carried. `from_tensordict(td)` wraps **without copying** — asserted by
comparing every leaf `data_ptr` against the `collate_molecules` output — so the
storage↔model seam stays copy-free. No methods, no storage: it is a view.

`TypedTensorDict` is the landed form of pytorch/tensordict#1768; there was
nothing to wait for.

**ty suppression is deliberate.** `TypedTensorDict`'s metaclass is
`@dataclass_transform`-decorated, so ty reads the class under PEP 681 rules
where PEP 655 `NotRequired` is not a legal field form. The annotation is
correct (it is what populates `__optional_keys__` at runtime); the six
`# ty: ignore[invalid-type-form]` comments name the protocol mismatch. Do not
"fix" it by making the namespaces required.

**Compile evidence** (torch 2.14.0, tensordict 0.14.1), plain `TensorDict` vs
`TensorFrame` at every real seam that takes a batch:

| seam | plain | TensorFrame |
|---|---|---|
| `PiNet.forward` | 1 graph, 0 breaks | 1 graph, 0 breaks |
| `MacePotential.forward` (matpes) | 8 graphs, 7 breaks | 6 graphs, 5 breaks |
| `MacePotential.forward` (polar) | 5 graphs, 4 breaks | 5 graphs, 4 breaks |

Outputs are equal leaf-for-leaf in all three. The MACE breaks are the
pre-existing `torch.autograd.grad` seam at `molpot/derivation/force.py:137`
(`trace_autograd_ops=False`), not the container — which is why the compiled
MACE seam is the flat `energy_core`, not `forward`.

**Env bump this required.** `torch 2.12.1+cpu → 2.14.0+cpu`,
`tensordict 0.13.0 → 0.14.1`, `cuequivariance(+torch) 0.10.0 → 0.11.1` in the
shared `.x86_64` env. torch 2.14 headers **require C++20**: building
`molix/op` at `CMAKE_CXX_STANDARD 17` fails inside `ATen/ExpandUtils.h`
(`'sizes' is not a member of 'at::symint'`), so the op CMakeLists is now `20`
and `libmolnex_opLib.x86_64.so` was rebuilt. pyproject pins `torch>=2.14`,
`tensordict>=0.14`, `cuequivariance*>=0.11`.

**Both arches are now on torch 2.14.** `.aarch64` was upgraded and rebuilt in
one GH200 job (2137657) and verified on a GH200 node (2139809, exit 0):
torch 2.14.0+cu130 with `cuda True`, tensordict 0.14.1, cueq 0.11.1,
molrs aarch64, `molix._lib_loaded True`, `torch.ops.molix.get_neighbor_pairs`
registered, and `TensorFrame` importing with the right required/optional keys.
Recipe: `.scratch/polar_npt/upgrade_aarch64_torch214.sbatch` (upgrade + molrs +
op lib, in that order — the rebuilds must follow the torch change, never
precede it) and `verify_aarch64_torch214.sbatch` (probe only, no rebuild).

Two traps that cost a job each: the inherited probe asserted
`from molrs import Record`, which molrs 0.14 no longer exports — a stale probe
fails a job whose real work succeeded. And pip's `Successfully installed` line
reported `cuequivariance-ops-torch-cu13-0.10.0` while the dist-info on disk read
`0.11.1`; trust the dist-info, not the summary line.

---

<!-- mol:note:topic:frame-store-rename -->
## `Block*` → `FrameStore*`; the format id stays `molix.block/1` (2026-09-08)

**Why.** `Block` already meant a molpy `Block` (a Frame's sub-table). molnex
never exported a bare `Block`, but every `Block*` type read as if it did. The
Python surface is what collided, so the Python surface is what changed:

| was | is |
|---|---|
| `BlockWriter` / `BlockReader` / `BlockManifest` | `FrameStoreWriter` / `FrameStoreReader` / `FrameStoreManifest` |
| `BlockDataset` / `BlockFrameView` | `FrameStoreDataset` / `FrameStoreView` |
| `BlockPlan` / `BlockPlanSource` / `BlockCoverage` | `FrameStorePlan` / `FrameStorePlanSource` / `FrameStoreCoverage` |
| `BlockNotReadyError` / `BlockSchemaMismatchError` / `BlockSplitMismatchError` | `FrameStoreNotReadyError` / … |
| `molix.store.block` / `block_view` / `molix.batch.block_plan` | `molix.store.frame_store` / `frame_store_view` / `molix.batch.frame_store_plan` |

**What deliberately did not change**: the format id `molix.block/1`, the
manifest keys, and lowercase "block" as the name of the on-disk unit. That id
is a wire contract already written into every store MolHub has produced;
changing it is a format revision, not a rename. So: a `FrameStoreWriter`
writes a frame store, whose on-disk form is one `molix.block/1` block.

**Downstream**: molhub funnels its coupling through exactly one import,
`src/molhub/conveyor/prepare.py:446` (`from molix.store.block import
BlockWriter, SamplingUnit`), plus `molix.batch.neighbor.NeighborList` in
`src/molhub/dataset/neighbor.py`. That line was already broken by the
`SamplingUnit` deletion; the rename makes the module path stale too. Fixing it
is one line: `from molix.store.frame_store import FrameStoreWriter`, then
`writer.append(frames, key=…, task=…)`.

---

<!-- mol:note:topic:frame-collate-fallout -->
## Fallout of the Frame-native collate — three live bugs, and the rule behind them (2026-09-08)

Deleting `TargetSchema` / `FrameSample` / `SamplingUnit` left more than stale
imports. Found and fixed while porting the consumers:

1. **Integer meta scalars were being cast to float.** `collate_molecules` sent
   every frame `meta` value through `float(...)`, but
   `TaskRouter.route` **refuses a floating task label** rather than truncate one
   silently — so multi-task routing was broken for every frame-fed batch. Fixed
   with `_meta_scalar` (integral in, integral out), test in
   `tests/test_molix/test_batch/test_collate.py`. **Producers of
   `graphs.total_charge` must now emit a float**: `total_charge=0` collates to
   int64 and the EEM solver wants the dtype it solves the charges in.
2. **`DataLoaderProfiler` raised on use.** `profiler/dataloader.py` called
   `_ProfilerCollate(self.target_schema, self.batch_tasks)` against a
   one-argument constructor, reading an attribute that no longer existed. No
   test covered it.
3. **`molpot` never enumerates its own pairs** (since `composable-a0-03b`), so
   every batch chain feeding an OPLS-AA pair style must place a list. Since
   `composable-a0-debt-06` (2026-09-08) the field declares it: mount
   `batch_tasks=[ff.neighbor_list(pbc=...), TotalCharge()]` (or apply the same
   two tasks at a collate choke point that bypasses `DataModule`) — the radius
   comes from the field's own pair styles, so nothing restates it; a bare open
   `pair:coulomb` is refused (every pair), declare `cutoff=` on it.

**The rule this teaches**: deleting a declaration layer moves the contract from
"declared" to "derived", and every derived value silently acquires a dtype and a
default. Grep for the deleted symbol, then grep for what *read* what it produced.

Ported consumers: `scripts/mm_param_learning/{class_i_fit,align,espaloma_mm_protocol_train,scale_audit}.py`
(the fourth was in nobody's ledger) and `scripts/store/bench_block_read.py`.
`Frame.meta` returns a **copy** — `frame.meta["x"] = 1` is a silent no-op, so
adding a meta scalar means rebuilding the frame (`_with_meta`).

**Still open**: `molhub` cannot write a store. Its conveyor flattens frames into
molnex's storage column vocabulary (`RawUnit`) and calls a writer API that no
longer exists. The fix is *not* an import change: `encode()` already takes
`Sequence[Frame]`, so the flattening is now molnex's job and molhub should hand
its frames to `FrameStoreWriter.append(frames, key=…, task=…)` directly. Only
three lines consume the columns (`prepare.py:471-472`, `encoder.py:147,156`),
but `RawUnit` and its tests are built around them — a molhub design change,
owned there, not here.
