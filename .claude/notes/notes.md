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
- Gate: `python scripts/check_test_mirror.py --strict-spine` (renamed from
  `--strict-pinet` 2026-10-15; `SPINE` covers the whole model zoo).

**PiNet spine layout (superseded 2026-10-15, chain `zoo-unify`):** the
`molrep/interaction/pinet/` subpackage is gone; the PiNN layers live with the
model in `molzoo/pinet/layers.py` (PI/IP, PIX/Scale/Dot, `ResUpdate`,
`GCBlock`, `OutLayer`), molrep keeps only the generic
`embedding/mlp.py::FFLayer`, and `molzoo/pinet/` is
`{spec,encoder,layers,potential,properties}.py` in the MACE shape
(`PiNetPotential(Potential, PiNetEncoder)`, flat keys). Edge geometry has one
owner, `molrep/embedding/geometry.py`.

**Status.** active (model spine done; full-repo mirror is incremental).

---

## Force derivation — one op (2026-09-11; supersedes "one head" and "dual explicit backends")

**Decision (operator, spec `derivation-head-04-molix-derivative`, D-a..D-e).**
One derivative op across the project, at the molix base layer:
`molix.autograd.Derivative(energy, *, of="graphs.energy", wrt=(), outputs=None)`
— `energy(frame) -> frame` writes `of`; `wrt` are TensorFrame keys (dotted
strings or tuples, no variable classes): `atoms.pos → atoms.forces` (scale −1),
`graphs.strain → graphs.virial` `(1, 3, 3)` (ε maps `pos`, `cell` and the
lattice remainder by `v ↦ v (I + ε_sym)ᵀ`; symmetrised gradient; must follow
`atoms.pos`), any other column `→ <ns>.<name>_grad`; `outputs=` renames. The
op owns every leaf (`molix.autograd.leaf`), rebuilds `pair_diff` / `pair_dist`
once from the perturbed columns (`shifts` = `edges.shifts` or `pair_diff −
Δpos`, detached), one `torch.autograd.grad`, restores the caller's objects,
detaches `of` / `atoms.energy` iff grad is off; a potential is handed in as
`pot.energy`, never as the object. Every potential (`MacePotential`
family, `PiNetPotential`, `Potentials`) exposes `energy(frame)` and takes
`wrt=()` at construction; `forward` is the op built in `__init__`. `MD`
composes `Derivative(pots.energy, wrt=("atoms.pos"[, "graphs.strain"]))` at
`bind_work`, evaluates it once eagerly and applies `set_compiler`'s compiler
to it. Deleted: `DerivativeHead`, `ForceReadout` / `EnergyReadout`,
`StressDerivation`, `compile_core` / `CoreCompilable`, `compute_forces`,
`energy_forces` raw adapter, `calc_forces` / `calc_energy_forces` /
`calc_energy_forces_virial`, the analytic classical core `_classical_core`
(D-c), `Potentials`' `pos=` overrides and `Mapping` input. Measured (torch
2.15 nightly CPU): op inference 1 graph / 0 breaks for `("atoms.pos",)` and
`("atoms.pos", "graphs.strain")`; training breaks once at
`molix/autograd/variables.py::leaf` (pytorch#196653). Residue: the
`molix.deploy.AtomEnergyAdapter` keeps its own leaf (`molix.autograd.leaf`) because the partitioned
`local_mask` energy is not frame-in/frame-out; molzoo specs still name
`grad_force_pass` (auditor 📝).

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

**Context.** Multi-arch HPC needs
both `x86_64` and `aarch64` artifacts in the same tree. A stale
`libmolnex_opLib.aarch64.so` (wrong torch ABI) silently disabled native
neighbor ops and MD fell back to `bin=0.0`. Login-node `maturin` then overwrote
in-tree `molrs/_lib.abi3.so` with x86-64, so GH200 `import molix` died even
after the molnex op rebuilt. Probe without GCC 14 `libstdc++` on
`LD_LIBRARY_PATH` failed with `CXXABI_1.3.15`.

**Rule**: CMake `OUTPUT_NAME = molnex_opLib.${CMAKE_SYSTEM_PROCESSOR}`; loader
loads only the arch-tagged path (no untagged fallback). The library links
LibTorch only — no Python, no `torch_python` (2026-10-16: the old reason,
`_wrap_outputs`, lives in `libtorch_cpu` in torch 2.14+; a Python-linked library
cannot be loaded by molnex's C++ deployment runtime, which loads it for every
package that calls `molix::*`). It exports `molnex_op_library_requirement()`
= `molix_ops==<n>`, bumped with any schema change in `molix_library.cpp`.
**Every `.aarch64` env/torch
change rebuilds ALL native libs together on GH200 — never reuse a stale
`.so`, never rebuild molnex op without also re-asserting molrs aarch64.**

1. `molrs/_lib.abi3.so` via `maturin develop --release` with
   `RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu` and
   `CARGO_TARGET_DIR=molrs-python/target-aarch64`. The in-tree file is
   untagged: login rustc emits x86-64 into the same path. Keep
   `_lib.abi3.so.aarch64` / `_lib.abi3.so.x86_64` copies. Never run
   molrs maturin on the login node if GH200 still needs that tree.
2. `libmolnex_opLib.aarch64.so` from `csrc/molix` (out of source, `build/csrc-aarch64/molix/`;
   `src/molix/op/` until 2026-10-16) against **this** interpreter's
   torch. Export `TORCH_CUDA_ARCH_LIST=9.0` (PyTorch cmake ignores
   `CMAKE_CUDA_ARCHITECTURES`). GH200 CUDA arch is `90`, not the cache's `75`.
3. Runtime `LD_LIBRARY_PATH` = GCC 14.3 `lib64` + `torch/lib` + CUDA `lib64`
   **before** `python` starts (uv CPython is linked to system libstdc++).

Recipe: `.jobs/op-rebuild-aarch64.sbatch` (2026-10-15: cmake against
`.venv-aarch64`'s torch, CUDA arch 90, then a load + CUDA `scatter_sum`
check; the earlier "ABI mismatch" was only the missing GCC 14 `lib64` on the
runtime `LD_LIBRARY_PATH`). The older `.scratch/polar_npt/rebuild_op.sbatch` is gone. Missing/ABI-mismatched `.so` is an environment
error for MD, not a reason to skip the rebuild.

**Supersedes**: "warn and continue with pure-Python NeighborList" as the GH200
workflow; "rebuild only molnex op, leave molrs".

---

<!-- mol:note:topic:cue-kernels-only -->
## cuEquivariance on its kernels — no fallback on CUDA (2026-10-15, operator rule)

**Rule.** On CUDA molnex never runs cuEquivariance's pure-PyTorch path: every
cue operation is built through `molrep.cueq`, which pins the kernel method
(`Linear` / `FullyConnectedTensorProduct` → `fused_tp`; `ChannelWiseTensorProduct`
/ `SymmetricContraction` / `SphericalHarmonics` / `SegmentedPolynomial` →
`uniform_1d`). There is no `use_fallback` parameter or spec field anywhere and
no `method="naive"`. On CPU cuEquivariance swaps in its own naive path — that
is accepted; there are **no runtime checks** (no device guards, no warning
interception) and CPU tests stay on CPU. `.double()` works: the four ops with a
`dtype` argument rebuild their kernel submodules on a dtype change.

**Why it mattered beyond speed.** Under `use_fallback=None` cue's `Linear`
already ran `naive` — 27 sites in molnex executed the PyTorch path by default.

**Open debt.** `skip_tp` (one-hot `89x0e` operand, density blocks) was pinned
to `naive` because `fused_tp` is ~19x slower there (33 vs 1.7 ms at 193
atoms); it now runs `fused_tp`. Candidate fix: express it as an
`indexed_linear` per-species map. Done 2026-10-16 as `ElementLinear` (one
`indexed_linear_B` per path on CUDA) and replaced on 2026-10-17 by a per-atom
gather on every device: the kernel cannot be captured into a CUDA graph (see
the GH200 verification paragraph of the 2026-10-16 zoo-unify entry).

**Related finding (same day).** MACE checkpoints' symmetric-contraction
weights live in MACE's Clebsch-Gordan phase convention, which matches
`cue.O3` for every coupling among `l <= 1` and differs in sign beyond
(`molzoo.mace.coupling.MaceO3`). MACE-POLAR-1-L, the one product emitting
`2e`, was wrong by 0.25 eV/Å under `cue.O3`; with `MaceO3` and the module's
own float64 projection (the checkpoint's projection is dropped by
`CheckpointRemap`) it agrees with the official model to 1e-7 eV/Å.

<!-- mol:note:topic:cueq-use-fallback -->
## cuEq `use_fallback` split by force backend (2026-08-08) — SUPERSEDED by `cue-kernels-only`

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
`molix.data.cache.PackedCache` — the single-`.pt`, schema-inferred, run-scoped
packed cache — is gone, along with `MmapDataset` / `CachedDataset` and the
`_PackedCollateFn` route in `DataModule`.

**Why it could go.** MolHub owns dataset production and already writes only
`molix.block/1` (`molhub/loader/materialize.py`: `_BLOCK_FORMAT`,
`_WRITER_MODULES = ("molix.data.block", "torch")`); it never imported
`PackedCache`. Inside molnex `src/`, the only production writer left was a
profiler helper that round-tripped samples through a temp file purely to
"exercise the same code path real workflows do". Everything else that wrote
one was a test or a `regressions/` script.

**What survived, and why.** The *packed payload* is the real contract and it
stays: `FrameStoreReader.frame_payload` produces it, `PackedView` wraps it, and
`collate_packed` turns it into a batch without materialising per-sample dicts.
`PackedCache` was only one on-disk spelling of that payload, and the worse one
(no manifest, no seal, no digests, no resume, no splits). The unpacker moved to
`molix.data.block_view._unpack_packed_sample`; `SubsetDataset.packed_view()`
is defined explicitly rather than inherited, because `__getattr__` forwarding
would silently hand back an **unmapped** view.

**Consequences that are features, not losses.** Blocks carry no neighbour
graph by design, so the cutoff graph is built downstream by
`molix.data.NeighborList` — the cache's `edges` bucket, the sampler's
`max_pairs` budget and `avg_num_neighbors` had no producer left. The
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
`DerivativeHead` (formerly `grad_force_pass`) resolves `create_graph` from the *ambient* grad mode
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
`torch.bmm((512, N, 1), (512, 1, 512))`; for k = 1 torch 2.14 does not call
cuBLAS but its own Triton override `_bmm_outer_product_kernel`
(`torch/_native/ops/bmm_outer_product/`), whose store offset
`pid_b * stride_ob` is int32 and wraps from N = 8 209 atoms on
(2³¹ / (511·512) = 8 208.03; standalone `cudaErrorIllegalAddress`, inside the
model silent garbage), fp32 and fp64 alike — a PyTorch bug that MACE exposes.
`TORCH_DISABLE_NATIVE_JIT=1` (cuBLAS fallback) makes the official fp32 forces at
8 319 atoms agree with molnex fp64 to 2e-4 eV/Å again. Unfixed upstream as of
2026-09-09 (kernel unchanged on `main`; nightly 2.15.0.dev20260908+cu130
reproduces, job 2198256). Fix submitted upstream as
https://github.com/pytorch/pytorch/pull/196431 (int64 offsets + regression
test); issue draft in the lab experiment's
`knowledges/pytorch-issue-bmm-outer-product-int32.md`. Forward energies stay
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

## [2026-09-18] A unit is the dataset's to declare — and `kcal/mol` is two units in molpy

`ClassIPrepare` defaulted its source unit to eV, which is OMol25's and was right
only where the transform started. THEMol publishes kcal/mol, so that default
multiplied every label by 23.06 — silently, the result still a plausible energy
(caught only because `residual std 36.35` did not match `barrier 7.014`).

**The fix is not a better default; it is reading what the dataset says.** MolHub
datasets already declare `@dataset(units={"energy": ...})`, so
`ClassIPrepare.from_dataset(dataset)` reads it and `source_unit` has **no**
default: built without one, the transform refuses to `convert` (bond / label /
topology still work, and those are the paths that never needed a unit).

Two things this needed on the MolHub side:

1. **A loaded dataset must be able to state its own units.** `THEMolTorsionScan`
   declares them, but `load()` returns a *different* object
   (`THEMolShardDataset`) which dropped the manifest, so a consumer holding the
   loaded dataset had to restate the unit — which is where drift comes from.
   `THEMolShardDataset` now carries it, `SubsetDataset.manifest` reads through to
   its parent (same reason `source_id` is derived), and the narrow protocol
   `DeclaredDataset` names the capability next to `IdentifiedDataset`. OMol25
   needed nothing: its `load()` returns `self`, so it never lost the manifest.
2. **`kcal/mol` and `kilocalorie_per_mole` are not the same unit in molpy.**
   `units.parse("kcal/mol")` is energy per *amount* (`L² M T⁻² N⁻¹`);
   `units.kilocalorie_per_mole` is the per-particle energy chemistry means
   (`L² M T⁻²`). Converting between them raises `UnitsError: dimension mismatch`,
   so reading a declaration and handing it to pint is not enough — MolHub's
   chemistry spelling is bridged to molpy's registry name by
   `MOLHUB_ENERGY_SPELLING`, which covers all of MolHub (`eV`, `kcal/mol`) and
   refuses a third spelling by name rather than guessing. Note also that
   `units` is not callable: parse with `units.parse(expr)`.

Verified: declared `kcal/mol` → scale 1.0, declared `eV` → 23.0605, unknown
spelling and absent manifest both refused, and `convert` without a declared unit
refused.

## [2026-09-18] The Class-I step is CPU-bound on per-batch bond inference

Measured on a GH200 120GB, one process per cell, batch fill **measured** from
`graphs.mask.sum()` rather than inferred from the capacity:

| cell | mol/batch | fill | ms/step | peak GiB | molecules/s |
|---|---|---|---|---|---|
| small / 262144 atoms | 16,503 | 80 % | **6332** | 5.01 | **2,606** |
| small / 524288 | — | — | — | — | DataLoader worker died |
| small / 1048576 | — | — | — | — | DataLoader worker died |

**~379 us of serial work per molecule**, from 6332 ms for 16,503 molecules
against the ~85 ms fixed cost measured at near-zero fill.

**Where it goes.** `FrameCollator` calls `ClassIPrepare.execute(frame)` per frame
per batch, and `ClassIPrepare.bond()` opens with `if "bonds" in frame: return
frame` and otherwise calls `infer_bonds(numbers, xyz)`. OMol25 ships **no bonds**,
so that early return never fires: bond inference plus valence enumeration re-runs
for every molecule every time it appears in a batch — every epoch. At 16.5 k
molecules per batch that *is* the step, and it is why the two larger capacities
killed their DataLoader workers rather than reporting a number.

**This is the performance half of a correctness fix that already landed.** The
topology work established that topology must not be derived per batch, because
the distance criterion misreads a stretched bond (butane C-H at 1.09 -> 1.30 A
turns 13/24/27/0 into 12/21/24/**1**). The correctness half shipped; the cost did
not. Bonds are still derived per batch instead of once.

**Do not read a capacity sweep taken at low fill.** An earlier sweep (job
2627024) timed 16384 / 65536 / 131072 against a 48-molecule split whose padder
budget was 10,375 graphs — a 0.5 %-full batch — and concluded that 8x the
capacity costs 1.07x the time, i.e. that capacity is nearly free. It was timing
dead rows. Deriving molecules-per-batch from `capacity / mean_atoms_per_molecule`
turned that into a quoted 26,037 molecules/s against a real 2,606. Record the
fill with every cell, and treat a fill below ~0.5 as a measurement of padding.

What the low-fill sweep *did* establish, because the padded shape is what
production allocates: `unique_graphs: 1` in all nine cells — one graph compiled
per configuration on the real training step, not just the isolated `energy()`
probe — and `large/131072` peaking at 11.95 GiB where the pre-Rondo table records
20.2 GiB, so that table's ceiling is not this model's.

<!-- mol:note:topic:compile-no-graph-breaks -->
## [2026-09-18] A global config read is a graph break, and dynamo skips the caller

`molrep.chem.encoder._empty` built the `(0, D)` stand-in for an absent
namespace with `like.new_zeros((0, dim), dtype=config.ftype)`. That one
attribute read cost Rondo's traced half **12 graph breaks** (13 graphs);
without it the same measurement is **1 graph / 0 breaks** (GH200 job 2616579).

Two things worth keeping:

1. **Why it breaks.** `molcfg.Config` resolves every non-underscore attribute
   through a custom `__getattribute__` that calls
   `object.__getattribute__(self, "_data")`. Dynamo cannot trace that —
   `TypeError: expected 1 argument, got 2` — so it is a break wherever a
   config attribute is read, which is the general form of CLAUDE.md's
   "pydantic / config attribute reads". `molix.config` is that kind of object.
2. **Why it was expensive out of proportion.** The break was not graceful:
   `Failed to handle graph break gracefully. Skipping the function and falling
   back to eager`. One unreachable-looking line therefore de-compiled whole
   enclosing functions, which is why 1 read produced 12 breaks.

**The fix is not to hoist the read — it is to not need it.** A tensor standing
in for features derived from the atom embedding has the atom embedding's
dtype by construction, and `like.new_zeros((0, D))` already carries dtype and
device. Reading a global to restate what the reference tensor already knows
was both the break and a latent mixed-precision bug.

**How it hid.** The empty-namespace path only runs for a molecule with no
impropers — butane has none, so the smallest real batch exercised it, but no
*eager* test could see it: the result is numerically identical either way.
Only the compile gate catches this class, which is the argument for keeping
one `explain` test per compile seam rather than trusting eager tests.

**Diagnosing the next one.** `Compiler.explain` reports `break_sites` from
`break_reasons`, and on torch 2.15.0.dev this came back **empty** for these 12
breaks while reporting them correctly for a control function with a `print`.
So an empty `break_sites` next to a non-zero `graph_break_count` does not mean
"no attributable site" — run the target under `TORCH_LOGS=graph_breaks`, which
printed the full user traceback down to the offending line.

**The second site, and the sweep that found it.** `molrep.chem.embed`
`AtomChemEmbedding.forward` had the same read behind `if Z.numel() == 0`, so the
compile gate could not see it — butane has atoms. Fixed the same way: the empty
answer is a non-persistent `(0, D)` buffer built in `__init__` from
`next(self.joint.parameters()).dtype`, which is the dtype the skipped
`self.joint` call would have produced, and which follows `.to(device)` /
`.to(dtype)` with the module.

An AST sweep is the cheap standing check, because a *guarded* read is invisible
to any single trace: walk `src/`, flag `config.<attr>` whose enclosing
`FunctionDef` is one of `forward` / `energy` / `execute` / `compose` /
`embeddings` / `tables` / `charges`. It found exactly these two; after the
fixes it reports **0**. Every other `config.ftype` in the tree is an
`nn.Linear(..., dtype=config.ftype)` in a constructor, which is correct and not
on a traced path.

**The same sweep, widened to the other forbidden sources** (2026-09-18). Flag
`.item()` / `.tolist()` / `nonzero()` and `int(...)` / `float(...)` / `bool(...)`
of a non-literal, in the same set of runtime methods, over `molzoo/rondo`,
`molrep/chem`, `molpot/forcefield`, `molix/core/steps`, `molix/autograd` and
`molix/data`. Nine candidates, and **none is a hot-path hazard** — worth writing
down because the sweep will keep reporting them:

- Seven are `int(x.shape[0])` (or `int(graphs.batch_size[0])`): reading a
  **static shape**, not a tensor value. Under the static-shape MD/training
  default dynamo constant-folds these, which is why Rondo's traced half measures
  1 graph / 0 breaks with them present. A shape read is not a sync.
- Two are `molix/data/residual.py`'s unsupported-Z diagnostic —
  `bool(illegal.any())` and `z_long[illegal].tolist()`, a host sync and a
  boolean-mask index. They are in a `BatchTask.execute`, and
  `datamodule._CollateFn` runs the batch tasks **before** `batch_to`, so they
  operate on CPU tensors in the collate (in a DataLoader worker), never on the
  compiled path. Not a device sync, and not in the compiled region.

So the honest summary of the classification is: a sweep of this kind cannot tell
a shape read from a value read, or a collate-time task from a traced one. Read
the hit before believing it, and keep the two facts that decide it — static
shapes are the default, and batch tasks precede `batch_to`.

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

**Task label (2026-10-09).** `TaskRouter.route` is `torch.unique(labels).tolist()`,
one host sync per batch. Owner: `molix.core.steps.multitask`. It stays in the
eager prologue of `MultiTask*Step.forward_loss`. The traced body is the model
plus the already chosen loss (`tests/test_molix/test_core/test_steps/test_multitask.py::test_routed_body_explains_as_one_graph`).
Under a process group, `Trainer._wrap_distributed` refuses `compile(step=True)`:
`DistributedDataParallel.forward` is the gradient-sync seam and is not part of
that region. `compile(step=False)` compiles the module; `train()` wraps DDP
around it. Token-budget plans are strided by `BatchPlanShard` so every rank
takes the same number of steps.

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

**MACE E+F seam on torch 2.15dev (2026-09-10/11).** Why `energy_forces` was
not one graph, in three independent parts. (1) `torch._dynamo.config
.trace_autograd_ops` defaults to False (2.14 and 2.15 nightly), so the in-graph
`torch.autograd.grad` at `force.py` graph-breaks — `molix.compile.Compiler
.__call__` now sets it process-wide and `Compiler.explain` patches it for the
trace; with it, inference E+F is 1 graph / 0 breaks on all four MACE models
(force parity 1e-16…1e-14, GH200). (2) Training (`create_graph=True` inside,
`backward` outside) is refused by dynamo's leaked-intermediate check
(`output_graph.py _check_requires_grad_intermediate_outputs`: FX-taint from the
`requires_grad_()` leaf, refuses any tainted requires-grad output) although the
force is differentiable w.r.t. the parameters and AOTAutograd is correct there
(grad parity 5e-17…8e-15 with the check relaxed). Upstream PR
pytorch/pytorch#196653 relaxes it to autograd reachability of a graph input;
molnex does not wait for it: the **caller-owned leaf** rule (2026-09-11) —
`Derivative` reuses a column that already is a live leaf instead of making
one, `Derivative.mark_leaves(frame)` marks the columns eagerly, and the
default steps run it in `forward_loss` before the traced body — makes the
leaf a graph input (probe C10: 0 breaks, same parity). (3) The virial strain
`torch.zeros(3, 3, requires_grad=True)` is an unconditional dynamo break;
upstream PR pytorch/pytorch#196652 traces the kwarg as `requires_grad_()`;
molnex now writes `torch.zeros(3, 3).requires_grad_(True)` in
`autograd_forces_and_virial` (2026-09-11), which traces on stock torch —
`test_force.py::test_autograd_forces_and_virial_traces_to_one_graph` (slow).
Also fixed here: `autograd_forces_and_virial` returns `energy.detach()` when
`create_graph=False` — the consumed energy graph was a legitimate dynamo
break ("autograd.grad consumed returned tensor's grad_fn"). Measured with both
PRs (GH200, matpes fused, 256 atoms / 3080 edges, fp64): training step
`fullgraph` inductor 43.5 → 13.9 ms, peak 0.048 → 0.044 GiB; E+F+virial
inference 20.8 → 6.3 ms — both vs pure eager, NOT vs the current
compiled-`energy_core` + eager-grad seam, which was not measured. **Periodic
E+F+W (2026-09-11, `repro/probe_periodic_virial.py`, job 2292738: 128 atoms,
12 Å cube, 4936 edges of which 1888 images, `pair_diff = pos[t]-pos[s]+shift`
strained with pos and cell, fp64):** autograd virial vs central differences
1.2e-5 on |W| = 620 (h = 1e-4), net force 7e-13. Inference on the **stock**
nightly: 1 graph / 0 breaks, inductor 21.1 → 5.9 ms (3.6×), peak 0.040 → 0.037
GiB, dF 1e-13 / dW 2e-13. Training (loss on E, F, W): stock 4 breaks (both
leaves leak, `force.py:120` / `:222`); with #196653: 1 graph / 0 breaks,
inductor 46.8 → 14.7 ms (3.2×), peak 0.053 → 0.046 GiB, param-grad diff
2.4e-11. **Static shapes + CUDA graphs (2026-09-11, jobs 2293234/2293350,
`repro/probe_design.py`, `repro/probe_cudagraph.py`):** E+F inference on
fixed-capacity edges (live prefix + dead self-loops on atom 0 at 10·r_cut,
capacity 1.35×, dE 0 / dF 7e-15 vs live-only) is launch-bound at these sizes —
eager ≈ 20 ms and default inductor ≈ 5.5 ms flat from 128 to 1024 atoms, and
+35 % dead edges cost nothing (5.30 vs 5.37 ms). `Compiler(cuda_graphs=True)`
on the static seam: **0.41 ms (128, 256 atoms), 0.72 ms (1024 atoms)** — 8–14×
over inductor, 27–48× over eager, forces identical (1e-14); outputs must be
consumed (cloned) each step with `torch.compiler.cudagraph_mark_step_begin()`.
**Static shapes end to end (spec `derivation-head-02-static-shapes`,
2026-09-11).** The MD driver hands the composed op the list's **whole capacity
buffer** every step (`_expose_live_edges` / `_restore_edges` / `n_live` are
gone); dead rows are `(0, 0)` self-loops at the dead-edge shift, which every
producer now takes from one place — `molix.schema.dead_pair_length` /
`dead_pair_shift` (`NeighborList._dead_shift`, `PadMolecularBatch`, which
takes `cutoff` and pads *dead* edges instead of `r = 0` self-loops, and
`StaticForward`). Under `MD(potential_dtype=…)` the list writes the geometry
at `NeighborList.geometry_dtype` and refreshes its cast shifts in place per
rebuild — no per-step cast, the bound `edges` namespace is one object for the
run. `MacePotential.energy_core(..., atom_mask=)` multiplies every per-atom
energy by the mask in `_sum_energy` (Polar also zeroes ghost densities and
Fukui sources), `energy(frame)` forwards `atoms.mask`; `energy_force_mse`
divides the force MSE by the real atom count. `Compiler.cuda_graphs` tells the
driver to apply the replay contract (`cudagraph_mark_step_begin()` + cloned
outputs). **GH200 benchmark through the real `NeighborList` + driver op**
(job 2301635, `repro/probe_static_md.py` / `run_static_md_gpu.sbatch`,
`results/static-md-aarch64.jsonl`; tiny fused MatPES fp64, random periodic
box, skin 0.5, capacity 1.35×, forces at the same geometry): per force
evaluation `md.compute_force` (eager list update + op) / op alone / whole
`md.run` step — 256 atoms (5660 live / 7642 capacity edges): eager 20.6 /
20.0 / 21.3 ms, inductor `fullgraph` 5.8 / 5.9 / 7.4 ms,
`Compiler(cuda_graphs=True)` **1.03 / 0.56 / 1.9 ms**; 1024 atoms (22184 /
29949): eager 23.8 / 21.0 / 24.0, inductor 5.9 / 5.7 / 7.5, cuda_graphs
**1.48 / 0.86 / 2.2 ms**. Force parity vs eager 5.6e-16 (256) / 8.9e-16 (1024)
for both compiled arms. The op alone matches the 2026-09-11 probe (0.41 ms
hand-built vs 0.56 ms through the driver at 256 atoms); the rest of the
`compute_force` time is the eager `NeighborList.update` displacement check
(one host sync) plus `_write_geometry` over the capacity, and the `run` step
adds the integrator and the rebuilds a random box provokes (28 / 36 rebuilds
in 100 steps). Next lever is therefore the eager list update, not the op.
Training leaf placement is worth ≤ 8 % (stock leaf-inside 2 breaks
14.3 ms, stock leaf-outside 0 breaks 14.5 ms, #196653 leaf-inside 13.4 ms):
the caller owns the leaf in training (`mark_leaves`), the op makes it
otherwise. **TensorDict
path (2026-09-11):** `kernels.py grad_force_pass` used to leave its own
position leaf in `batch["atoms","pos"]` (and the core's `atoms.energy`
attached to it), so the leaf escaped through a graph-input side effect and
dynamo refused it — correctly, even with #196653. Now the kernel installs the
leaf only for the core forward, puts the caller's positions back, and the
`detach_energy` policy covers `atoms.energy` too; callers that need the leaf
afterwards own it themselves (`HessianDerivation.graph_blocks`, like
`GradMode.run_energy(backward=True)` already did). Measured on the stock
nightly (GH200, matpes fused, `model(batch)`): inference 1 graph / 0 breaks,
fullgraph parity dF 1.2e-16; training 1 break at the leaf (needs #196653; 0
breaks and dF 1.1e-16 with it). Toy kernel: `test_kernels.py::TestGradForcePass
::test_traces_to_one_graph_when_kernel_creates_leaf` (slow) asserts the single
graph. `compile_core` still compiles `energy_core` only; compiling
`energy_forces` / a training step as one graph is a separate change.
Probe, repros and jobs: `.scratch/compile-seam/` (`probe_autograd_seam.py`,
`repro/min_a_factory.py`, `repro/min_b_leak.py`, `results/*.jsonl`).

**Step compile (spec 03, 2026-09-11, GH200 job 2302140, tiny MACE-MatPES
`wrt=("atoms.pos",)`, 256 atoms, fp64, `energy_force_mse`, Adam).**
`Trainer.compile()` traces the step's body `_traced_forward_loss` (model
forward + loss; backward from AOTAutograd; autocast / accumulation /
GradScaler / hooks / optimizer stay outside) after the eager prologue
`forward_loss` runs: `Derivative.mark_leaves(batch)` on the op the model runs
(the model itself, or the potential's `derivative` property, through a
`compile(step=False)` wrapper's `_orig_mod`). **Stock torch 2.15.0.dev20260909,
job 2306601 (`repro/probe_step_leaf.py`, `results/step-leaf-aarch64-2306601.jsonl`):
the E+F step is 1 graph / 0 breaks** — eager 46.0 ms/step; step compile
(inductor) 15.3 ms (3.0×, param-grad diff 4.7e-9 vs eager); `fullgraph=True`
15.5 ms; `cuda_graphs=True` **2.30 ms (20×)**, grad diff 1.4e-9. Marking the
leaf costs nothing measurable (PR-B in-region leaf was 14.6 / 2.25 ms), and
pytorch/pytorch#196653 is no longer needed for training. **Open — the
classical container on a fresh batch** (job 2306962,
`repro/diag_polar_marked.py`): `Potentials` memoises its typed view and F29
geometry gate on the batch *object* (`potentials.py::_frame` /
`_check_geometry`), so a batch it has never evaluated pays that host work
inside the region and the wrap breaks at tensordict
`typedtensordict.py:514 __setattr__` ("mapping proxy affected by dictionary
mutation"), then `derivative.py` "autograd.grad with external grad_fn" — 7
graphs marked, 11 unmarked; the same batch evaluated once eagerly is 1 graph /
0 breaks with grad enabled. MD (one bound batch) is unaffected; classical-FF
*training* through `Trainer.compile()` hands a new batch every step and is
therefore not one graph until the wrap / gate move out of the region (a
`Potentials` prologue, or a plain-batch reader) — a design decision, not
taken here. Found on the way: `Derivative` made its
leaves inside `enable_grad`, so on stock torch a compiled training backward was
a double backward for model-only and step compile alike ("aot_autograd does not
support double backward") — fixed by making the leaves in `Derivative._leaves`
before the block (a loop or comprehension inside `forward` re-breaks it);
pinned by `test_derivative.py::test_training_backward_through_the_compiled_op_matches_eager`.
`MultiTaskTrainStep` declares `untraceable` (`TaskRouter.route` host sync) and
`compile(step=True)` refuses it by name.

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
fp64 compiled vs eager: `dE = 0.0`, `max_dF = 4.97e-14`, `max_dW = 6.29e-11`, `n_pairs = 285546` (job 1974822 had `max_dW = 1.087e-10` against the same 1e-9 virial contract; |W| ~ 5200 eV, ~2e-14 relative). fp32 compiled: mean step **0.190 s** (eager baseline 0.38 s), peak **25.87 GiB** (eager 28.3 GiB; the larger k-grid did not OOM), recompiles after warm-up **0** (live E 285546 → 285446). Official goldens unchanged: `regressions/mace-polar-periodic-gto.py` diffs ~1e-5 eV (`atol` 1e-3), `mace-polar-official-water.py` `dE = 1.3e-7 eV`.
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
**Retained break — closed 2026-10-17.** The isolated-molecule all-pairs
enumeration (`_complete_directed_pairs` / `cached_complete_directed_pairs`:
`.item()` ×5, `torch.equal`, boolean masks) is gone; `gto.hartree.graph_pairs`
is the block-diagonal per-graph enumeration at a bound `max_atoms_per_graph`
capacity sketched here (`B·K(K−1)` rows, overflow asserted on device), see
the 2026-10-17 entry `no-compat-before-one`.

**Status.** active.

---

<!-- mol:note:topic:cueq-ops-capability -->
## [2026-09-14] Static capacity packing — the padder owns the shape, the packer fills it (user rule)

**Rule.** A compile-static training loader is *capacity-first*: the invariant
is the per-column capacity, never the molecule count. `MoleculePadder` is the
one owner of the static shape — `from_counts(counts, atoms=2**k, cutoff=)`
sizes every cap as `max(round_up(Σc/Σatoms × (atoms − N_GHOSTS) × slack),
max_c)` and the graph-slot count as `ceil(graphs_slack × budget / mean_atoms)
+ 1` — and `padder.budgets()` are the `TokenBudgetPacker` budgets (atoms
minus ghosts, graphs minus the sink, every other axis its cap), wired by
`DataModule(batch_counts=, val_batch_counts=)`. So no batch can exceed any
cap (construction, not probability), `padder.check(counts)` refuses a
dataset with a molecule no batch could carry, and the graph count is padded
too (`max_graphs`: dead slots with `num_atoms = 0`, `graphs.mask`; ghost
atoms' `batch` = the last slot, the sink, so a per-atom head that forgets
`atoms.mask` pollutes a slot the loss weights out). `energy_loss` /
`energy_mse` take `weight_key="mask"` on `graphs` like `force_loss` on
`atoms`; the Class-I script's Huber / MAE / per-structure L2 weight by both
masks. `PadBuckets` (smallest-fit among ≤ 8 padders) is deleted: one shape
suffices, and a per-task shape — if ever — is a `TaskMixture` concern, not a
pick by live size.

**Why (OMol25 4m, extent `molhub-home/derived/class-i-extent`, 3,986,754
rows).** Mean 54.85 atoms / 132.5 propers per molecule; Σpropers/Σatoms
2.415; max 350 atoms / 8516 propers. The retired scheme fixed `B` and padded
to `B × per-molecule max`: small@2048 = 716,804 atoms / 17.4 M proper rows
(OOM rc=137, job 2439130), large@512 = 59.5 GiB / 619 ms with a random batch
filling 16 % of the atom slots and 1.6 % of the proper slots. Next-fit dry
pack with caps at density × A × 1.10: A = 65,536 → ~1194 molecules / batch,
A = 131,072 → ~2390, A = 262,144 → ~4781; atom fill 99.9 % in every case,
cheap columns ~91 % (min 85–88 %), atoms close 100 % of bins. At 131,072 the
proper cap is 348 k — 50× under the retired scheme for *more* molecules.

**The recompile that "capacity-first" hit before** (`runs/pad-buckets-2418298.err`)
was not capacity-first: `graphs.num_atoms` was never padded, so every
distinct live `B` was a new shape (`expected 282, actual 312 … 425`,
`FailOnRecompileLimitHit` after 8), and the valence caps came from a p99
quantile while only atoms were budgeted, so a high-density molecule pushed
propers over the cap. Caps come from max and density, never from a quantile;
every padded axis is a budget; the graph count is an axis.

**Edges are a budget column, and on 4m they are *the* column** (user ruling:
every potential has non-bonded terms; there is no bonded-only training). The
extent scan counts each molecule's cutoff list by the collate list's own rule
(`0 < r <= cutoff`, both orientations — `get_neighbor_pairs.cpp`), keyed by
`(source_id, cutoff)`. Measured at the field's 10 Å open-boundary list (scan
job 2448990, 32 cores, 4 min 19 s): mean **3768.7 edges/molecule**,
Σedges/Σatoms **68.7**, max 70,250 — at 10 Å a 55-atom molecule's cutoff
graph is nearly complete. So the edge cap is ~76× the atom cap (A = 65,536 →
4.95 M edge rows) and **edges, not the valence columns, set HBM**; the atom
grid of the sweep starts at 2^13, and the provisional `packed_max_atoms` is
2^16 / 2^15 / 2^14 (small / medium / large). The `pad_atoms × 128` heuristic
is gone (it could overflow a dense 350-atom molecule *and* shipped dead
edges). `nrows(store, "propers")` now resolves the `<family>_n` count column,
so a FrameStore budgets valence families without a scan.

**Scan cost**: the per-frame edge count is a numpy `(N, N)` distance matrix
(0.17 ms), not a one-frame collate through the O(N²) kernel (248 ms, 316×) —
verified equal on 34 real 4m frames. Whole 4m + val-neutral scan is ~22
core-hours: 32 cores, 4 min. The job asks for 128 cores / 20 min so it
backfills.

**Calibration, done (checklist §4/§5/§7; `runs/class-i-hbm/*.json`, GH200 96
GiB, 2026-09-15).** `benchmarks/bench_class_i_hbm.py --size <tier> --atoms
2**k --view random|top10`: `random` = a random 10 % of 4m; `top10` = the union
of the top 10 % by every budgeted column, packed under the **same** padder —
the recompile gate (`torch._dynamo.config.error_on_recompile`). **Every one of
the 24 cells: 0 recompiles**, 12–20 distinct live graph counts each, so one
compiled graph served every shape the packer produced.

| atoms | mol/batch | small GiB/ms | medium GiB/ms | large GiB/ms | steps/epoch |
|---|---|---|---|---|---|
| 65536 | 1185 | 6.4 / 77 | 7.1 / 84 | 10.1 / 106 | 3362 |
| 131072 | 2388 | 12.7 / 143 | 14.1 / 154 | 20.2 / 198 | 1669 |
| 262144 | 4806 | 25.4 / 264 | 28.1 / 285 | 40.3 / 369 | 829 |
| 524288 | 9598 | 47.1 / 513 | 47.5 / 555 | OOM (77.5 peak) | 415 |

Operating point **131072 for all three tiers** (89 / 90 / 93 % of each tier's
peak molecule throughput at a quarter of 524288's memory and twice 262144's
optimizer steps); HBM ceiling 2^19 small / 2^19 medium / 2^18 large. The fills
are the design working: on `random` the atom column fills 1.000 and pairs
0.907; on `top10` atoms drop to 0.810 and **pairs go to 1.000** — the big
molecules close their bins on pairs, and the step time and peak memory are
unchanged between the two views, which is the whole point of fixing the shape.

**Three bugs the gate caught, all of them "a shape that depends on the data":**
(1) the first batch carried one extra `atoms` column because the one-time seed
ForceField build ran the encoder on the caller's batch (`chem_features` written
in place) — seed on a `clone()`; a namespace's *column count* is a dynamo guard
exactly like a tensor size. (2) `ExclusionTable._keys` was sized by how many
1-2/1-3/1-4 pairs survived the dedup, i.e. by content — now padded to one slot
per bonded column with the `_NO_MATCH_KEY` sentinel, so its length is a
function of the input shapes. (3) `NeighborList(max_num_pairs=)` is a
**per-molecule** floor (the kernel runs per molecule and grows to that
molecule's own complete half graph), and the loader was handing it the whole
*batch's* edge capacity — 4.95 M pairs allocated per molecule per worker, which
took the host out of memory at 11.7 GiB RSS. It is `n(n-1)/2` of the largest
molecule (`_list_capacity`).

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

**Open routes** (named, not silent): `/mol:fix` scoped to `molix.data.task`
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

**Runtime / engine split (2026-10-15).** `interface/` is two layers, one-way:
`interface/runtime/` (`molnex_runtime`, LibTorch only) ← `interface/lammps/`.
The runtime owns everything about the model — `Metadata` (the versioned
`DeployMetadata` `{name}.json`, model units, no LAMMPS unit table),
`ModelArtifact::open` (format → loader; `AotiArtifact` is the only format,
ONNX etc. would be siblings), `OpLibraries` (`dlopen` of `lib<name>.so` from
`MOLNEX_OP_LIBRARY_PATH`, requirement checked through the library's exported
`molnex_op_library_requirement()`, never closed because the registrations are
process-wide), `AtomEnergyExec` (the calling convention; three outputs, CPU
float64). The LAMMPS layer owns everything about LAMMPS, including the unit
conversion: `pair_molnex` maps the model energy unit onto a LAMMPS style
(eV → metal, kcal/mol → real) and calls `utils::get_conversion_factor`;
`CutoffGraphImpl` refuses a non-angstrom length. The stub in
`test_contract_impl` is a `ModelArtifact`, not an `Exec`: the `Exec` /
`DynamicExec` hierarchy, `read_model_outputs`, `MetaSchema`, `LammpsContract`
and `ModelRunner` were removed with the split. `update_weights` /
`constant_names` are `AotiArtifact`-only (constants are an AOTI notion).
The earlier `Contract::fixed_graph` / `registry.cpp` note is obsolete — both
were deleted before this split; a fixed-topology impl for a learned Class-I
field would be a second `ContractImpl` beside `CutoffGraphImpl`.

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
in the model it is `molix.data.TensorFrame` (`mn.TensorFrame`) — a
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
| `molix.data.block` / `block_view` / `molix.data.block_plan` | `molix.data.frame_store` / `frame_store_view` / `molix.data` |

**What deliberately did not change**: the format id `molix.block/1`, the
manifest keys, and lowercase "block" as the name of the on-disk unit. That id
is a wire contract already written into every store MolHub has produced;
changing it is a format revision, not a rename. So: a `FrameStoreWriter`
writes a frame store, whose on-disk form is one `molix.block/1` block.

**Downstream**: molhub funnels its coupling through exactly one import,
`src/molhub/conveyor/prepare.py:446` (`from molix.data.block import
BlockWriter, SamplingUnit`), plus `molix.neighbor.NeighborList` in
`src/molhub/dataset/neighbor.py`. That line was already broken by the
`SamplingUnit` deletion; the rename makes the module path stale too. Fixing it
is one line: `from molix.data.frame_store import FrameStoreWriter`, then
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
   `tests/test_molix/test_data/test_collate.py`. **Producers of
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
~~`Frame.meta` returns a **copy** — `frame.meta["x"] = 1` is a silent no-op, so
adding a meta scalar means rebuilding the frame (`_with_meta`).~~
**Wrong, and corrected 2026-09-22.** `Frame.meta` hands back `FrameMeta`, a
live write-through view: `frame.meta["x"] = 1` lands in the frame (verified
against the installed wheel), and so do `del`, `pop`, `update`, `clear` and
whole-dict assignment. `_with_meta` never had a reason to exist; it rebuilt the
frame — 434 418 `schema.column` calls, 22 % of a production batch — to set two
floats, and it is gone from `class_i_fit.py`. What *is* still not dict
behaviour: iteration order is alphabetical rather than insertion order,
`keys()`/`values()`/`items()` return lists instead of views, and a nested JSON
document is handed back as a snapshot, so `frame.meta["run"]["step"] = 3`
**is** a silent no-op. Read, modify, write the whole value back.

**Still open**: `molhub` cannot write a store. Its conveyor flattens frames into
molnex's storage column vocabulary (`RawUnit`) and calls a writer API that no
longer exists. The fix is *not* an import change: `encode()` already takes
`Sequence[Frame]`, so the flattening is now molnex's job and molhub should hand
its frames to `FrameStoreWriter.append(frames, key=…, task=…)` directly. Only
three lines consume the columns (`prepare.py:471-472`, `encoder.py:147,156`),
but `RawUnit` and its tests are built around them — a molhub design change,
owned there, not here.

## torch 2.14 AOTI miscounts a defaulted kwarg-only `Scalar`; fixed upstream (2026-09-09)

`ctest -R bridge_dynamic_exec_matches_reference` fails against torch **2.14**
with

```
Mismatch between ints consumed and num_ints, got int_id = 1, num_ints = 2
oss_proxy_executor.cpp:835 → AOTInductorModel::run_impl
```

The two halves of AOTInductor's extern-kernel path disagree about
`aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar
alpha=1)`:

- the generated wrapper calls
  `aoti_torch_proxy_executor_call_function(…, num_ints=2, {0, 1}, num_tensors=4, …)`
  — `dim=0` **and** the defaulted `alpha=1`;
- the extern-node JSON in the `.pt2` lists only `self`, `dim`, `index`,
  `source`. `serialize_inputs` takes a kwarg-only argument only out of
  `kwargs` (its positional branch is guarded by `not schema_arg.kwarg_only`),
  so a kwarg-only argument left at its default is dropped.

`OSSProxyExecutor::call_function` then walks the JSON, consumes the one int
the schema mentions, and asserts it consumed all of them.

**Already fixed upstream.** On torch nightly `2.15.0.dev20260908` the same
fallback op is emitted with `num_ints=0` — the wrapper no longer counts the
defaulted kwarg-only Scalar — and both the minimal reproducer and this ctest
pass unmodified. The three Python functions involved
(`FallbackKernel.export_extern_kernel_node`, `serialize_inputs`,
`generate_extern_kernel_args_decl_if_needed`) are byte-identical between 2.14
and that nightly, so the change is on the argument-collection side feeding
them. No PR was opened: there is nothing left to fix upstream.

**Reproducer** (`.scratch/aoti-alpha/probe_scalar.py`): a namespaced custom op
declared with an explicit schema ending in `*, Scalar alpha=3`, made an
inductor fallback, exported with `aoti_compile_and_package`. Fails on 2.14,
passes on the nightly. Note that `torch.library.custom_op` infers **SymInt**
for an `int` annotation, which takes a different path and does *not*
reproduce — the schema has to be spelled out.

**Reach while molnex is on 2.14.** Anything exported through AOTI whose graph
contains the *functional* `Tensor.index_add`: `molrep/embedding/angular.py:200`
(the only functional use; the other 227 call sites are the in-place
`index_add_`, which export functionalises back to the same node) and the
interface fixture `interface/tests/generate_fixtures.py:125,133`. Two
molnex-side workarounds were measured and both fail: passing `alpha=1`
explicitly (FX normalises it back to the default before serialisation) and
switching to `index_add_` (functionalised back to `aten::index_add`).

**Status.** The interface ctest under the CI filter is 68/69 on torch 2.14 and
69/69 once torch carries the fix. Not to be papered over by excluding the
test — the exclusion would outlive the torch pin.

Related: `compile-no-graph-breaks` (why `index_add` is an inductor fallback in
the first place), and the k=1 `torch.bmm` int32 overflow
(pytorch/pytorch#196431) — same family, found the same way, but that one *is*
still open upstream.

---

<!-- mol:note:topic:one-data-layer -->
## One data layer; molnex has no file format (2026-09-17)

**What changed.** `molix.store` + `molix.dataloader` + `molix.batch` → **`molix.data`**.
`TensorFrame` → `molix/frame.py`, `NeighborList` → `molix/neighbor.py`: one is what
the *model* reads, the other is geometry, so neither is loading machinery.
`molix.block/1` and the whole `FrameStore*` family are **deleted**.

**Why the format went.** It had no producer, no consumer, and no instance:

- `find <workspace> -name commit.log` → **0**; no store was ever written.
- The only `FrameStoreWriter(` outside its own tests was a benchmark script.
- The Class-I production path reads MolHub's LMDB — measured call stack
  `molhub/io/lmdb.py:94 get` → `omol25.py:_frame_from_record` (job 2577512).
- molhub's import of it had been broken since the 2026-09-08 rename and nobody
  noticed, because nobody used it.
- `DataModule._resolve_collation` probed `_block_capable(dataset)` on every
  dataloader build for a route that could never fire.

~6 200 lines of code and tests went; the unit tier went 8:26 → 5:20.

**The naming note above (`frame-store-rename`, 2026-09-08) is superseded.** It
recorded a decision *I* made and gave *my* rationale; it cited no operator
instruction, and the operator's position was that the `FrameStore` spelling was
never asked for. A note recording "I decided X" is not evidence that X was
approved — do not read `.claude/notes/` that way.

**Two things that looked dead and were not.** Both were caught by tests, and
both are now pinned in docstrings:

1. The `column` / `frame_mol` branch of `nrows` is a **count-sidecar protocol**,
   not a store detail: a source that answers `n_<table>` without materialising a
   sample lets the sampler budget a whole epoch with **zero** `__getitem__`
   calls. Deleting it turned an O(1) metadata read into materialising every
   sample (`assert 128 == 0`). Same reason `SubsetDataset.nrows` reads *through*
   the parent instead of via `self[i]`.
2. `tests/test_repository_boundaries.py` banned the *name* `molix.data` because
   a deleted ETL layer had owned it. The guard was re-aimed at the invariant it
   always meant — molnex acquires no data — and now fails on any acquisition
   verb in `molix.data.__all__`, which is stricter than the name check was.

**`nrows` is a member now**, not a module: the scalar is molpy's
(`block.nrows`), `BaseDataset.nrows(table)` is the per-index vector, and
`BaseDataset.nrows_of(source, table)` is the adapter for a foreign dataset,
since `DataModule` is handed whatever dataset the caller owns.

**Superseded 2026-09-18 (operator ruling).** `BaseDataset.nrows` is deleted:
a *dataset* must not spell a request the same way a *block* spells an
attribute. `Block.nrows` stays the free attribute; the dataset side is three
named things — `counted_tables` (a class-level **declaration**, for the common
tables a dataset developer already knows: `OMol25Dataset.counted_tables =
{"atoms": "_n_atoms"}`), `count_rows(table)` (a **verb**, for a container that
must decode to count and wants to cache what it measured), and
`nrows_of(source, table)` (the **request** every batch budget goes through,
resolving declaration → verb → materialise-every-sample).

**Open, not decided here**: molhub still imports `molix.store.block`
(`conveyor/prepare.py`, `loader/materialize.py`) — its half of the deleted
format. Re-pointing it would be a fake fix; whether molhub's store *writer*
goes too is a cross-repo call for the operator.

<!-- mol:note:topic:scripts-hold-no-model -->
## The Class-I dataset is THEMol torsion-scan, and a row is 24 molecules (2026-09-20)

Rondo trains on `dataset:themol/torsion-scan@1`, not OMol25. Two reasons, both
measured, and the second one changed a load-bearing invariant.

**It ships the topology.** OMol25 frames carry no bonds, so `ClassIPrepare.bond`
inferred them per molecule inside collate at 379 us/molecule, which is what made
the step CPU-bound. A THEMol record carries a mapped SMILES and a `bonds` block,
and `bond()` already short-circuits on `if "bonds" in frame` — so that cost is
simply gone, not optimised.

**A row is a whole scan.** One record is one complete constrained profile: 24
conformers of the same molecule about one dihedral, which is what a Class-I
torsion series is fitted to and what ByteFF (arXiv 2408.12817) trains on. Two
consequences that do not look alike:

1. *The split unit is the molecule, not the row.* 83 951 scans cover 82 332
   distinct molecules (1.02 scans/molecule); 1 592 molecules contribute 2-3
   scans, and those sit up to 81 834 rows apart in the shard. A row-wise shuffle
   leaks 1 619 scans. `THEMolShardDataset.molecule_index()` is the group vector,
   from `mapped_nonisomeric_smiles` — deliberately coarser than the frame's
   `molecule_id` (a uuid5 of the *isomeric* SMILES), because stereoisomers share
   a topology and therefore every Class-I parameter.
2. *An extent row must cost what a row costs.* `_scan_extent_slice` read
   `sample[0]` and recorded one conformer: 33 atoms where the row costs 792. The
   padder would have sized every cap 24x too small and every batch overflowed it.
   OMol25 hid this completely — there a record is exactly one frame, so the first
   frame *was* the row. Fixed by summing over the record's frames, `pairs`
   summed per frame rather than multiplied because the cutoff edge count is a
   function of the geometry and a scan is 24 geometries.
3. *The graph slot count is not the row count.* The same assumption was baked
   into three more places, and fixing the atom sums did not fix them — the first
   real batch came out of the collate with **529 graphs against max_graphs=30**.
   `MoleculePadder.from_counts` divided the atom budget by the mean atoms per
   *row* and called the answer a graph count; `DataModule._budget_sampler` skipped
   the `graphs` table outright; and `TokenBudgetBatchSampler._budget_counts`
   hard-coded `torch.ones(n)` for it. All three were right while a row was one
   graph. The extent now carries a `graphs` column (frames per row, not a Frame
   field), `from_counts` takes it and floors the slot count at one row's graphs
   plus the sink, and both sampler layers use the vector when it is given —
   defaulting to one graph per row, which is every per-molecule dataset. Adding
   the column changes the npz schema, so the frozen scan had to be redone; the
   reader raises `KeyError` on the old file rather than reading around it.

**Two sidecars, because the shard is on Lustre.** Listing the 83 951 HDF5 group
keys took **194 s** on a cold page cache and 0.10 s warm; the molecule index is
one string read per group, 245 s warm for the whole shard. Both are properties of
an immutable file, so both are cached in a `molhub.cache.CountSidecar` keyed by
the shard's own size and mtime. `CountSidecar.load` gained `n_rows=None` for the
key vector, which cannot be length-checked against anything because it *is* the
row count. Earlier in the same session I quoted 29 min for the molecule index —
that was a cold-cache extrapolation from the first 2 000 rows and was wrong.

**One frozen extent serves both splits.** It is keyed by `(source_id, cutoff)`
and `MoleculeExtent.for_dataset` slices a parent scan for a `SubsetDataset`, so
`scan_class_i_extent.py` scans the whole shard once (134 ms/row single-core, 188
min; ~minutes on the 128-core CPU partition) and both views read it. Do not scan
the two views separately.

**The OMol25 capacity grid was void for this dataset, and its 13 cell files are
deleted** — but it carried one signal worth keeping. Step time was nearly flat,
85.6 -> 91.8 ms across an 8x capacity range for the small tier and 93.9 -> 114.8
for the large, i.e. across a 29x parameter range too, while peak memory tracked
capacity faithfully (0.31 -> 2.41 GiB small, 1.53 -> 11.95 GiB large). A step
time that does not move with either the work or the model is a **loader-bound**
step. Above that grid it stopped being subtle: `small/262144` took 6 332 ms/step
and `small/{524288, 1048576}` died with `DataLoader worker exited unexpectedly`.

So when reading a fresh grid, the first question is whether ms/step tracks
capacity at all. If it is flat again the answer is workers — the calibration
sbatch asks for 16 CPUs and `class_i_data_module(..., num_workers=4)` uses four
of them — not a smaller capacity.

**The three tiers' parameter counts, re-measured for this architecture** (the old
ones described the model Rondo replaced and are void). Powers of two throughout;
the counts are the scaling-law x-axis, not a choice between candidates:

| tier | hidden | blocks | torsion terms | parameters |
|---|---|---|---|---|
| small | 32 | 2 | 4 | 41 108 |
| medium | 64 | 4 | 8 | 204 188 |
| large | 128 | 8 | 16 | 1 191 212 |

1 : 4.97 : 28.98 — a decade and a half of span. ByteFF's own parameter count is
not recorded in `src/molzoo/specs/rondo.md`, so no comparison is claimed here.

**Still open, named not fixed:** the energy target. THEMol absolute energies
carry a molecule-dependent offset (a sampled record: -503 415.87 kcal/mol), and
the loader fits an epsilon[Z] table on the train side to absorb it — the in-tree
path OMol25 used. ByteFF instead fits energies *relative to each scan minimum*,
which is available cheaply here because the whole profile is one record, and is
the more faithful target. Not switched silently; it changes what is trained.

## molpy and molrs move in pairs, so a run has to freeze both (2026-09-20)

`ImportError: cannot import name 'LammpsCpuUse' from 'molrs.io'`, in a DataLoader
worker, three cells into calibration 2726079. Neither library was broken:

| | |
|---|---|
| molrs `8876e938` adds `LammpsCpuUse` | 14:50:36 |
| molpy `651a818e` starts importing it | 14:56:33 |
| the aarch64 wheel the job installed | built 14:47:15 |

A GPU job reads molpy **live** through `PYTHONPATH` and molrs from a **wheel**
built minutes earlier, so a job straddling a coordinated pair is broken by
construction. The gate had said the wheel did not match the sources; I overrode
it with `MOLRS_WHEEL_ALLOW_STALE=1`, reasoning that a bounded job validates the
training path rather than pinning molrs. **That reasoning was wrong** — a stale
molrs breaks *molpy's imports*, not just molrs behaviour — and rebuilding faster
does not help, because the tree moves during the four-minute build.

`lab-new/projects/class-i/train/snapshot.sh` takes a `git worktree` of molrs,
molpy, molhub and molnex at their current commits; `build_molrs_aarch64.sbatch`
and the run scripts take `SNAPSHOT=` and work from it.

**Worktrees, not a copy.** The first version rsynced the four trees behind a
hand-written exclude list and reached **30 GB** before being killed. The list was
guesswork over trees I had never measured: `molrs/molrs-python` is 17 GB of build
output, `molhub/node_modules` 611 MB (molhub has a JS app), `molnex/src` 93 MB of
compiled objects. A worktree copies the tracked files only, hardlinks the
objects, excludes every build artifact by construction, and takes **8.4 s**.

**Three things the snapshot has to get right**, each of which I got wrong first:

* The wheel lives under the molrs root it was built from. Splitting that (build
  to `${SNAPSHOT}/.wheels-aarch64`, gate reads `${SNAPSHOT}/molrs/...`) made a
  correct wheel read as a missing one.
* The fingerprint is taken **after** the build. Cargo writes
  `molrs-python/Cargo.lock` (57 635 bytes) during it, gitignored — so
  `git status` stays clean while the fingerprint changes, 524 files recorded
  against 525 present, and the job rejects its own wheel.
* `CARGO_TARGET_DIR` stays outside the snapshot. Inside, it left 604 MB in a tree
  meant to be cheap; shared, snapshots reuse each other's incremental artifacts.

**And the freeze alone is not enough**: a snapshot taken *between* a coordinated
pair is itself inconsistent. The build job therefore installs the wheel it just
built and imports the snapshot's molpy against it, touching
`molpy.io.read_smiles` — the attribute whose lazy import found the skew — and
fails there, at the cost of one build rather than a GPU allocation.

**The cost, stated:** uncommitted work is not in a snapshot (molhub had 15 dirty
files, molpy 4, molrs 2). For a 24-hour run that is the right default — it is
reproducible and nameable — and `REVISIONS` records the count.


## The torsion-scan step is the loader, and the loader is one HDF5 read (2026-09-20)

Building one packed batch takes **26.8 s** at a 131 072-atom capacity and
**43.5 s** at 524 288. The GPU step at those sizes is tens of milliseconds. So
the loader is not a contributor to step time, it *is* step time, and the flat
ms/step the old OMol25 grid showed across an 8x capacity range was that.

Where it goes, measured rather than guessed:

| | cost |
|---|---|
| read one record (24 frames) from HDF5 | **64.9 ms cold / 15.1 ms warm** |
| `ClassIPrepare.topology` + `label` over the same frames | **0.89 ms/frame** |

4.3x from the page cache, and the shard is 6.5 GB against 485 GB of node memory,
so it warms after one pass. The read is **I/O bound**, which is why workers help
close to linearly — and the loader was running **four** of them against 16 to 32
allocated cores.

**Two hypotheses the measurement killed**, both of which I would have acted on:

* *The valence enumeration is redone for all 24 conformers of a scan.* It is not.
  `_read_torsion` builds one topology per record and reuses it; all 24 conformers
  come back with identical `(34, 59, 76, 9)` shapes and byte-identical angle
  lists. Prepare is 3 % of the cost, not the problem.
* *The padded tensors dominate host memory.* They are about a fifth of it. One
  in-flight batch costs **0.90 GiB** at 131 072 atoms and **1.48 GiB** at
  524 288, against 0.17 GiB for the padded columns — the unpadded frames each
  batch is built from, and their Python objects, are the rest.

That second number is what killed calibration job 2724601 in all six cells it
ran: `num_workers=4, prefetch_factor=4` is four workers holding five batches
each, 17.7 GiB in flight at the *smallest* calibrated capacity, against 22.8 GiB
for the job — because Slurm hands out memory **per CPU** (`MinMemoryCPU` 1459M)
unless a job states otherwise. `run_day_gh200.sbatch` documented `--cpus 8`,
which is 11.4 GiB, so production would have died in its first batch.

`lab-new/projects/class-i/train/loader_plan.py` reads the allocation, sizes the
in-flight set against the measured per-batch cost, and fails closed when it
cannot feed two workers: (3, 2) at 16 CPUs / 22.8 GiB, (16, 4) at 32 / 96 GiB,
and a refusal at 8 CPUs.

**The remaining ceiling is the random read itself** — 176 to 694 records per
batch, 24 small HDF5 datasets each. Warm and with 16 workers that is about
0.1 s/batch of reading against ~0.1 s of prepare, so the loader stops dominating;
below that needs an on-disk shape built for batched reads, which is MolHub's
(one data layer, molnex has no file format).

**A failed cell must keep its output.** The parent kept `tail[-1]`, and for a
*killed* process that is whatever was last on the pipe: an `ast.AnnAssign`
codegen line for one cell, a plain INFO log line for another. Two readings of
one failure, and six cells' worth of GPU time with no diagnosis. Cells now write
`cell-*.log` in full.


## A validation number of 231 651 was one missing element (2026-09-20)

Smoke 2723274 passed every gate and reported `val_mae_kcal` 231 651, frozen
across all three tiers and both legs while `train_loss` moved 81.96 -> 48.34.
I proposed three causes before measuring any of them, and **all three were
wrong**. They are worth writing down because each was plausible and each would
have sent the fix somewhere useless.

| Guess | What the measurement said |
|---|---|
| "THEMol's absolute energies dwarf the torsion barrier by 4-5 orders, so the energy channel carries no signal" | eps[Z] removes the offset: raw std 687 614 kcal/mol -> dressed std **53.3**, against a within-scan barrier of median **9.76**. A factor of 5. |
| "Training and validation compare different targets" | They don't. `predictions["energy"]` is dressed (5.14e5), `energy_mm` is not (70.2), `e0` 6.336e5 against a label of 6.34e5, residual 35.0, and `val_omol_mae` on a train batch is **94.3**. |
| "The metric is structurally broken" | Same metric, same code, 94.3 on train and 231 651 on val. A metric bug does not know which split it is on. |

The cause was **one element**. `load_themol_torsion_split` fitted eps[Z] on its
*bounded view* of the train side; the smoke's view was 80 rows, which held
`[1, 5, 6, 7, 8, 9, 14, 16, 17, 35]` and no iodine. The held-out molecules carry
iodine. `fit_atomic_reference` solves `counts @ eps = E_DFT` over the records it
is handed and leaves **eps = 0** for every element none of them contained — and
zero is not neutral: a molecule holding such an element keeps its whole ~1e6
kcal/mol offset inside what the loss reads as a residual.

The shape of the evidence is the lesson: val **median** |E_DFT - e0| was 35.6 and
the **mean** was 217 162. The median says the fit is fine; the mean says a
handful of molecules are off by a million. A summary statistic that hides the
tail hides exactly this class of bug, and the fix for the *report* is to print
both.

Two changes, because either alone leaves the trap:

* eps[Z] is fitted from the **whole** train side, before `n_train`/`n_val` cut a
  view out of it. Those bound how much a job looks at; the dressing is a
  property of the dataset and the split. Smoke and calibration now read one
  cached 12-element table fitted on 79 765 rows.
* `_require_element_coverage` refuses a table with no value for an element either
  split needs, naming it. It reads `THEMolShardDataset.element_mask()` — one
  cached int64 per row, bit *z* per element — so it costs a bitwise OR, not a
  record decode. Verified against the known failure: with the 80-row table it
  raises "no value for element(s) [53] that the val split needs".

At the calibration sample (4 096 records) every element is covered and the two
sides agree: median 52.9 train vs 56.9 val.


## A GH200 job ran a molrs 27 commits old, and `import molrs` could not tell (2026-09-20)

`.aarch64` installs molrs as a **wheel**, because an editable build writes one
untagged `.so` that the login node's x86 rustc and a GH200 build overwrite in
turn. Every run script gated the reinstall on `if python -c "import molrs"` —
and a 3-day-old extension imports perfectly. The installed one was the wheel
built 09-17 21:13, byte-identical to it, missing nine breaking changes and the
mrec `molrec_version` stamp. Importability is not currency.

Three things conspired, and each is worth remembering separately:

* `prod_day`'s launcher used a **second** wheel directory, `molnex/.wheels-gh200`,
  holding an even older wheel (09-16 10:41) on a branch that never rebuilds.
  Same version string, different bytes. A `lab-new` campaign note records this
  trap biting once already.
* molrs deleted its own `scripts/build_aarch64_wheel.sbatch` (`8e942d18`,
  "no CI, hook or docs consumer"). It had one: every GH200 job in the workspace.
  The build step now lives in the workspace, which is the thing that consumes it.
* The aarch64 rustup home had a complete toolchain and no `default_toolchain`
  key, so the shim refused with "rustup could not choose a version of cargo" —
  and `command -v cargo` said yes, because that is a shim test, not a cargo test.

The gate is now a **content fingerprint** (`molrs_fingerprint.py`: sha256 over
every `*.rs` plus the cargo manifests, 530 files / 12.3 MB, 0.23 s with the
build dirs pruned — 70x faster than an unpruned `rglob`) compared against what
`.wheels-aarch64/BUILT_FROM` records. Not mtime: molrs is a shared checkout under
active development and a touch or a reverted edit must not fire it. A mismatch
fails the job closed; `MOLRS_WHEEL_ALLOW_STALE=1` is the operator's explicit,
logged override.

## A model does not live in `scripts/` (2026-09-17)

`scripts/mm_param_learning/` went from 7 661 lines to 3 439. Deleted outright:
`align.py`, `profile_class_i.py`, `overfit_gate.py`, `scale_audit.py`,
`materialize_workspace.py`, `workflows.py`, `prod_day.py`, `quasi_prod.py`,
`diagnose_phalkethoh_mm.py`, `zinc_typing_recovery.py`, `gpu_bench.py`,
`train.py`, `molcfg.yaml`, 15 `.sbatch`/`.sh` job scripts, the whole of
`tests/test_scripts/`, and the two Class-I benchmarks. Workflow drivers belong
to a molab project/experiment; dataset acquisition belongs to MolHub; the
profiler already exists.

The model definitions that were living there — `espaloma_like_template`,
`build_espaloma_like_model`, `ForceAwareModel` — are gone too, superseded by
`molzoo.rondo` (`RondoForceField` / `RondoSpec` / `Rondo` / `RondoPotential`).
`ClassIFit` now *holds* a `Rondo` and exposes `forcefield` as
`self.parameterizer.template`, so the field the loader mounts and the field the
model fills are one object rather than two equal-looking ones (finding **F21**
closed by construction, not by convention).

What is left under `scripts/` is composition: `class_i_fit.py`,
`espaloma_mm_protocol_train.py`, `configs.py`, `constants.py`,
`omol25_fairchem_4m.py`, `scan_class_i_extent.py`. The OMol25 loaders inside
`class_i_fit.py` (~350 lines: `load_omol25_*`, `_index_omol_source`, the e0 and
count caches) still owe a move to MolHub, and since the Class-I retarget
(2026-09-20) four of them are outright **dead**: `load_omol25_official`,
`load_omol25_official_tiny`, `load_omol25_records`, `load_omol25_split` and
`load_phalkethoh_val` have no caller anywhere. They are not deleted yet because
`load_omol25_fairchem_4m` — which the FAIRChem-protocol script does use — shares
`_official_source`, `_measured_counts`, `_fit_e0_subsample` and the two cache
paths with them, so the cut is a refactor, not a deletion.

## The model zoo has one shape: spec → encoder → potential (2026-10-15)

<!-- mol:note:topic:zoo-unify -->

**Decision (operator, chain `zoo-unify`).** MACE, Allegro and PiNet2 share
MACE's shape: a torch-free `spec.py`; an `encoder.py` body that exposes
primitives and whose `forward(td)` is the encoder contract; a `potential.py`
product that **subclasses** the body (flat checkpoint keys) with
`atomic_energy` / `energy_core` (flat, compilable) / `energy(batch)` /
`forward = Derivative(energy, wrt)`; and, where official weights exist,
`checkpoint.py` key tables + an archive reader behind
`.load(path, format="official")` (that product surface was superseded on
2026-10-16 by the two-method convention in the follow-up below). Public names
changed without aliases:
`PiNet` → `PiNetEncoder`, `Allegro` → `AllegroEncoder` / `AllegroPotential`.

- **Model-specific blocks live with the model** (operator ruling, same day,
  superseding a first pass that had put them in molrep under generic names):
  `molzoo/pinet/layers.py` (PI/IP, PIX/Scale/Dot, `compute_d5`, `ResUpdate`,
  `GCBlock`, `OutLayer`), `molzoo/allegro/layers.py` (`ProductTypeEmbedding`,
  `WeightedChannels`, `Contracter`). molrep keeps only generic components:
  `FFLayer`, `ScalarMLPFunction`, `SincBesselRBF`, cutoffs, spherical harmonics
  (cuEquivariance), edge geometry (`molrep.embedding.geometry`). One owner each
  for activation lookup (`molix.nn.activation_from_name`) and the
  official-checkpoint machinery (`molzoo.checkpoint`).
- **Allegro = mainline nequip-allegro 0.8.3**, loads `.nequip.zip` packages
  without importing nequip / allegro / e3nn (exact-global allowlist +
  `torch.package` storages). The previous port could not load any official
  model: no two-body MLP, raw-Z embeddings, fixed non-persistent Bessel
  weights, irrep-major env weights, fused-per-output TP paths with `randn`
  init, linear readout, SH-only irreps. Parity: the two small packages are
  bit-level (≤ 1.5e-13 eV/atom, ≤ 1.3e-13 eV/Å, virial ≤ 2e-13, periodic case
  with 24 self-images); Allegro-OAM-L:0.1 (fp32 package, evaluated in fp64) at
  fp32 rounding (5e-7 eV/atom, 4e-5 eV/Å). cue O3 SH and CG equal e3nn's
  (CG factor 1 exactly). Oracle venv + assets (out of tree):
  `molcrafts/.build-alloc/allegro-oracle/` (`assets/` is what
  `MOLNEX_ALLEGRO_WEIGHTS_DIR` points at; `REPORT.md` is the reconciliation).
- **PiNet2: no PiNN (TensorFlow) checkpoint loading** (operator ruling).
  `out_activation` defaults to `None` — PiNN's `OutLayer` is linear (Keras
  `Dense` default); molnex had hard-coded `tanh`, so a model trained before
  2026-10-15 reproduces only with `out_activation="tanh"`.
- `pairs.shifts` is written by the MD bind path but read by no model; every
  model reads the imaged `pairs.pair_diff` value with the gradient through
  `atoms.pos`.
- **GH200 (jobs 3870277 / 3870345, fused cuEq kernels).** Allegro
  (`uniform_1d`), PiNet2 rank 3 and MACE-MatPES (`use_fallback=False`):
  `energy_core` and inference `forward` each 1 graph / 0 breaks;
  `Compiler(cuda_graphs=True)` vs eager ≤ 2e-13 eV/Å; 24-atom cluster
  8–20 ms eager → 0.4–0.5 ms. The CUDA-graph op must run under
  `torch.no_grad()` (with grad on, the attached forces leak the leaf and
  fullgraph refuses — the documented training-trace break).
- **TF32 is not safe for Allegro-OAM-L.** Under TF32 matmuls (`"high"`)
  OAM-L's periodic golden is off by 4e-3 eV/atom and 0.1 eV/Å; at
  `"highest"` the fused kernel sits at fp32 noise (≤ 2.3e-6 eV/atom,
  ≤ 1.2e-4 eV/Å) and equals the naive kernel; fp64 is 5e-7 / 4e-5 (the
  goldens are fp32). **Operator ruling 2026-10-15:** molix no longer sets the
  fp32 matmul policy at all (the `fp32` preset used to force `"high"`);
  `set_precision(mode)` takes no `matmul_precision`, PyTorch's default stands.
- **One reader pattern, no security gating** (operator ruling, same day):
  `molzoo.checkpoint.OfficialArchive` — `read(path) -> (state, config)`,
  unpickling through the single `StubUnpickler` policy (torch / stdlib / numpy /
  cuEquivariance resolved, everything else stubbed). `MaceArchive` and
  `NequipPackage` subclass it; MACE's `trusted=` flag and the tensor-only-first
  attempt, and Allegro's exact allowlist, are gone.
- **cuEquivariance only** (operator ruling, same day): `molrep.embedding.angular.
  SphericalHarmonics` wraps `cuet.SphericalHarmonics` (equal to the old
  pure-torch monomial table to 4e-16, double backward supported); the
  pure-torch version existed only for functorch, which the force path no
  longer uses. Docs and comments do not name e3nn.
- **GH200 re-check after the second round (jobs 3871556 / 3871557):** aarch64
  op rebuilt and loading; Allegro / PiNet2 / MACE on cuet SH: 1 graph / 0
  breaks, CUDA graphs = eager (≤ 2e-13 eV/Å), force-training double backward
  finite; OAM-L fp32 at PyTorch's default matmul precision 7.7e-7 eV/atom,
  3.7e-6 eV/Å against the fp32 goldens.


**Follow-up (2026-10-16, operator ruling): one potential, two methods.** The
product surface above (`atomic_energy` / `energy_core` / `energy(batch)` /
`.load(path, format="official")`) is gone. A potential has exactly two public
methods and no layer between them. `energy(frame) -> frame` is the one method
a model writes: it reads `atoms.pos` / `atoms.Z` / `pairs.pair_index`, the
optional `pairs.pair_diff` / `atoms.mask` and (MACE) `graphs.{total_charge,
total_spin, cell}` straight off the frame, writes `atoms.energy` and
`graphs.energy`, takes `B = frame["graphs"].batch_size[0]` (so the `graphs`
namespace is required and the `atoms.batch.max().item()` host-sync fallback is
deleted) and derives, detaches and validates nothing. `forward =
Derivative(self.energy, wrt)` is built as the last line of `__init__`. The
operator's words for why: single responsibility, clear boundaries, no double
forward, fully typed, no meta methods and no dynamic copies. A first proposal
that put a `atom_energy` tensor layer under `energy` and read the frame through
a `Geometry.from_frame` view object was rejected as convoluted and as an access
wrapper — the tensor-level work *is* the body of `energy`, composed from the
encoder primitives. Landed:

- `molix.autograd.Potential(nn.Module, abc.ABC)` is the base of every
  potential (`src/molix/autograd/potential.py`): abstract `energy`, provided
  `forward`, `derivative` property. `Derivative.__init__` no longer duck-types
  (`getattr(energy, "energy")` is gone): it takes a `Callable[[frame], frame]`,
  spelled `Derivative(pot.energy, wrt=…)`.
  `molix/core/steps/base.py::_mark_leaves` dispatches on
  `isinstance(model, Derivative | Potential)`. molpot's classical `Potentials`
  subclasses `Potential` too.
- Products take **typed spec objects** and the defaults live on the spec only:
  `MaceMatpes(MACEMatpesSpec(...), *, wrt=(), energy_dtype=torch.float64,
  input_dtype=None)` (likewise `MaceOmol` / `MaceOff` / `MaceLes` /
  `MacePolar`; `MACELesSpec` is new, `PolarHeadSpec` is folded into
  `MACEPolarSpec`), `AllegroPotential(AllegroSpec(...), *, wrt=())`,
  `PiNetPotential(PiNetSpec(), *, wrt=())`. Specs are
  `ConfigDict(extra="forbid")` with validators instead of
  `getattr(spec, …, default)`. `configure_input_dtype` became the
  `input_dtype` constructor kwarg, cast at the top of `energy`, so the batch
  and compiled paths agree (the old `MacePotential.energy_core` skipped the
  cast — a bug).
- Official loading is typed: `MaceMatpes.load(path, *, config=None, wrt=(),
  energy_dtype=…, input_dtype=None)` → `MaceArchive.read(path, config) ->
  (state, MaceOfficialConfig)` (pydantic, `extra="ignore"`, only the fields
  that are read) → `MACEMatpesSpec.from_official(cfg)` (torch-free, holds the
  rejection rules, testable without weights) → `CheckpointRemap(matpes_keys,
  skip=MACE_SKIP).load(model, state)`. `CheckpointRemap(table, *,
  on_unexpected, skip=())` replaces the `ClassVar _keys / _unexpected` tails;
  the key tables are module-level constants in `molzoo/mace/checkpoint.py`.
  Allegro: `NequipPackage.read(path) -> (state, AllegroSpec)` normalises the
  layout once in the reader (`.w3j` dropped there);
  `AllegroPotential.load(path, *, wrt=(), tp_method=None)`.
- **Eager subpackage imports.** `molzoo/__init__.py` is a docstring; the
  `_LAZY` + `__getattr__` + `__dir__` tables of `molzoo/__init__` and
  `mace/__init__` are deleted with the two lazy-import probe tests and the
  subprocess gate. `from molzoo import X` no longer appears in the tree:
  `from molzoo.mace import MaceMatpes`, `from molzoo.allegro import …`.
- **MACE-only blocks live in `molzoo/mace`**: `layers.py` (conv mixin,
  `RadialMLP`, `GatedNonlinearity`, `ElementLinear`, the interactions,
  `EquivariantProductBasis`), `heads.py` (`JointFeature*`, the readouts), the
  Polar blocks inside `polar.py`. `molrep/readout/`, `molrep/utils/`,
  `molrep/interaction/polar/` and every MACE / research interaction module are
  deleted; molrep keeps `embedding/*`, `interaction/product.py`, `cueq.py`,
  `heads/`, `chem/`, `condensation/`, `perception/`, `analysis/`. The research
  `MACE` encoder (`molzoo/mace/research.py`; not compile-clean, the sole user
  of ~700 lines of molrep) is deleted; `MACEEncoder.forward(td)` writes
  `atoms.node_features (N, L, F)`, the `0e` block of each layer. The rotation
  test helpers moved to `tests/conftest.py`.
- `molzoo/encoder.py::ElementTable` mixin owns `validate_elements` /
  `_validate_once` for the three encoders — the eager precondition spent on
  the first forward and folded out under tracing.
- `molpot.derivation.protocol` is deleted (`call_energy`, `PotentialModule`,
  `write_energy`, `ensure_graphs`, `absorb_model_output`, `write_forces`, and
  finally the re-export shim): `energy` writes the `molix.schema` keys
  directly, and an existing `graphs` namespace needs no ensuring.
  `hessian.py` and `forcefield/potentials.py` import `molix.schema`.

Follow-ups not done this round: `_mark_leaves` still carries the
`physics(batch)` Class-I hook through `getattr`; `molix/data/dataset.py` and
`molix/data/prepared.py` each keep a `__getattr__`; `Sonata` and `Rondo` are
not yet on `Potential`; the three
`test_potential.py::test_forces_match_central_differences` tests compare
`h = 1e-5` central differences at an absolute tolerance that sits at the
finite-difference truncation floor — make it relative. Hot-path changes (MACE
element index / reshape removal, Allegro env fusion): see the paragraph below.

**Hot paths (2026-10-16, after the convention landed; every change checked
against an fp64 seed-0 reference captured before editing, <= 2e-14 relative).**

- MACE (`molzoo/mace/{layers,encoder,potential,polar}.py`): the per-layer
  `_reshape` (per-irrep `(N, mul*d)` -> `(N, d, mul)`, cat on `d`) followed by
  `EquivariantProductBasis`'s `flatten(1)` was an identity copy in `ir_mul`
  layout and is gone - interactions return the `(N, target.dim)` message.
  `ElementIndex(index, index32)` is built once per `energy` call by
  `MACEEncoder.element_index(Z)` and threaded through
  `layer_features(elements=)` into `ElementLinear` (the per-atom gather of
  the element's weight block, every device; see the GH200 paragraph below for
  the `indexed_linear_B` kernel it replaced) and the products (no per-layer
  one-hot `argmax`; the element-agnostic `% 1` fold is decided at
  construction). `OneHotLinear`
  (`F.embedding(index, weight.view(C, F)) * c`, same `weight` key and cue
  segment layout) replaces the one-hot matmuls of `node_embedding` and the
  residual `source_embedding` / `target_embedding`; MatPES / OMol stay
  bit-identical, Polar differs by `x * (1/n)` vs `x / n` rounding
  (`_field_norms_inv` non-persistent buffer; `field_feature_norms` key kept).
  Polar: `vectors.roll(-1, dims=1)` for `[y, z, x]`, the dead `clone()` and
  `new_zeros` are gone, `local_electron_energy` runs only when enabled. The
  `ElementLinear` per-call weight transpose stays: removing it changes the
  checkpoint tensor layout. Compile: encoder / MatPES / OMol / Polar `forward`
  1 graph, 0 breaks.
- Allegro (`molzoo/allegro/{encoder,layers}.py`): `WeightedChannels` is one
  `molrep.cueq.SegmentedPolynomial` (`uniform_1d`, `math_dtype=config.ftype`)
  built from `cue.SegmentedTensorProduct.from_subscripts(",u,u")` - basis
  segments `()` per SH component, weight segments `(u,)` per irrep
  (irrep-major, so the channel-major `(E, u*R)` latent output is transposed
  per call, a copy `R/D` the size of the message it replaces), one path per
  component. The in-loop environment sum calls it with
  `output_indices={0: source}` / `output_shapes={0: N}`, so the `(E, D*u)`
  message is never materialised and there is no Python scatter; the per-edge
  first-layer call uses the same instance without output indices (one
  implementation). `SphericalHarmonics(normalize=False)` on unit vectors from
  the lengths already in hand; `source` is made contiguous once per
  `layer_features`; `__init__` asserts every layer's schedule starts with `0e`
  (the `tensor[:, :u]` scalar slice relies on it). Official-package parity
  unchanged (<= 1.5e-13 eV/atom incl. the periodic case); forces/virial differ
  from the reference by <= 2.1e-15 relative (cue `scatter_add_` summation
  order). The per-layer `torch.cat(scalars + [tensor[:, :u]])` was measured
  and left alone: 0 breaks before and after, and inductor fuses the cat into
  the consuming GEMM. The fused kernel is verified on GH200 (below).

**GH200 verification (2026-10-17; jobs 3945974, 3972523, 3978328, 3978557,
3978849;
`benchmarks/run_gh200_hot_paths.sbatch` -> `verify_gh200_hot_paths.py`, fp64,
`.venv-aarch64`, native ops from `build/csrc-aarch64`).**

- Allegro: the fused `WeightedChannels` env sum (`uniform_1d`,
  `output_indices`) equals the scatter form to 1.4e-14 (fp64) / 7.6e-6
  (fp32); tiny, the two small official packages and OAM-L match their goldens
  on CUDA eager; `Compiler(cuda_graphs=True)` = eager; explain 1 graph / 0
  breaks. The OAM-L periodic virial under CUDA graphs sits at 2.4e-13 eV
  against the script's 2.0e-13 gate - fp32 package replay noise, not a defect;
  the gate is the number to move if it recurs.
- MACE: MatPES / OMol / Polar CUDA eager vs the CPU reference
  (`benchmarks/_gh200_verify/step7_ref.pt`) <= 4.8e-14 relative; explain 1
  graph / 0 breaks for the three `forward`s (Polar on the periodic path) and
  `MACEEncoder.forward`.
- `ElementLinear` (job 3972523, every check in a fresh process): cue 0.12.0's
  `indexed_linear_B` fails inside a bare `torch.cuda.graph` capture on its own
  (`cudaErrorStreamCaptureInvalidated`; under `set_sync_debug_mode("warn")`
  the eager call shows one host sync), so `Compiler(cuda_graphs=True)` could
  not record MatPES. The per-atom gather (`index_select` of the element's
  `(u, w)` block per path, the former CPU path) records and replays to 0.0 eV
  (MatPES) / 4.5e-13 eV (OMol; the same as eager's own replay noise), is
  1.1-1.3x slower forward and 1-2x faster forward + input gradient, and holds
  `u*w` floats per atom and path (first MACE-MP `skip_tp`, 4096 atoms, 89
  elements, fp32: 0.79 vs 0.45 GiB peak forward). It is now the only
  implementation (`ElementIndex` is `index` / `index32`; the fake-op
  registrations for the cue kernels are gone; spec A6 updated). Re-check when
  cue ships a capture-safe grouped linear. A fresh-process OMol run with the
  kernel passed under `Compiler(cuda_graphs=True)` without explanation (the
  harness did not then assert that a graph was recorded; it does now through
  `cudagraph_trees.get_manager`).
- Polar periodic under CUDA graphs failed one op later:
  `aten._linalg_check_errors` from `torch.linalg.inv(cell)` in
  `molpot.potentials.gto.kspace.reciprocal_basis` - now `inv_ex` on CUDA, the
  pattern of `elec/lib/kvectors.py` / `mesh_interpolator.py` /
  `forcefield/pair_terms.py`. The non-periodic Polar path stays outside CUDA
  graphs and compiles with 34 graphs / 33 breaks: `molpot/potentials/gto/
  hartree.py:130-164` reads `.item()` for its cache key (pre-existing, topic
  `compile-no-graph-breaks`; the periodic path is the tested seam).
- Job 3978328 (after the gather landed): 76 checks pass - every
  `Compiler(cuda_graphs=True)` parity now also asserts that inductor recorded
  a graph; MatPES replays to 0.0 eV, OMol to 4.5e-13 eV, the Allegro packages
  to <= 1.6e-14 eV. Its two failures were the non-periodic Polar path (explain
  34 / 33 and the data-dependent guard under capture, both the hartree pair
  cache above); the script now reports that path as SKIP with the reason and
  checks the periodic Polar seam under CUDA graphs instead.
- Job 3978557 (periodic Polar, `inv_ex`): inductor records the graph, explain
  1 / 0, per-atom energies replay to 0.0 eV; the graph-level k-space Hartree
  term differs by 2.3e-12 eV and the forces by 1.3e-12 eV/A - the
  reciprocal-space sum reordered by inductor (every per-atom term is exact),
  not a defect. The script gates that one force comparison at
  `TOL_GRAPH_F_KSPACE = 5e-12` with the reason; the short-range seams keep
  2e-13.
- Job 3978849, the final run of `verify_gh200_hot_paths.py` on this state: 80
  checks pass, 0 fail, 2 skipped (the non-periodic Polar path, by design);
  every CUDA-graph parity recorded a graph; the periodic Polar force replay
  came in at 1.7e-12 eV/A.
- The slow compile tests cannot run on the node: they are CPU tests and cue's
  aarch64 wheel has no CPU kernels (job 3945974 parts 2/3); the sbatch runs
  the verification script only.

**Status.** active.

<!-- mol:note:topic:no-compat-before-one -->
## [2026-10-17] No global precision, no schema version, no "site": molnex's compat removal

**Decision (operator, 2026-10-17: "all of it, no backward compatibility").**
The 2026-10-07 ruling for the sibling repos (no schema versions, migrations,
shims or legacy spellings before 1.0) now has its molnex section, landed as
one commit per step on `dev`. Four rulings went into CLAUDE.md's Forbid list:
no global precision singleton; the `graphs` namespace is part of every batch
and the graph count is `frame["graphs"].batch_size[0]`; the per-atom energy is
the atom energy and nothing is versioned; `getattr` / `hasattr` / `__getattr__`
only at an external-format boundary.

**What went.**
- `molix.config` (molcfg singleton, `set_precision`, `ftype` / `itype` /
  `use_amp`): 73 `config.ftype if dtype is None else dtype` fallbacks in 35
  modules became `dtype: torch.dtype` required; `itype`, `use_amp`,
  `amp_dtype` had no consumer (autocast is `Trainer(autocast=)`); the collate
  boundary already took an explicit dtype. `molcrafts-molcfg` left
  `pyproject.toml`. Tests give dtypes as module constants.
- `molpot.potentials.base.BasePotential`: `calc_energy`, `calc_forces`,
  `_get_positions`, the `data=` dict form, the legacy `atoms.x` read and the
  `**kwargs` catch-alls of the seven term kernels; terms take keyword-only
  tensors (`pos`, `<term>_index`, `<term>_types`, `term_batch`, `num_graphs`,
  tables) and share one `per_graph` reduction. `ImproperHarmonic` had been
  returning a scalar into a `(B,)` sum through the catch-all — fixed.
- `scatter_sum` / `batch_add` `dim_size` and every `num_graphs` are required;
  the `index.max().item() + 1` fallbacks (scatter, nonbonded, polarization,
  lj126, gto features / hartree, the forcefield bind) are gone; the MD driver
  gives a system its `graphs` namespace in `_cast_system`.
- `_mark_leaves`'s `physics(batch)` hook (no definer anywhere); the five
  hand-rolled `_orig_mod` / `.module` unwraps are `molix.compile.unwrap`
  (isinstance on `OptimizedModule`, DDP, FSDP); loss-term publishing is typed
  on `WeightedLoss`. `SubsetDataset.__getattr__` / `PreparedRows.__getattr__`
  are explicit delegations (`task_labels` / `group_ids` gathered at the view's
  indices; `PreparedRows` is a `BaseDataset`). `JournalHook(writer=)`.
- Deploy: `SiteEnergyModel` / `SiteEnergyAdapter` / `site_energy` are
  `AtomEnergyModel` / `AtomEnergyAdapter` / `atom_energy` in Python and in
  `interface/` (runtime `atom_energy_exec`, pair_style, ctests: 69 pass on the
  x86 tree); `DeployMetadata` has no `schema_version` and ignores unknown
  fields; `molpot.manifest` lost `MANIFEST_SCHEMA` and its schema-1 upgrade.
- `MultipoleHead.embed_moments` (a reserved flag that raised),
  `EwaldMultipoleEnergy.compute_field`, `Sonata.from_spec` / `from_encoder`,
  `Rondo.table_columns`, `molpot.derivation.force` (`ForceDerivation`;
  `hessian.py` calls autograd directly).

**Potentials on the two-method convention.** `Sonata(spec, encoder, *,
short_range_heads, wrt, dtype)`, `PotentialComposer(head, potentials, *, wrt)`
and `Rondo` subclass `molix.autograd.Potential`: `energy(frame)` runs the
pipeline in place, `forward = Derivative(energy, wrt)`; forces and the virial
come from one forward instead of the `energy_of_pos` closure that re-ran the
pipeline. `Sonata.bind_cell` enumerates the k-vector keep-set once (the former
per-call `kvec_indices`). Rondo writes `graphs.energy` and `atoms.forces_pred`;
placing predictions beside labels is the joint step's job
(`molix.frame.beside_labels`). `Derivative` now detaches every column the
energy wrote after the single backward (a key-set diff around the call, no
`grad_fn` probing — that probe was itself a dynamo break): a potential that
writes extra columns traces to one graph again (`test_rondo/test_compile.py`).

**Hot paths.**
- Molecular Hartree pairs: `gto.hartree.graph_pairs(batch, num_graphs,
  capacity)` enumerates the off-diagonal grid of every graph's contiguous
  block — `B·K(K−1)` rows, `K = max_atoms_per_graph` (×4 on the copy layout),
  counts from `index_add_`, indices clamped, a keep mask for the padding,
  `orig[src] != orig[dst]` ANDed in by the consumers; overflow and a
  non-contiguous `atoms.batch` raise through `torch._assert_async` (traces
  clean). `r` is replaced by a safe distance behind the mask so coincident
  atoms of different graphs stay finite in forward and backward. The capacity
  is bound like the cell (`GTOHartreeEnergy` / `GTOFieldFeatures` /
  `MacePolar.bind_capacity(max_atoms_per_graph)`; unbound raises) — a batch
  layout shape, not a model hyper-parameter, so it stays out of specs and
  checkpoints. No host cache key, no `.item()`, no data-dependent shape: the
  Polar molecular `forward` traces to 1 graph / 0 breaks (42 / 41 before) and
  records into a CUDA graph. A first cut masked every pair of the whole batch
  (commit 8d8610b); that is B× the rows a batch of B molecules needs — the
  2026-09 note on this break had already worked that out — and was replaced
  the same day (4edcaa8).
- `ElementLinear.weight` is `(C, Σ u·w)` and `OneHotLinear.weight` `(C, F)`:
  one row gather and a `bmm` per path, no per-call view / transpose of cue's
  flat row. The flat layout is converted once at the official-checkpoint
  boundary (`molzoo.mace.checkpoint.element_linear_weight` /
  `one_hot_weight`, selected by `CheckpointRemap`'s type-driven rule through
  `ConvertedWeight`; any other shape is refused). GH200 `bench_mace_matpes.py`
  (random weights, 30 steps; job 3983478 before, 3993200 on the final tree):
  CUDA graphs 7.44 → 7.39 ms (N=192 fp32), 11.13 → 10.97 (fp64), 20.18 →
  20.17 (N=1024 fp32), 28.22 → 28.12 (fp64); eager 22.5 / 22.7 / 29.8 / 35.9
  → 26.2 / 25.1 / 29.2 / 35.7 (eager swings ±15 % between runs on the shared
  node, the CUDA-graph numbers do not). Within noise: the skip_tp gather is not
  where MACE's time goes.
- GH200 verification on the final tree (`verify_gh200_hot_paths.py`, jobs
  3992992 / 3993199): 83 of 85 checks pass; the molecular Polar path now
  records a CUDA graph (per-atom energies replay to 1e-14, explain 1 / 0) and
  the two misses were gate sizes — its graph energy / force replay at
  1.2e-11 / 1.3e-11 eV against a 1e-11 gate (the per-graph Hartree pair sum
  reordered by inductor; gate now 5e-11 with the measured range), and the
  OAM-L fp32 virial at 2.1e-13 against 2e-13 (gate now 5e-13). A first final
  run (3992993) crashed every compiled benchmark with "returning intermediate
  with requires_grad_()": the key-set version of the Derivative detach missed
  columns the energy overwrote (a batch that already carried `atoms.energy`
  kept it attached) — hence the identity-based detach in f6d8adf.

**Official-weight parity is now against upstream.** The MACE weight assets
live in `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/.build-alloc/mace-weights/`
(`MOLNEX_MACE_WEIGHTS_DIR`): MatPES config + cueq state, OMol / OFF24-medium /
LES-OFF-small cueq states, their config jsons, and `*_cluster_reference.pt`
computed with upstream mace-torch (0.3.16 in `.venv-convert`; LES with 0.3.17
in `.venv-les`) on the fixture cluster (`REFERENCE.md` beside them). The slow
tier compares energy and the whole force array: MatPES 1e-14 / 6e-14, OFF24 0 /
6e-15, OMol 5e-8 / 2e-7, LES 2e-7 / 2e-6 eV / eV/Å (the LES gap is the
fp32-rounded constants of its pickle; tolerances 1e-6 / 1e-5). The
self-measured OMol goldens are gone. Found on the way: `MaceOfficialConfig`
only read the pickled module's `interaction_blocks`; upstream's json dialect
(`interaction_cls_first` / `interaction_cls` / `num_interactions`) was
refused, so real OFF24 / LES configs could not load — fixed.

**Follow-ups (not done).** `Potentials.calc_energy(system, tables)` is a
second energy entry on a `Potential` (a tensor return plus the F29 whole-molecule
geometry check, ~100 test call sites): fold the check into `energy` and delete
it, so the classical container has the same two methods as everything else.
Padded Polar training batches and the capacity:
`MoleculePadder` files ghosts in a sink slot (`max_graphs`) or at the tail of
graph 0, so a padded molecular batch trips the overflow or contiguity assert
— the clean fix counts live rows from `atoms.mask` (a `mask` argument to the
GTO forwards and `MacePolar.energy`), not done. An isolated-molecule Polar MD
run must call `bind_capacity(N)` itself. `Potentials.partial_hessian` still drives terms by
`getattr(self, attribute)`; `_atom_batch` zero-fills when `atoms.batch` is
absent; `PreparedRows._parent` probes molhub's subset view by `getattr`;
`EwaldMultipoleEnergy.forward` loops `torch.unique(batch)` with `.item()`, so
`Sonata` is not compile-clean; the MD driver and its tests still accept both
`graphs` layouts (`batch_size ()` with a `(3, 3)` cell and `batch_size (1,)`
with `(1, 3, 3)`) — one layout is the next cut; `mace/utils.py` reads the
upstream pickled module by `getattr` (the allowed boundary);
`test_rondo/conftest.py::excite` and `joint.py:242` keep a `getattr`.

**Status.** active.

<!-- mol:note:topic:native-channelwise-csr -->
## Native indexed ChannelWiseTensorProduct — CSR schedule, compile-time terms, fused adjoints (2026-10-23)

Operator ask (2026-10-23 evening): own the algorithm, data flow, scheduling and
differentiation of `molrep.equivariance.ChannelWiseTensorProduct`; reuse a
mature primitive only where it wins end-to-end; evaluate beating cuEq. The
morning's grouped-atomics backend (below, superseded) was 5–8× slower than cuEq
on the op; this is its replacement. Still opt-in: MACE constructs cuEq's
operation by default; the benchmark harness swaps `conv_tp` only.

**Design.** `csrc/equivariance/terms.hpp` enumerates every Clebsch–Gordan
term at compile time for MACE's natural-parity O3 family
(`U×0e+U×1o+…+U×L1`, `0e+…+L2`, filter `0e+…+L2`; L1 0..3, L2 2..3,
U 32/64/128) from the selection rule of the real spherical-harmonic basis
(`m3 ∈ {±(|m1|+|m2|), ±||m1|−|m2||}`, cosine/sine type by the factors' types,
zonal factor passes the other's m through) — verified against cue for every
l ≤ 4: 0 violations, 1164 candidates for 1158 non-zeros. The CG *values* come
from cue's descriptor at construction and travel as kernel parameters
(constant-bank FMA operands; no device buffer, no per-call copy). cue sorts the
output irreps by l3 and the weight segments follow (wi == oi); both sides map
path → sorted segment and the Python descriptor refuses any descriptor whose
structure differs. Kernels (`channelwise.cuh`): `product` — one CTA per
target row, one thread per channel, the row's incoming edges summed in
registers in the schedule's stable order and written once; `adjoints` — one
fused pass by target returning (ga, gb, gw): the cotangent row is read once per
node (it is four times an edge's weights at MACE widths), gw/gb written per
edge, ga written per edge and summed by source (`segment_sum`); the same kernel
with `second=True` and the cotangents (da, db, dw) is the exact double backward
in one launch (fp64: two passes, grad_g and the edge adjoints, because 340
registers do not fit). `schedule` — CSR by target and by source via CUB's
narrow-key radix sort (⌈log2 rows⌉ bits) + a binary-search rowptr, int32,
built once per batch (the MACE encoder builds it in `layer_features` and
threads `schedule=` through the interaction blocks when `conv_tp` is native).
Per-chunk shared-memory staging of edge ids, source rows and the angular rows
removes the dependent chain order → source → row and the per-edge scalar
registers; the edge loop is unrolled 2× for the product and first order so the
next edge's loads overlap the contraction; rows split over up to 8 CTAs
(`row_splits`, partials reduced) so 193 atoms still fill the device. No
atomics anywhere: bitwise reproducible, deterministic mode honoured.
Autograd: `product.backward = adjoints`, `adjoints.backward = adjoints(second)`;
third order raises. Meta kernels for both; fullgraph Inductor and CUDA-graph
replay tested.

**What cuEq does (nsys of its uniform_1d):** no sort — three JIT kernels
(fwd 1.25, bwd 1.09, bwd_bwd 1.80 ms at the layer-1 wat1544 shape), the bwd
with atomics; each ~3× above the HBM roofline (w is 959 MB per pass).

**Standing (job 4314788 = v7, the delivered kernels; fp32 unless noted;
same-job pairs, GH200, MACE-MatPES official weights, capacity edges, TF32
off).** Layer-1 wat1544 (E = 187 316): forward 0.48 vs 1.05 ms (2.2×; cuEq's
forward varies 1.05–1.25 across nodes, ours 0.48–0.50); forward+backward 1.68
vs 2.18 (1.29×); +second 4.39 vs 4.72 (1.08×); fp64 1.15× / 1.00× / 0.75×
(second order after the two-pass split). Full E/F, Inductor fullgraph +
CUDA graph, schedule built once per forward: 4425 atoms 19.7 vs 22.9 ms
(1.16×), 1544 atoms 7.69 vs 8.40 (1.09×), 193 atoms 1.59 vs 1.50 (0.95×);
fp64 1.09× / 1.04× / 0.90×. Correctness vs cuEq: normalized RMS ≤ 5e-15
(fp64), ≤ 5e-6 (fp32) on every cell; 32/32 GPU tests incl. gradgradcheck,
fullgraph explain, CUDA-graph replay with in-place index changes, bitwise
reproducibility. Kernel times per call (fp32, L1=1, v7 nsys): product
0.371 ms, first-order adjoints 1.066 + segment_sum 0.129, second order 2.062
+ 0.129; schedule ≈ 45 µs. Registers (fp32, L1=1, U=128): product 122,
first-order 161, second-order 255, no spills; the fp64 first order spills
104 B at 255 and the fp64 second-order edge pass 292 B. Launch-bound
calibration (job 4314633): forcing the first-order float adjoint to 128
registers spills and loses 8 %; the fp64 product at 168 spills and loses 20 %
— kept at 3 / 2 CTAs per SM. Version trail (records/v1..v7): v1 CSR +
compile-time terms (op 5–8× faster than the old backend, full E/F 0.9×
cuEq); v2 fused adjoints; v3 shared-memory index staging; v4 2× edge unroll +
row splits + the hoisted schedule; v5 angular rows in shared memory; v6 fp64
two-pass second order; v7 launch bounds.

**Open.** 193 atoms is occupancy-bound (one CTA per row, ~2 µs per
serialised edge even at 3 CTAs per row); the second-order float kernel is at
255 registers (2 CTAs/SM) and 1.15× cuEq's bwd_bwd; fp64 second order 0.78×.
The default backend stays cuEq until the operator decides. Evidence, snapshots
and job scripts:
`/nobackup/proj/disk/teoroo/personal/jicli594/work/lab/projects/molnex-cwtp-csr/20261023/`
(`records/v1..v7`, `summarize.py`, `jobs/v1.sbatch` with SNAP/OUT/BUILD).

<!-- mol:note:topic:native-channelwise-hopper -->
## Superseded: the grouped-atomics channelwise backend (2026-10-23 morning)

Kept for the record; the kernel is gone from the tree (only the experiment
snapshots under `molnex-cueq-replacement/20261023/` and
`molnex-op-profiler/20261023/` retain it). It grouped cue's CG paths by
destination, ran one CTA per (group, edge tile) with atomics into the node rows
and re-read every operand once per group; after tiling and metadata staging it
was still 2.5–3.1× slower than cuEq on full E/F and 5–8× on the op (jobs
4303158, 4309927, 4310025, 4310029). Its validation apparatus — paired
microbenchmarks on frozen frames, `report.py` / `audit.py` fingerprints, the
C++ AOTI adjoint smoke — carried over to the replacement above.

<!-- mol:note:topic:native-op-profiling -->
## Native operator profiler and first Hopper optimization (2026-10-23)

`molix.profiler.OperatorProfiler` separates preparation (fullgraph compilation,
warmup and optional CUDA graph), CUDA-event timing, Torch tracing, and a bounded
CUDA profiler API/NVTX capture window. The caller defines forward/adjoint/second
derivative workloads and numerical checks. Default timing is five groups of at
least 100 calls and 200 ms each after 30 warmups; reported percentiles describe
group averages. First-call time and warmup/capture memory peaks are separate.
Use exact tensor fingerprints, including intermediate features and derivative
directions, rather than assuming a frozen molecular frame implies bitwise
identical intermediate inputs after atomic aggregation.

`molnex profile doctor`, `nsys` and `ncu` expose the independent NVIDIA wrappers.
They retain raw reports, stdout/stderr, tool versions and argv; export CSV plus
normalized JSON; reject reused output directories; and kill their owned process
group on capture timeout. Workloads call `OperatorProfiler.capture` after
preparation. Profilers are separate runs, never a source of speedup claims.
`MOLNEX_EQUIVARIANCE_LINEINFO=ON` adds line information without `-G`. See
`docs/molix/user-guide/profiling.md` for the reusable API and CLI examples.

GH200 calibration artifacts, immutable candidate snapshots, scheduler scripts
and raw reports are under
`/nobackup/proj/disk/teoroo/personal/jicli594/work/lab/projects/molnex-op-profiler/20261023/`.
The latest candidate is `snapshot-dispatch` / `build-dispatch`, with measurements
in `release/`; `final/` is an earlier candidate before size dispatch, not the
delivered result. `report.py` verifies same-allocation old/new/cuEq comparisons,
exact frame/operand fingerprints, correctness budgets and timing protocols.
`audit.py` additionally checks source/library hashes, reports and job exits.

The kernel-level findings of that first optimisation round (tiles, staged
metadata, size dispatch; 12–17 % on the old backend) are superseded by the
rewrite in topic `native-channelwise-csr`; the profiler, its CLI wrappers and
the paired-measurement protocol are what remains in use.
