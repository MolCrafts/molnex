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
tests/test_molix/test_data/test_tasks/test_pad.py
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
