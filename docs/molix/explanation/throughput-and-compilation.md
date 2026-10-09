# Training Throughput & `torch.compile` (developer notes)

A developer-facing report on what limits PiNet training throughput, how the
`torch.compile` front/back ends behave on it, the fixed-length-padding + CUDA
graphs path that recovers ~5x, and the compile configurations that fail (and
why). Measured on an NVIDIA GH200 (Hopper), revMD17/aspirin, `batch_size=32`,
fp32.

## The workload

- **Model**: `molzoo.PiNetPotential` — a PiNet2 encoder (`depth=5` GC blocks,
  `rank=3`, `hidden_dim=64`, atom types H/C/N/O) with an energy head. **Forces
  are derived with functorch** (`torch.func.grad` of the energy w.r.t. positions,
  in `_forward_functorch`) — no double-backward barrier, but it means each step
  runs the energy graph **twice** (energy pass + grad pass).
- **Batch**: paper protocol `batch_size=32`. For aspirin that is 32×21 = 672
  atoms and ~6k–8k edges (geometry-dependent). The atom/edge counts vary per
  batch → shapes are *ragged* unless padded.
- **Step**: forward → loss (`energy_force_mse`) → backward → Adam.

## The bottleneck: launch-bound, not compute-bound

Profiling one compiled step (`torch.profiler`) at `batch_size=32`:

| metric | value |
|---|---|
| graph breaks | **0** (compile captures one graph) |
| GPU kernels launched / step | ~1887 |
| `cuLaunchKernel` / step | ~1422 |
| of which tiny GEMMs (`mm`+`bmm`+`addmm`+`gemv`) | ~770 |
| GPU busy time / wall time | **31 %** (GPU idle 69 % of the step) |

PiNet's many small per-block MLPs at `batch_size=32` produce hundreds of tiny
matmuls. Each kernel computes for only a few µs but every launch costs ~5–10 µs
of CPU→GPU dispatch, so **per-step time is dominated by launch overhead** and the
GPU is starved (~23–31 % utilised). `torch.compile`'s fusion already does all it
can (0 graph breaks); it cannot fuse cuBLAS GEMMs, and the per-launch cost
remains. *(A red herring ruled out by measurement: ~224 `.item()` syncs/step from
the default Adam step-counter — removing them via `fused`/`capturable` Adam
changed throughput by <3 %. It is the launches, not the syncs.)*

The only way to remove launch overhead is **CUDA graphs**: record the whole
step's launches once and replay them as a single submission. CUDA graphs require
**static shapes**, which is what the fixed-length padding provides.

## `torch.compile`: front end vs back end

`torch.compile` is a pipeline with two halves:

- **Front end (capture)** — *TorchDynamo* traces Python bytecode into an FX graph;
  *AOTAutograd* traces the joint forward+backward graph and functionalises it.
  `fullgraph=True` requires the front end to produce **one** graph (error on any
  graph break); `dynamic=True/False/None` controls whether dimensions are traced
  symbolically (dynamic) or specialised to concrete sizes (static).
- **Back end (lowering)** — turns the captured graph into runnable kernels. We
  benchmarked three:
  - **`inductor`** (default): generates fused Triton/C++ kernels — collapses many
    small ops into fewer kernels.
  - **`aot_eager`**: runs the captured ATen ops eagerly, **no codegen / no
    fusion** — useful to isolate inductor's fusion benefit.
  - **`cudagraphs`**: wraps the (unfused) ATen graph in CUDA graphs.

`mode` is an **inductor-only** preset (it is ignored/invalid for other backends):

- `default` — fuse, no CUDA graphs.
- `reduce-overhead` — fuse **and** wrap in CUDA graphs.
- `max-autotune` — autotune GEMM kernels (benchmark many Triton tile configs) +
  CUDA graphs.
- `max-autotune-no-cudagraphs` — autotune without CUDA graphs.

## Sweep results (`fullgraph=True`, padded static shapes, via `Trainer.compile`)

steps/s = true wall-clock, one config per fresh process, no hooks.

| backend | dynamic | mode | steps/s | note |
|---|---|---|---|---|
| inductor | False | default | 33.6 | fusion, no CUDA graphs |
| inductor | False | **reduce-overhead** | **155.8** | ★ fusion + CUDA graphs |
| inductor | False | max-autotune | **FAILED** | shared-mem OOM (below) |
| inductor | False | max-autotune-no-cudagraphs | **FAILED** | shared-mem OOM |
| inductor | True | default | 34.0 | dynamic ≈ no effect on padded shapes |
| inductor | True | max-autotune-no-cudagraphs | **FAILED** | shared-mem OOM |
| aot_eager | False | — | 17.3 | no fusion |
| aot_eager | True | — | 17.4 | |
| cudagraphs | False | — | 36.3 | CUDA graphs **without** fusion ≈ inductor-default |

Auto-excluded (incompatible; never run):

- `mode != default` with a non-inductor backend → `mode` is inductor-only.
- `dynamic=True` with `reduce-overhead` / `max-autotune` / the `cudagraphs`
  backend → CUDA graphs require static shapes.

Takeaways:

1. **`fullgraph=True` works on every runnable config** — the functorch force path
   compiles to a single graph (0 breaks). It is now used in production.
2. **Best = inductor + `reduce-overhead` + `dynamic=False` + `fullgraph=True` =
   155.8 steps/s.** Fusion **and** CUDA graphs together are what give the ~5x over
   the ~33 steps/s default (and ~25–34 on the non-padded variable-shape path).
3. **`dynamic` on/off is irrelevant on padded fixed shapes** (33.6 vs 34.0).
   Shapes are already static; `dynamic=True` only makes compilation slower.
4. **Fusion matters**: inductor 33.6 vs aot_eager 17.3. The `cudagraphs` backend
   alone (graphs, no fusion) only reaches 36.3 — close to inductor-default. You
   need fusion + graphs in the *same* backend (`reduce-overhead`).

## The error: `max-autotune` shared-memory exhaustion

All three `max-autotune` configs fail to compile with:

```
No valid triton configs. OutOfMemoryError: out of resource: triton_mm
  Required: 524288  Hardware limit: 232448
  Reducing block sizes or `num_stages` may help.
```

### What "shared memory" is — and why 96 GB doesn't help

This is **not** the 96 GB of HBM (global device memory / "显存"). A GPU has a
memory hierarchy:

| level | what | size (Hopper / GH200) | speed |
|---|---|---|---|
| registers | per-thread | KB per thread | fastest |
| **shared memory** | **per-SM on-chip SRAM scratchpad**, shared by a thread block | **~228 KB per SM, max ~227 KB usable per kernel (232448 B here)** | ~TB/s |
| L2 cache | chip-wide | ~50 MB | fast |
| **global memory (HBM)** | the **96 GB** "显存" | 96 GB | ~4 TB/s but high latency |

A Triton matmul kernel stages its operand tiles in **shared memory** to reuse
them across the block. The amount needed is roughly
`(BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · dtype_bytes · num_stages` (software
pipelining keeps `num_stages` copies in flight). `max-autotune` benchmarks
aggressive tile configs; some demand **524288 B (512 KB) of shared memory per
SM**, but a Hopper SM exposes only **232448 B (~227 KB)** to a kernel. That tile
therefore has *no valid launch configuration*. When **every** autotune candidate
for that GEMM exceeds the limit, inductor reports "No valid triton configs" and
the compile aborts.

So the failure is a **per-SM on-chip SRAM limit**, completely independent of the
96 GB HBM — the 96 GB is global memory; shared memory is the small fast
scratchpad inside each of the GPU's SMs. Plenty of HBM cannot relieve a
shared-memory-per-SM shortfall.

### Why `reduce-overhead` is fine but `max-autotune` is not

`reduce-overhead` uses inductor's default GEMM templates (modest tiles that fit
in 227 KB) plus CUDA graphs. It never requests oversized tiles, so it compiles
and runs (155.8 steps/s). `max-autotune` only adds *kernel autotuning* on top —
which, for this model's GEMM shapes on this GPU, generates only oversized-tile
candidates and fails. **Use `reduce-overhead`, not `max-autotune`, here.**

## Production configuration

A training entry point such as `pqtn.training.train_pinet` builds the compiled
model with `molix.compile.Compiler`, which returns a *new* compiled object of
the same type it was handed:

```python
from molix.compile import Compiler

if cfg.cuda_graphs:                                  # fixed-length padding path
    model = Compiler(cuda_graphs=True)(model)
else:                                                # variable-shape path
    model = Compiler(dynamic=True, fullgraph=True)(model)
```

Inside MolNex, the winning combo is the named `cuda_graphs` preset — use
`trainer.compile(cuda_graphs=True)` (or `molix.compile.Compiler.CUDA_GRAPH_PRESET`)
instead of spelling out the four flags, so the config can't drift.

A model that derives forces is the same object: `Derivative(model,
wrt=("atoms.pos",))` or `PiNetPotential(..., wrt=("atoms.pos",))` — one
`nn.Module`, compiled with the same `Compiler(...)(model)` /
`trainer.compile(...)` and handed to `Trainer(...)` like any other. There is
no separate compile seam for the force path: leaves, energy and the one
`autograd.grad` are one region.

**Migration note.** `PiNetPotential.compile(...)` has been removed. It shadowed
`nn.Module.compile`, whose contract is different: the base method compiles
**in place** and returns `None`, so `model.compile(...)` without rebinding the
result silently kept running eager. Use `molix.compile.Compiler` whenever you
need a new compiled object (`model = Compiler(...)(model)`), and
`nn.Module.compile` only for its in-place effect.

`cuda_graphs=True` additionally: compose `FrameCollator` with `MoleculePadder` (pads atoms→
`max_atoms`, edges→`max_pairs`, with a masked ghost region so energy and
real-atom forces are numerically identical to the unpadded batch — see
`tests/test_molix/test_data/test_pad.py`), and sets `train_drop_last=True` so the
graph count is fixed too. Set `max_atoms` / `max_pairs` above the largest
per-batch counts; overflow raises rather than truncating.

## Real Trainer run of the winning combo

`backend=inductor, fullgraph=True, dynamic=False, mode=reduce-overhead`, padded
fixed shapes (bs=32 aspirin), via `Trainer.compile` + `trainer.train`, measured
true wall steady-state:

| hooks | steps/s | ms/step |
|---|---|---|
| FULL production stack (metrics E+F, step-speed, gpu-mem, async journal, grad-clip, checkpoint; `journal_every=1000`) | **129.8** | 7.70 |
| none (reference) | 132.2 | 7.57 |

At production journal cadence the full hook stack costs only ~2 % (129.8 vs
132.2) — the earlier ~91 steps/s figure was an artifact of a deliberately
frequent `journal_every=20`. So the real production throughput with the winning
combo is **~130 steps/s** (≈10x eager's ~12, ≈4–5x plain compile's ~25–34).

## Does this transfer to cuEquivariance models? (MACE-MatPES, 2026-08-07)

The sweep above is PiNet: **pure torch**, so dynamo has nothing exotic to trace.
MACE runs cuEquivariance fused kernels, which are custom `autograd.Function`s —
a plausible reason for dynamo to break the graph and for `fullgraph=True` to
fail outright. Measured on `molzoo.MACEMatpes` with official
`mace-matpes-r2scan-0` weights, 193-atom periodic water box (17344 edges), one
GH200, compiling `_compute_energy` with `autograd.grad` taken *outside* the
compiled region:

| | fp64 ms (step/s) | fp32 ms (step/s) |
|---|---|---|
| eager | 35.7 (28.0) | 35.0 (28.6) |
| inductor default | 19.0 (52.5) | 19.6 (50.9) |
| **inductor + `reduce-overhead`** | **4.18 (239)** | **2.40 (417)** |
| + `fullgraph=True` | 4.19 (239) | 2.40 (417) |

**It transfers unchanged.** `CUDA_GRAPH_PRESET` is the winner here too (8.5x
eager at fp64, 14.6x at fp32).

Three things worth knowing:

1. **cuEq does not break the graph**: `graph_breaks=0, graphs=1, ops=381`.
   `fullgraph=True` is therefore free rather than beneficial — it closes no
   breaks, it only asserts there are none. Keep it as the assertion.
2. **Compiling fp64 is numerically free**: ΔE = 0, ΔF = 2.6e-14 against eager.
   fp32 shifts results ~2e-3 eV / ~1.6e-3 eV/Å (inductor fusion reassociates
   float adds) — expected, not a defect.
3. **Precision is invisible until the launches are gone.** Eager fp32 ≈ eager
   fp64 (35.0 vs 35.7 ms) because the step is latency-bound. Under CUDA graphs
   fp32 is 1.75x fp64. Anyone benchmarking precision *in eager mode* on a small
   system will wrongly conclude precision does not matter.

Static shapes come free for MD here, and they are the MD default. Open-system
runs freeze the neighbour list for the trajectory; periodic runs rebuild it on
a cadence into fixed-capacity buffers (`molix.md.NeighborList` — contents
change in place, shapes never do), and the MD driver hands the composed
derivative op that **whole** buffer on every step, dead rows included (they sit
at the dead-edge displacement and price exactly nothing). Either way no padding
registry is needed, unlike the training path, and `Compiler(cuda_graphs=True)`
is the MD compiler to use: `MD` applies the replay contract (a
`cudagraph_mark_step_begin()` before each call, batch-side outputs cloned out
of the static pool) whenever the installed compiler reports `cuda_graphs`, and
keeps the rebuild eager, outside the captured region. `dynamic=True` is for
ragged training batches that are not padded — not for MD. The measured
MD-path numbers (eager vs default inductor vs the CUDA-graph preset, through
the real neighbour list and driver op) live in `.claude/notes/notes.md`, topic
`compile-no-graph-breaks`.

## Compiling the step

`Trainer.compile(...)` compiles the training **step**, not the model
(`step=True`, the default): the traced body of `train_step.forward_loss` and
`eval_step.forward_loss` — `predictions = model(batch)` followed by
`compute_loss` — is swapped in place for its `torch.compile` wrapper under
the `Compiler` config. Dynamo captures forward and objective as one region and
AOTAutograd emits its backward, so `loss.backward()` replays a compiled graph
too. Everything else in `DefaultTrainStep.on_train_batch` stays eager Python
around the call: the accumulation window, `GradScaler`, the
`on_after_backward` hooks, the optimizer and the LR scheduler. The autocast
context wraps the compiled call from *outside* — dynamo guards on the autocast
state and captures the ops under it, and the `config[...]` read that resolves
it never enters the trace. `Trainer.smoke` warms up the same attribute `train`
runs, so the smoke pass compiles exactly the training graph (and, through
`eval_step.on_eval_batch`, the eval graph).

```python
model = Derivative(MaceMatpes(...), wrt=("atoms.pos",))       # or MaceMatpes(..., wrt=...)
trainer = Trainer(model, energy_force_mse(...), optimizer_factory, ...)
trainer.compile()                   # inductor — one graph, zero breaks on stock torch
trainer.compile(cuda_graphs=True)   # padded fixed shapes: the fastest force-training config
trainer.compile(step=False)         # model only, the loss stays eager
trainer.smoke(datamodule)
trainer.train(datamodule, max_steps=...)
```

**The caller-owned leaf.** `forward_loss` is two things: an eager prologue and
the traced body. The prologue finds the derivative op the model runs — the
model itself when it is a `Derivative`, otherwise the potential's `derivative`
property — and calls `op.mark_leaves(batch)`, which replaces every variable's
column (`atoms.pos`, and `graphs.strain` for a virial) by a fresh live leaf
*before* anything is traced. The op reuses a column that already is a live
leaf instead of making one, so inside the region the leaf is a graph input:
the neighbour geometry rebuilt from it, the energy, the one `autograd.grad`,
the writes onto the frame and the loss trace as **one graph with zero breaks
on stock torch 2.15**, forces attached to the parameters. (A leaf made inside
the region would leak through those attached outputs — "requires_grad_()
intermediate leaked as output", the break pytorch/pytorch#196653 lifts — which
is why the leaf is made outside.) A model with no op, or an energy-only op
(`wrt=()`), needs no prologue. Because the region is break-free,
`fullgraph=True` and `cuda_graphs=True` work for force training on stock
torch; the measured numbers (eager / inductor / CUDA graphs on a 256-atom tiny
MACE, fp64, GH200) are in the notes topic `compile-no-graph-breaks`. The slow
gates: `tests/test_molix/test_core/test_steps/test_train.py::TestDefaultTrainStepTrace`.

**Which losses trace.** Anything with no tensor-valued Python control flow:
`energy_mse`, `energy_force_mse`, `MSELoss`, `MAELoss`, `WeightedLoss`,
`ChargeMSELoss`, `ScaleNormalizedLoss`, and the body of
`ProjectedInternalHessianLoss`. Not `center_by_group` and what is built on it
(`molecule_centered_energy_mse`, `TorsionProfileLoss`): they loop over
`group_ids.unique()` with boolean-mask indexing — a data-dependent shape per
group. A step running one of those compiles the model alone
(`compile(step=False)`).

**The router seam.** `MultiTaskTrainStep` / `MultiTaskEvalStep` resolve the
task in the eager prologue with `torch.unique(labels).tolist()` — one host
sync per batch, the named break. `_traced_forward_loss` is the model plus
the already chosen loss, so `compile(step=True)` traces that body and dynamo
specializes it once per task. A loss that itself does not trace
(`TorsionProfileLoss`, `center_by_group`) still does not.
