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

`pqtn.training.train_pinet` compiles the model in place:

```python
if cfg.cuda_graphs:                       # fixed-length padding path
    model.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
else:                                      # variable-shape path
    model.compile(dynamic=True, fullgraph=True)
```

Inside MolNex, the winning combo is the named `cuda_graphs` preset — use
`trainer.compile(cuda_graphs=True)` (or `molix.compile.Compiler.CUDA_GRAPH_PRESET`)
instead of spelling out the four flags, so the config can't drift.

`cuda_graphs=True` additionally: registers `PadMolecularBatch` (pads atoms→
`pad_max_atoms`, edges→`pad_max_edges`, with a masked ghost region so energy and
real-atom forces are numerically identical to the unpadded batch — see
`tests/test_molzoo/test_pinet_padding.py`), and sets `train_drop_last=True` so the
graph count is fixed too. Set `pad_max_atoms` / `pad_max_edges` above the largest
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

Static shapes come free for MD here. Open-system runs freeze the neighbour
list for the trajectory; periodic runs rebuild it on a cadence into
fixed-capacity buffers (`molix.md.PeriodicNeighborList` — contents change in
place, shapes never do). Either way no padding registry is needed, unlike the
training path.
