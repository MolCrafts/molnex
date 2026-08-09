# Profiling

`molix.profiler` is a suite of five standalone measurement tools. Each one
answers a single "where does the time go?" question about one layer of the
stack, and none of them asks you to wire up a real training run first.

| Profiler | Question it answers |
| --- | --- |
| `TaskProfiler` | How long does one preprocessing task take per sample? |
| `ModuleProfiler` | How long does one `nn.Module` take per forward and backward pass? |
| `DataLoaderProfiler` | How long does the consumer block waiting for the next batch? |
| `TrainerProfiler` | How much per-step time does `Trainer` add on top of a bare loop? |
| `DatasetProfiler` | What is in my dataset, and what does reading one sample cost? |

All five have the same shape: **configuration goes into the constructor, data
goes into `run()`, and `run()` returns a result dataclass that knows how to
print itself.**

```python
from molix.profiler import DatasetProfiler

profiler = DatasetProfiler(n_samples=200)   # configuration
result = profiler.run(dataset)              # data
result.print_report()                       # prints to stdout, returns None
```

Every field on a result object is a plain attribute, so anything the report
prints can also be asserted on in a script — `result.avg_num_neighbors` from
`DatasetResult`, `result.timing.p95_ms` from `TaskResult`,
`result.overhead_ms_per_step` from `TrainerResult`, and so on.

This suite is not the same thing as `ProfilerHook` in `molix.hooks`. That hook
wraps `torch.profiler` around a *live* training run and writes a Chrome trace
file; the profilers here run outside training, on components you hand them
directly.

## Reading the Reports

A few conventions are shared by every report.

**Warmup.** The first few iterations of anything in PyTorch are unrepresentative:
memory allocators grow their pools, kernels are selected and cached, and
memory-mapped file pages are faulted in from disk. Each profiler therefore takes
an `n_warmup` count of iterations that are executed but discarded before timing
starts.

**Wall-clock time.** Unless a report says otherwise, times are wall-clock
milliseconds measured with `time.perf_counter` — real elapsed time, not CPU
time. `mean` / `std` / `p50` / `p95` are the mean, standard deviation, median
and 95th percentile over the measured iterations. `p95` matters more than the
mean for anything that stalls: a loader that is fast on average but occasionally
blocks for 200 ms will show a mean close to zero and a large `p95`.

**The `Data:` line.** Reports name their input by calling `describe()` on it if
it has one (`MockSource` and `MockBatch` do), and otherwise fall back to the
class name — a `CachedDataset` simply prints as `CachedDataset`.

**`[WARN]` lines.** Diagnostics are printed, never raised. A profiler that
notices something suspicious — a skewed size distribution, a missing pointer, a
`NaN` label — appends a `[WARN]` line to the report and keeps going. Exceptions
are reserved for inputs the profiler genuinely cannot measure.

## Synthetic Inputs

When you want a measurement before a dataset exists, `molix.profiler` ships
three stand-ins.

`MockSource` implements the `DataSource` protocol — `__len__` plus
`__getitem__` returning a flat sample `dict` with `Z` `(N,)` (atomic numbers)
and `pos` `(N, 3)` (Cartesian positions). Atom counts are either fixed or drawn
from an inclusive `(lo, hi)` range.

`MockBatch` is a callable factory returning a post-collate nested `TensorDict`
with the `atoms` / `edges` / `graphs` namespaces the models expect. Atom, edge
and graph counts are fixed or drawn from ranges on every call, which is how you
stress-test a module against variable input shapes.

```python
from molix.profiler import MockBatch, MockSource

source = MockSource(n_samples=500, n_atoms=(5, 20), seed=0)
sample = source[0]                     # {"Z": (N,), "pos": (N, 3)}

factory = MockBatch(n_atoms=(32, 96), n_edges=(100, 600), n_graphs=4, seed=0)
batch = factory()                      # nested TensorDict
```

Both are **seed-reproducible**: with a seed set, two instances constructed in
the same process produce the same shape sequence, because the size draws come
from the instance's own `random.Random` rather than the process-global `random`
module. Leave `MockBatch(seed=None)` unseeded if you deliberately want a
different shape sequence per instance.

`MockModel` is the third stand-in: an `nn.Module` that honours the encoder
contract (reads `atoms.pos`, writes `atoms.node_features` `(N, 1, n_features)`)
with one scalar multiply and one scalar `Parameter`. Its compute is negligible,
which is exactly what `TrainerProfiler` needs. Its companion loss,
`mock_node_feature_loss`, sums those features so `backward()` reaches the
parameter.

## TaskProfiler

A *task* is one step of the preprocessing pipeline (`molix.data.task`):
`SampleTask` transforms one sample dict, `DatasetTask` first fits global
parameters over the whole dataset and then transforms each sample, and
`BatchTask` transforms an already-collated batch. `TaskProfiler` times
`task.execute(...)` on its own, with no DataLoader, workers or collation in the
picture.

```python
from molix.data.tasks import NeighborList
from molix.profiler import MockSource, TaskProfiler

source = MockSource(n_samples=500, n_atoms=(5, 20), seed=0)

result = TaskProfiler(NeighborList(cutoff=5.0)).run(source, n_samples=100, n_warmup=10)
result.print_report()

print(result.timing.p95_ms)
```

The task is the constructor argument; the sample provider is the `run()`
argument. Any object with `__len__` and `__getitem__` works — a `MockSource`, a
real `DataSource`, or a dataset. Indices wrap modulo the source length, so
`n_samples` may exceed the number of distinct samples available.

`TaskResult` carries `task_name`, `task_id` (the task's cache-key identity, e.g.
`nlist:cut=5.0:max=512:pbc=False:sym=True`), the `timing` statistics in
milliseconds, `n_samples`, and `data_description`. The report is a single
"Execute time" row with mean, std, p50, p95, min and max.

Two things to know before pointing it at a `DatasetTask`. First, the profiler
calls `fit()` on the *entire* source before timing begins, mirroring what
`PipelineSpec.run` does — that materialises every sample in memory, so use a
subset for large sources. Second, the fit itself is not timed; only the
per-sample `execute` is. For a `BatchTask`, whatever `source[i]` returns is fed
straight to `execute`, so the "source" must already yield collated batches.

## ModuleProfiler

`ModuleProfiler` times any `torch.nn.Module` in isolation: no `Trainer`, no
`DataModule`, no hooks.

```python
import torch
from molix.profiler import MockBatch, ModuleProfiler

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
profiler = ModuleProfiler(model, loss_fn=my_loss, device="cuda:0", optimizer=optimizer)

factory = MockBatch(n_atoms=(32, 96), n_edges=(100, 600), n_graphs=4, device="cuda:0")
result = profiler.run(factory, n_steps=200, n_warmup=10)
result.print_report()
```

`run()` accepts a `MockBatch` factory (called once per step), a list of
pre-built batches (cycled), or a `DataLoader` or other iterable (cycled).
Batches are moved to `self.device` before each step, so generating them on the
target device — as above — avoids paying for a host-to-device copy inside the
measurement. Passing `loss_fn` adds the backward pass; passing `optimizer` as
well (one that already wraps `module.parameters()`) adds the optimizer step,
making the measured step identical in structure to what `Trainer` runs. The
optimizer step is only taken when a `loss_fn` is present.

Three separate measurements go into one `ModuleResult`.

**Component breakdown** — forward, backward and optimizer timed individually.
On CUDA each component is bracketed by a pair of `torch.cuda.Event` markers and
resolved by synchronizing, which serialises CPU and GPU. Read these as
*relative* costs and upper bounds, not as throughput.

**Full-step block** — `n_steps` steps run back to back with no synchronization
in between, followed by a single one at the end. This is `wall_ms_per_step`, the
number that reflects real throughput. It is measured on CUDA only; on CPU the
field stays `None` and the report omits the block.

**Op breakdown** — a short `torch.profiler` window, also CUDA only. Summing
per-kernel *device self-time* (time inside the kernel itself, excluding anything
it called) gives `gpu_active_ms_per_step`: on a single stream kernels run
serially, so the sum is the true GPU-busy time. A single event pair around the
whole block would instead report the GPU-timeline span, which already contains
the idle gaps and therefore just reproduces the wall time.

From those two numbers comes the most useful line in the report:

$$
\mathrm{launch\ bound\ \%} =
\frac{t_\mathrm{wall} - t_\mathrm{gpu\ active}}{t_\mathrm{wall}} \times 100
$$

where $t_\mathrm{wall}$ is `wall_ms_per_step` and $t_\mathrm{gpu\ active}$ is
`gpu_active_ms_per_step`, both in milliseconds per step. This is the percentage
of each step during which the GPU sits idle because the CPU has not yet
enqueued ("launched") the next kernel. The report converts it
into a one-line verdict: at or above 60 % it prints `LAUNCH-BOUND — too many
tiny kernels / batch too small`, at or above 30 % `partially launch-bound`,
and below that `compute-bound`. A launch-bound model does not get faster from a
faster GPU; it gets faster from bigger batches, fused kernels, CUDA graphs or
`torch.compile`.

The rest of `ModuleResult`: `peak_memory_mb` (peak CUDA allocation during the
forward pass, `0` on CPU), `throughput_atoms_per_sec` and
`throughput_graphs_per_sec` (computed from the **forward** mean only, using the
atom and graph counts read out of each batch), `n_params`,
`op_calls_per_step`, and the pre-rendered `op_table` of the top operators by
self-CUDA time. Pass `submodules=True` to add a second, forward-only hooked
pass that attributes time to the module's top-level named children —
`ModuleList` / `Sequential` / `ModuleDict` containers are expanded into their
entries, since the container itself is never called. The gap between the sum of
the children and the reported forward mean is work done inline in `forward`
rather than inside a named child.

For sub-modules that do not take a batch `TensorDict` — a radial basis function
that takes a distance tensor, say — use `run_fn` with explicit callables:

```python
distances = torch.rand(512, device="cuda:0")

result = ModuleProfiler(rbf, device="cuda:0").run_fn(
    forward_fn=lambda: rbf(distances),
    backward_fn=lambda out: out.sum(),
    n_steps=200,
    label="edge_dist E=512",
)
```

`run_fn` reports the forward/backward breakdown and memory only; throughput is
reported as zero, because raw tensors carry no atom or graph counts.

## DataLoaderProfiler

`DataLoaderProfiler` measures **stall time**: how long the consumer of a
`DataLoader` blocks waiting for the next batch to arrive. It uses the
inter-batch gap technique — the wall-clock elapsed from the moment the loop
finishes with batch *i* to the moment batch *i+1* is handed over. That interval
covers worker scheduling, sample reads, collation and pinning, which is exactly
the time a training step cannot overlap with compute.

```python
from molix.profiler import DataLoaderProfiler, MockSource

profiler = DataLoaderProfiler(batch_size=32, num_workers=4, pin_memory=True)
result = profiler.run(MockSource(n_samples=2000, n_atoms=(5, 20), seed=0), n_batches=100)
result.print_report()
```

Configuration is the DataLoader configuration you want to test: `batch_size`,
`num_workers` (worker subprocesses; `0` means load in the main process),
`pin_memory` (allocate host buffers in page-locked memory so the copy to the GPU
can run asynchronously), `persistent_workers` (keep workers alive across epochs;
silently forced off when `num_workers == 0`), `target_schema` (how labels are
routed during collation, defaulting to `DEFAULT_TARGET_SCHEMA`), and an optional
`PipelineSpec` whose `batch_nodes` are applied inside `collate_fn`.

`run()` resolves its argument in three ways. An existing `DataLoader` is used
as-is and your constructor settings only affect what the report *says*. A
`torch.utils.data.Dataset` is wrapped in a new DataLoader built from those
settings. Anything else is treated as a `DataSource`: every sample is
materialised, written to a `PackedCache` file in a fresh temporary directory,
and reopened as a `CachedDataset`, so the measurement exercises the same
cache-file path a real workflow uses. Be aware that this last route writes to
disk and holds the whole source in memory while packing.

With `num_workers > 0` the loader is built with the `spawn` start method and a
top-level picklable collate object, so the profiler works under `spawn` and
`forkserver` where a closure would fail to pickle.

`DataLoaderResult` reports `load_time` (the stall statistics),
`throughput_graphs_per_sec` and `throughput_atoms_per_sec`, and the per-batch
size distributions `batch_graph_stats` and `batch_atom_stats` — the latter is
the one to watch when batches are size-heterogeneous, because a large `std`
means padded batches waste compute. The report ends with a `[WARN]` when
`p95 > 3 × mean`, which points at worker stalls or collation spikes and suggests
more workers or `persistent_workers=True`.

If the loader yields no batch beyond the warmup window, `run()` raises
`RuntimeError` rather than reporting statistics over an empty sample.

## TrainerProfiler

The other four profilers measure work you asked for. `TrainerProfiler` measures
the work you did not: the per-step cost of the `Trainer` machinery itself —
`Step` protocol dispatch, hook calls, `batch_to` device transfer, `TrainState`
writes, and the optimizer/scheduler/eval-cadence bookkeeping.

Isolating that cost needs two tricks. First, the model must be nearly free, or
its FLOPs drown everything else out; the default is `MockModel`. Second, a bare
training loop still costs something — forward, backward, `optimizer.step()` —
and that cost is not the Trainer's fault. So the profiler measures a raw loop
with no Trainer at all and subtracts it.

```python
from molix.profiler import TrainerProfiler

result = TrainerProfiler(device="cpu").run(n_steps=2000, n_warmup=50, top=15)
result.print_report()

print(result.overhead_ms_per_step)
```

The constructor takes `model` (default `MockModel`), `loss_fn` (default
`mock_node_feature_loss`), `device`, and `hooks`. Hooks are the interesting
knob: run once with none to price the bare loop, then again with the hook list
you actually train with to price hook dispatch.

```python
from molix.hooks import StepSpeedHook

TrainerProfiler(hooks=[StepSpeedHook()], device="cpu").run(n_steps=2000).print_report()
```

`run(n_steps, n_warmup, batch, top)` drives a real `Trainer` for one epoch over
a fixed batch replayed `n_steps` times. `batch` defaults to a `MockBatch` with
32 atoms, 128 edges and 4 graphs on the configured device; pass your own
`TensorDict` to see how the overhead behaves at your batch size. The internal
optimizer is SGD with `lr=0.0`, so nothing is learned — weights stay put and
only machinery is measured. `top` caps the number of hotspot rows kept.

Four passes are made after the warmup: the raw loop timed, the raw loop under
`cProfile`, the Trainer loop timed, and the Trainer loop under `cProfile`.
Timing and profiling are deliberately separate runs, because `cProfile` adds
per-call overhead that would distort the wall-clock numbers.

The report opens with three lines:

- `Raw loop (baseline)` — `baseline_ms_per_step`, the irreducible per-step work:
  `zero_grad` → forward → loss → `backward` → `optimizer.step()`.
- `Trainer loop` — `wall_ms_per_step` for the same work driven through
  `Trainer`, with `steps_per_sec = 1000 / wall_ms_per_step` alongside.
- `Trainer overhead` — `overhead_ms_per_step`, the difference (clamped at zero),
  and what percentage of the loop it represents.

Below them is the hotspot table, headed *Trainer self-time minus raw-loop
baseline (per step)*. A function's **self-time** is the time spent in its own
body, excluding the functions it calls, as attributed by `cProfile`. For each
`(file, line, function)` the profiler subtracts the raw-loop self-time from the
Trainer-loop self-time and keeps the positive remainders, sorted descending:

| column | meaning |
| --- | --- |
| `func` | `file:line(function)`, truncated to 48 characters |
| `added_us` | microseconds per step this function adds *over* the baseline |
| `gross_us` | microseconds per step it costs in the Trainer run, unsubtracted |
| `calls` | calls per step |

The subtraction is what makes the table readable. Autograd, the model forward
and the optimizer appear in both runs at roughly equal cost and cancel out,
leaving framework machinery — the loop body, the `Step` wrapper, device
transfer, hook dispatch, state writes — at the top.

The same numbers are available on `TrainerResult` as `n_steps`, `device`,
`n_hooks`, `wall_ms_per_step`, `steps_per_sec`, `baseline_ms_per_step`,
`overhead_ms_per_step`, `model_name` and the `hotspots` list of dicts.

## DatasetProfiler

`DatasetProfiler` characterises the data itself rather than the loop around it.
Point it at anything that yields **flat sample dicts** — the raw-sample tier of
the two-tier data contract, so `sample["Z"]`, `sample["pos"]`,
`sample["edge_index"]`, `sample["targets"]["U0"]` — and it answers four
questions at once: how big are the records, what does one access cost, how are
the fields laid out, and what do the labels look like.

```python
import torch

from molix.data.cache import PackedCache
from molix.data.dataset import CachedDataset
from molix.profiler import DatasetProfiler


def ring(i: int) -> dict:
    """One literal sample: a ring molecule of 2–5 carbon atoms."""
    n = 2 + i % 4
    src = torch.arange(n)
    dst = (src + 1) % n
    return {
        "Z": torch.full((n,), 6),
        "pos": torch.arange(3 * n, dtype=torch.float32).reshape(n, 3),
        "edge_index": torch.cat(
            [torch.stack([src, dst], dim=1), torch.stack([dst, src], dim=1)]
        ),
        "edge_dist": torch.full((2 * n,), 1.5),
        "targets": {"U0": torch.tensor([float(i)])},
    }


sink = "/tmp/ring-cache.pt"
PackedCache(sink).save([ring(i) for i in range(100)], overwrite=True)
dataset = CachedDataset(sink)

result = DatasetProfiler(n_samples=20).run(dataset)
result.print_report()

assert result.counts_exact          # size stats cover all 100 records
assert result.avg_num_neighbors == 2.0  # 700 edges / 350 atoms
```

You can also hand `run()` a `MockSource(n_samples=100, seed=0)` or a plain
`list[dict]` directly, without packing anything. Those carry no packed pointers,
so the size statistics come from the sampled records instead — see below — and
`MockSource` samples have no edges at all, so the edges row is simply absent
from the size section.

Configuration is `n_samples` (how many records the sampled path may read),
`stride` (the step between inspected indices, so `stride > 1` spreads the sample
across an ordered dataset instead of reading a prefix), and `n_warmup`. Indices
are `range(0, len(data), stride)` truncated to `n_samples`.

### The exact fast path

Cache-backed datasets store their samples packed: every key is concatenated
across all records into one big tensor, with `atom_ptr` and `edge_ptr` cumulative
sum ("cumsum") vectors marking where each record's slice begins. Per-record
counts are then just `ptr[i+1] - ptr[i]`, one vectorised subtraction over the
whole file with no sample unpacked.

So when the object exposes `atom_counts`, `edge_counts`, `avg_num_neighbors`,
`max_atoms` and `max_edges` — `CachedDataset`, `MmapDataset`, and `SubsetDataset`
which remaps them to its own indices — the profiler reads those properties
directly. Size statistics then describe **every** record in the dataset, not the
sampled subset, and the result reports `counts_exact=True`. The report prints
the flag next to the neighbour count so you always know which regime produced
the numbers.

`avg_num_neighbors` deserves a definition, since models consume it. It is

$$
\langle |N(i)| \rangle = \frac{E_\mathrm{total}}{N_\mathrm{total}}
$$

where $E_\mathrm{total}$ is the number of edges and $N_\mathrm{total}$ the
number of atoms, both summed over the whole dataset — or, on a
`SubsetDataset`, over that split only, since a training subset must not peek at
validation data. The ratio is dimensionless. With
`NeighborList(symmetry=True)`, the default, the neighbour graph is fully
bidirectional, so this ratio is the mean number of neighbours per atom, which is
the normalisation constant MACE and Allegro divide their aggregated messages by.

Without a packed cache — a plain `list[dict]`, a `MockSource` — none of those
properties exist. The profiler then derives size statistics from the records it
sampled, sets `counts_exact=False`, and says so in a `[WARN]` line. The numbers
are still useful; they are just estimates from a subset.

One legitimate half-way case: a cache built by a pipeline that never ran
`NeighborList` has atoms but no edges, so `edge_counts` raises `ValueError`. The
profiler catches it, sets `edge_stats=None`, drops the edge row from the report,
and appends a `[WARN]` carrying the underlying message. Missing edges are a fact
about your pipeline, not a profiler failure.

### The sampled path

Some things cannot be read off a pointer vector and genuinely require touching
records one at a time:

- `cold_access_ms` — the very first `data[i]`, including memory-map page-in from
  disk. On an `MmapDataset` this is usually much larger than the steady state.
- `access_ms` — steady-state `__getitem__` latency after `n_warmup` discarded
  accesses, as a full `TimingStat`.
- `sample_bytes` — the leaf-tensor footprint of one record: the sum of
  `numel() * element_size()` over its tensor leaves. Non-tensor leaves
  contribute nothing.
- `est_total_mb` — `sample_bytes.mean × n_total / 1e6`, i.e. what a full in-RAM
  materialisation would cost in megabytes (10⁶ bytes). It is an extrapolation
  from the sampled mean, so it is only as good as the sample on a skewed dataset.
- `targets` — for every leaf under `targets.` that is a one-element tensor or a
  plain number, a `TargetStat` with mean, std, p50, p95 over the **finite**
  values, plus `min`, `max` and `n_nonfinite`. Non-finite entries are counted and
  then excluded from the moments, so a single `NaN` row cannot poison the column.

### Field layout

The `Fields` section lists one `FieldSpec` per key: the dotted key path, the
packing `axis`, the `dtype`, and the trailing shape after the packing axis
(`(3,)` for `pos` `(N, 3)`, `()` for `Z` `(N,)`).

| axis | meaning |
| --- | --- |
| `atom` | concatenated along dim 0 — one row per atom |
| `edge` | concatenated along dim 0 — one row per edge |
| `graph` | stacked on a new leading dim — one entry per record; `extra_shape` is the full per-sample shape |
| `scalar` | a non-tensor Python value; `dtype` is the type name and `extra_shape` is `()` |

When the dataset exposes `packed_view()`, this comes straight from the cache's
`payload["schema"]` — the layout inferred across *all* records at packing time —
and `fields_exact=True`. Otherwise the layout is inferred from the sampled
records (leading dimension matches the atom count, else the edge count, else
per-graph), `fields_exact=False`, and a `[WARN]` notes that axes and trailing
shapes could differ on unsampled records.

If the dataset has a `stats()` method, its fitted `DatasetTask` state is
collected into `task_states` and the names are listed in the report — that is
how you see at a glance that this cache was baked with, say, `AtomicDress`.

### Diagnostics and errors

The dividing line is strict. **Diagnostics never raise.** Non-finite labels,
inexact counts, inexact fields, a missing edge pointer, and an atom-count skew of
`p95 / p50 > 3` (padded batches will waste compute; consider a token-budget
sampler) are all `[WARN]` lines under the report's closing rule.

`ValueError` is reserved for inputs that cannot be profiled at all, and each
message says what to pass instead: `n_samples <= 0` or `stride <= 0` at
construction; an empty dataset, or an object with no `__len__` / `__getitem__`,
at `run()`.

No unit conversion is performed and no units are guessed. Positions, distances
and every target are printed exactly as the dataset stores them — if your
positions are in Ångström and your energies in eV, that is what you are reading.

## Where to Start

Working outwards from the data usually converges fastest. Run `DatasetProfiler`
first to learn what the records look like and whether sizes are skewed; then
`TaskProfiler` on any preprocessing step that looks expensive; then
`DataLoaderProfiler` to see whether batch production keeps up; then
`ModuleProfiler` to find out whether the model is compute-bound or launch-bound;
and finally `TrainerProfiler` if per-step time still exceeds what the model and
the loader together explain.
