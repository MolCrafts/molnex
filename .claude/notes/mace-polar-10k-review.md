# MACE-POLAR: 10k PEO execution review and multi-GPU design

Reviewed: 2026-09-05; precision/loading cleanup: 2026-09-06.
Status: single-GPU validation, **not production MD/training sign-off**.
The distributed types below are a design, not implemented APIs.

## Decision

Keep cuEquivariance as the only equivariant kernel backend. Receiver blocking
and recomputation solve a scheduling/lifetime problem; neither needs a new TP,
CG implementation, autograd Function, nor an e3nn dependency. There is no
current evidence requiring OEQ or vendored OEQ for the approximately 10k target.

The available near-10k PEO fixture is **8,850**, not 10,000 atoms. The 35,400
fixture is a memory scale probe, not an equilibrated MD benchmark. Tests ran
on GH200 reporting 95.0 GiB usable memory despite its “120GB” product name.
Do not claim measured performance on a different GPU or exactly 10,000 atoms.

## Findings, in priority order

1. **fp32 is not a sufficient precision specification.**
   `config.set_precision("fp32")` selects matmul `high`, allowing reduced
   internal precision. Fixed-coordinate E/F/virial repeats on 8,850 atoms gave
   max force differences 0.03125 eV/Å, virial differences around 15 eV, and a
   materially different Fmax (31.0156 versus strict-fp32 27.5804 eV/Å).
   With matmul `highest`, repeated force differences fell to 4.2e-5 eV/Å;
   forces-only versus strained-virial forces differed by 2.7e-5 eV/Å.
   This isolates the process-level matmul setting, **not a particular GEMM**.
   The validation runner now defaults to `highest`; the library-wide preset
   is unchanged. Pin the execution precision in every calibration record.
   PyTorch documents these [matmul precision semantics](https://docs.pytorch.org/docs/2.12/generated/torch.set_float32_matmul_precision.html).

2. **Large atomic baselines were summed in fp32.**
   Even strict-fp32 energy fluctuated by several eV between identical calls;
   versus fp64 its total differed by 36.0 eV, although force RMSE was 6.07e-5
   eV/Å and max difference 6.66e-4 eV/Å. Total energy is about -5.224e6 eV.
   `MacePolar.energy_core` now accumulates E0, local interaction energy and
   local electron energy in float64, and returns float64 energy with an fp32
   backbone. This is an intentional output-precision change; parameter keys,
   force dtype, model equations and checkpoint layouts do not change.
   A test checks that a large constant E0 neither rounds away interaction
   energy nor changes forces. It does not restore precision already lost
   when casting the frozen atomic-reference table to fp32.

3. **The observed large-system OOM was not the cuEq product kernel.**
   Job 2051229 failed in `molrep/interaction/polar/uvu.py`: a literal
   `(N,512,512)` channel-pair intermediate, 34.57 GiB for 35,400 fp32 atoms,
   followed by another scaling allocation. The reassociation
   `sum_d a[u,d] * sum_v W[u,v] b[v,d]` uses ordinary `F.linear` and avoids
   that tensor. It retains the existing sparse uvu math and weights; it does
   not implement a new kernel. Literal-contraction/gradient parity,
   mixed-second-derivative gradgradcheck, and a fullgraph CPU test cover it.
   Distinguish this from cuEq `SymmetricContraction` and its own workspace.

4. **Measure the loaded descriptor, not an assumed 86-path model.**
   This Polar-1-M dump has 15,238,506 parameters and two conv weight widths
   **2048 / 1280**, following the actual per-layer reduced edge channels.
   At E=1,097,832, the respective fp32 tables are about 8.38 / 5.24 GiB,
   excluding the rest of the tape. These are substantial but not the earlier
   assumed E×86×512 table. The runner prints both widths for every model.

5. **Official full conv and receiver blocks are a real speed/memory tradeoff.**
   `interaction/mace/conv.py` checkpoints radial MLP + indexed official TP
   per receiver interval, including source/target embeddings and cutoff.
   For max block size 4096, N=8850 yields three spans of 2950 atoms and
   454510 edge slots per span. cuEq owns gather/TP/scatter/backward. No
   Python per-CG-path scatter loop or local CUDA TP is needed.
   The [official indexed-convolution example](https://nvidia.github.io/cuEquivariance/tutorials/pytorch/MACE.html)
   documents the primitive this uses. Full conv can be faster when it fits;
   blocking is not a universal speed optimization.

6. **Neighbor rebuilding remains quadratic in the default pair backend.**
   `molix/neighbor.py` forces candidate capacity N(N-1)/2. At 35,400
   atoms the measured rebuild peak is 28.34 GiB, even after convolution and
   field memory improvements. Existing `NeighborList(bin=0.0)` is a
   pure-PyTorch cell-list alternative; no new neighbor op is proposed.
   On 8850 atoms, default rebuilding took only 2–3 ms versus hundreds of ms
   for E/F/virial, so it is not this size's first speed bottleneck.

7. **The current public training/runtime surface is not seamless multi-GPU.**
   Periodic GTO field and Hartree explicitly reject `num_graphs > 1`.
   Therefore “packed periodic training” cannot mean concatenate arbitrary
   cells into the current PBC path. Trainer has no complete distributed
   strategy lifecycle. CPU first-order checkpoint tests are not proof of
   cuEq GPU force-loss double backward or a distributed optimizer update.

8. **Receiver layout currently requires a manual producer/consumer agreement.**
   Model and NeighborList must receive the same maximum block size. Divisible
   edge count alone cannot prove target ownership. The next execution seam
   must bind and validate the layout outside the compiled core; arbitrary
   collated COO must never silently enter the blocked kernel. Existing tests
   cover interleaved dead slots, rebuild overflow and prefix-aware MD trimming.

## Reproducible measurements

All entries below use fixed coordinates, cutoff 6 Å plus skin 1 Å, two warmup
calls, and 3–5 timed E/F/virial calls under `no_grad`. CUDA synchronization
surrounds each sample. Model time excludes neighbor rebuild; peak is PyTorch
allocated memory, not the allocator's reserved pool. Raw records are
`.scratch/polar_npt/npt-JOB.out` (`BENCHMARK` JSON), with E/F/virial tensors
in `benchmark-JOB.pt` for newer jobs. Torch 2.12.0+cu130, cuEq 0.10.0.

| Job | N | Execution | Median E/F/virial | Model peak | Use of result |
| --- | ---: | --- | ---: | ---: | --- |
| 2057105 | 8850 | full cuEq, strict fp32 | 0.410 s | 44.55 GiB | before fp64 energy pooling |
| 2057021 | 8850 | blocked cuEq, strict fp32 | 0.637 s | 16.49 GiB | before fp64 energy pooling |
| 2057008 | 8850 | blocked cuEq, fp64 | 1.029 s | 32.87 GiB | numerical reference |
| 2057185 | 8850 | blocked cuEq, strict fp32 + fp64 energy pooling | 0.636 s | 16.49 GiB | stable-energy implementation |
| 2057059 | 8850 | blocked fullgraph Inductor, strict fp32 | 0.415 s | 28.45 GiB | before fp64 energy pooling |
| 2057314 | 8850 | full cuEq + fullgraph, strict fp32 + fp64 pooling | 0.258 s | 36.49 GiB | stable-energy implementation |
| 2057313 | 8850 | blocked cuEq + fullgraph, strict fp32 + fp64 pooling | 0.416 s | 28.45 GiB | stable-energy implementation |
| 2056954 | 35400 | blocked cuEq, fp32 matmul high | 2.273 s | 44.87 GiB | memory completion only; precision rejected |

The fp64 reference had max repeat differences E/F/W = 0 / 4.3e-14 / 1.7e-11,
and force-only versus virial-force difference 7.1e-14 eV/Å. Strict fp32 versus
fp64 figures above include coordinate and weight casting, not only pooling.
After stable energy pooling (2057185 versus 2057008): total energy error
0.003770 eV (4.26e-7 eV/atom), force RMSE 6.07e-5 eV/Å, max force error
6.58e-4 eV/Å, max virial error 0.01221 eV. Identical-coordinate energy
repeat spread is now 0.00262 eV instead of several eV. The model peak and
median time did not materially change. No fp32 atomic-table precision was
reconstructed from the fp64 reference.
Job 2057314's fullgraph versus eager max E/F/W deltas were
0.000483 eV / 4.77e-5 eV/Å / 0.00879 eV. Its 36.49 GiB is the warmed model
peak; the allocator reserved 69.53 GiB after eager preflight/compilation.
This is **not** a validation that cold startup fits a 40 GiB GPU.
Full compiled versus fp64 (2057314 versus 2057008): total energy difference
0.000369 eV, force RMSE 6.08e-5 eV/Å, max force difference 6.70e-4 eV/Å,
max virial difference 0.00804 eV. Blocked compilation (2057313) passed
fullgraph execution and had eager-comparison E/F/W deltas
0.000415 eV / 5.34e-5 eV/Å / 0.00806 eV. Compilation increased the blocked
path's allocated peak from 16.49 to 28.45 GiB while improving its time;
do not assume that the compiler preserves eager checkpoint memory savings.
Historical 1-step NPT numbers (1.78 s versus a Python-loop 1.83 s) include
startup and evolving geometry; they do **not** establish a steady speedup.
Historical compiled 4995-atom results likewise do not prove the new 8850
blocked/fullgraph configuration. Cold cuEq JIT can take minutes.

### Throughput budget, not just “fits”

At dt=0.5 fs, 0.258 s/E/F/virial corresponds to **0.167 ns/day before MD
overhead**. Ten ns would therefore take at least about 60 days at this rate.
Whether this is sufficient is a user throughput requirement, not a conclusion
from the absence of OOM. A 1 ns/day target requires about 43 ms/step: at least
six-way perfect scaling relative to this single-GPU kernel time, and more
realistically additional margin for communication and serial MD. This is a
lower-bound calculation, **not a prediction that six or eight GPUs achieve
it**. Do not enlarge dt or alter the model to claim execution speedup.

For the latest runner, explicitly supply the large box; the launcher default
is a **92-atom** smoke fixture. For example, set `POLAR_SYSTEM` to
`.../assets/boxes/dp25_n50.pt`, `POLAR_BENCHMARK_REPEATS=5`, and
`CUEQ_TARGET_BLOCK_SIZE=4096` (or 0 for full conv). `POLAR_PRECISION=fp64`
selects the oracle; `POLAR_COMPILE_CORE=1` requests fullgraph Inductor with
an eager comparison; `POLAR_NEIGHBOR_BIN=0` selects the existing cell list.
No production training or long trajectory was launched for this review.

## Single-GPU rollout contract

1. Preserve the official checkpoint, irreps layout, cutoffs, field recursion,
   k-grid and self-interaction conventions. No model-width reduction or PME
   substitution disguised as an execution optimization.
2. Use strict matmul and stable energy accumulation first. Calibrate full
   versus blocked cuEq on the actual GPU, reserving at least 20% HBM headroom
   in the target workload. Never choose training capacity from no-grad MD.
3. Validate eager versus fullgraph Inductor on the exact fixed receiver/edge
   layout; AOTAutograd supplies first-order backward. Keep the named external
   force-derivation seam. Do not label `donated_buffer=False` alone as proof
   that compiled force training supports higher derivatives.
4. Bind element tables, receiver layout and reciprocal-grid capacity eagerly.
   Neighbor/box overflow must raise before continuing; dead edges retain safe
   indices and zero cutoff, never merely trim an interleaved global prefix.
5. Accept force-training only after energy/force/virial loss, parameter grads,
   and one optimizer update match a small fp64 reference. Follow the existing
   production-training checklist; an inference benchmark does not sign it off.
6. Before long MD: use an equilibrated near-10k PEO box, compare physical
   observables and NVE drift across dt / precision, then bounded NPT with
   volume, temperature, pressure, charges/spins and neighbor completeness.
   The runner now labels MD heartbeat energies/forces in eV/eVÅ and captures
   initial volume before MD mutates the shared cell. Hot packed starting
   configurations are not evidence against displacement-gated Verlet lists.

## Multi-GPU framework: preserve one mathematical model

### Two independent parallel axes

| Axis | Owned data | Intended benefit | What it cannot do |
| --- | --- | --- | --- |
| `data` | independent configurations / trajectories | training or ensemble throughput | reduce memory of one large periodic graph |
| `atom` | disjoint receiver atoms of one periodic graph | single-system activation memory and latency | replace global Polar electrostatics by local graphs |

Use a two-dimensional process-group/DeviceMesh topology conceptually
`(data_replicas, atom_ranks)`. `(1,1)` is the current local execution.
Separate-trajectory inference uses independent workers, not DDP gradient
communication. For training, DDP synchronizes replicated parameters along
the data axis; current PBC code allows one cell per data-rank microbatch,
with gradient accumulation until proper multi-cell batching is implemented.
[DDP does not shard the input](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html).

FSDP is not the primary remedy: this model has only about 58 MiB fp32
parameters; parameters + gradients + two Adam moment tensors are about
233 MiB, versus tens of GiB of activations. Channel/tensor parallelism
would couple the framework to cuEq irreps and weight layouts; defer it.

### Narrow ownership boundaries (proposed, not yet library types)

- `molix` owns `AtomPartition`: global/local atom maps, owned/halo indices,
  static receiver edge capacity and communication counts only. It validates
  producer/consumer agreement and rebuilds plans eagerly. It does not own
  model parameters, physical equations, optimizers or observables.
- A `molix` feature-exchange component owns process groups and transport.
  Local execution is an identity; distributed execution exchanges only
  boundary features. Start with a correctness-only all-gather baseline,
  then spatial ownership and halo exchange. Index-based receiver blocks are
  not necessarily spatial domains and cannot be assumed to minimize traffic.
- `molrep` interactions continue to consume local tensors and invoke cuEq.
  They never read world size, launch MPI jobs or branch on model backend.
  Products, field update MLPs and readouts run on owned atoms. Halo atoms
  contribute source features, never duplicate local energies/charge sums.
- `molpot` exposes the two actual global algebraic seams: per-graph sums
  used by Fukui, and structure-factor construction followed by local-target
  field evaluation. A local/distributed reduction implementation is injected
  at construction, not inferred from hidden global process state.
- Trainer and MD own execution lifecycle and result delivery. Training calls
  the wrapped module's `forward`; bypassing it through `.module.energy_core`
  would bypass DDP's coordination. Loss output schemas and physical model
  checkpoints stay identical. Partition maps, groups and compile caches are
  runtime state, never new persistent model-weight keys.

### Global operations that must remain exact

For each short-range layer, exchange needed source features, calculate cuEq
messages for owned receivers, then do the owned-node product/readout. Reverse
communication must accumulate every remote use back to the source owner.
Ghost coordinates also need a differentiable exchange when building live
edge geometry; treating received coordinates as detached constants loses
cross-domain forces. Atom IDs and image translations are nondifferentiable
topology; positions, features and the shared strained cell are not.

For every Fukui correction, globally SUM both the predicted spin monopoles
and raw Fukui weights per graph; use these global totals to enforce Q and
M=2S+1. Local normalization would describe a different physical model.

For periodic GTO, each rank builds its partial
`S_r(k)=sum_{i owned} (q_i + i k·mu_i) exp(i k·r_i)`; globally SUM S before
evaluating fields at owned targets. Preserve the same integer half-box,
cutoff weights, cell, widths and on-site subtraction. Hartree is formed
from `|sum_r S_r|²`, **not** `sum_r |S_r|²`. Count its global energy once.
This retains the existing reciprocal calculation, not a new PME operator.

All ranks share the live cell, strain convention, thermostat/barostat state
and a consistent neighbor rebuild decision. Ownership migration and all
rank counts change outside graphs. Empty ranks, unequal halos, periodic
crossings and capacity overflow must be explicit supported cases or fail
closed; no rank may silently skip a collective.

### Derivatives and loss scaling are acceptance criteria

Use upstream differentiable communication primitives with explicit tensor
placement semantics. Ordinary in-place `dist.all_reduce` on a gradient-bearing
tensor is not a force-training solution. Do not pass DTensor blindly into
opaque cuEq kernels; keep local Tensor boundaries and explicit redistribution.
The [upstream collective implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py)
also shows why API/version/compile support needs testing, not assuming.

Define local partial energies versus replicated global energy explicitly.
A replicated global loss must not be seeded once per atom-rank without
compensating multiplicity. Forces are derivatives of one physical energy;
parameter gradients from owned domains are summed once across the atom
axis, then data-replica averaging follows the global sample/atom loss
denominator. Test this derivation rather than inserting world-size divisions
until numbers appear plausible. Include uneven atom counts and graph losses
as well as force losses. DDP's `autograd.grad` caveat applies to parameter
gradient accumulation; an inner coordinate gradient followed by a parameter
loss backward still requires an end-to-end test of this exact model.

Required 1/2/4-rank tests: E/F/W, total Q/spin, invariances, first and mixed
second derivatives through every collective, energy+force+virial losses,
one optimizer update, accumulation/no-sync behavior, checkpoint/resume and
rank-loss failure. Local fullgraph compilation is required; any retained
communication break must be named and bounded. Do not promise a seamless
training backend after inference-only all-gather parity.

Per-rank ring all-gather reception is approximately `(P-1)/P * N * D * bytes`
per feature exchange, before backward traffic; spatial halos replace N with
the actual boundary counts. Reciprocal communication is proportional to the
k-grid, not the edge-weight table. These are traffic models, not measured
speedups. The separable GTO representation reduces workspace but is not a
PME-like linear-complexity solver: its Miller grid grows with box size at a
fixed physical cutoff. Include this in larger-than-10k scaling estimates.

### When to implement it

First finish the single-GPU precision/compiled/MD gates. Multi-GPU data
parallelism is the next training/ensemble-throughput feature. Implement
atom-domain parallelism only if a measured target cannot meet single-GPU
HBM headroom or throughput requirements. Profile GPU interconnect/collective
latency before promising speedup: at 10k, communication can cost more than
the saved compute. Keep OEQ and vendoring outside this path unless a cuEq
capability test identifies a concrete unsupported operation.

## Verification at this revision

The initial optimization pass had **326 passed, 6 skipped, 15 deselected**. Focused CPU
fullgraph tests for blocked ConvTP, checkpointed density interaction and the
reassociated uvu contraction: **3 passed**, plus **1 passed** for the periodic
Polar energy-core one-graph test after fp64 pooling. The new uvu mixed-second-
derivative test and the fp32 large-baseline invariance test pass. These do
not replace GPU force-training or multi-rank tests. Ruff on the newly touched
Python surfaces, `git diff --check` and launcher `bash -n` pass.

The runner no longer monkeypatches `importlib.metadata.version`: both
architecture-specific environments have molpy/molrs 0.14 metadata. The
existing immutable aarch64 import shim remains; no shared native binary was
overwritten. Existing unrelated dirty changes, including generic scatter
and prior reciprocal-grid work, were preserved. No custom CUDA was added
by this re-review.

### Precision, official loading and commit preparation

MatPES/Polar now share `energy_dtype` (fp64 default, fp32 or model dtype
explicitly selectable), `configure_input_dtype`, and a single eager/compiled
dispatch. Both runners expose state/model/energy/GEMM precision separately.
`energy_dtype=None` survives official construction; it is not filtered out as
an absent architecture default. MatPES no longer monkeypatches `energy_core`,
loads in the model dtype before casting geometry, freezes inference parameters,
and does its startup/minimization force evaluations under `no_grad`.
Its compiler uses dynamic edge counts because the MD adapter trims live edges;
the old static CUDA graph claim was incorrect. MatPES saves a `.run.json`
numerical contract next to each output segment. Restart precision matching is
still the caller's responsibility; this is not bitwise restart certification.

The unsafe `_pack_e3nn_contractions` concatenation was removed. It did not
perform the upstream CG basis projection. Official converted tensor weights
plus JSON are preferred; `converted_weights=` can use a trusted original
dump for metadata while installing converted weights **once**. Pickles need
explicit `trusted=True`, and arbitrary deserialization errors no longer
trigger a pickle fallback. The legacy `e3nn2cueq` reader alias is deprecated.
Missing-parameter checks now precede writes. Key collisions and arbitrary
equal-numel reshapes are rejected; cuEq Linear's `(W,)` ↔ `(1,W)` shared-weight
axis remains explicitly supported (verified with the actual Polar archive).

The original dump has `kspace_cutoff_factor=1.5` but a serialized runtime
`kspace_cutoff=1.8467166423797607 Å⁻¹`, corresponding to a different heuristic
factor. **The runtime cutoff is authoritative**, supplied explicitly to both
GTO grids. Inferring the grid from the stored factor alone changes the model.
The old default happened to approximate this runtime cutoff; there is no
evidence that all previous measurements used the wrong physical cutoff.
Polar widths, radial settings, recursion and electrostatic flags are extracted
instead of silently relying on the Polar-1-M defaults.

New GPU records (GH200, torch 2.12.0+cu130, cuEq 0.10.0; 2026-09-05 job logs):

| Job | Configuration | Median fixed E/F/virial | Peak allocated / reserved |
| --- | --- | ---: | ---: |
| 2063743 | 8,850 atoms, fp64 state/model/energy, blocked 4096, eager | 1.0351 s | 32.87 / 47.52 GiB |
| 2063744 | 8,850 atoms, fp64 state/energy, strict fp32 model, full cuEq, compiled | 0.2636 s | 36.57 / 70.12 GiB |

2063744 versus 2063743: energy error 0.0019583 eV; force RMSE 6.6934e-6
eV/Å, max 5.3390e-5 eV/Å; max virial error 0.0016469 eV. Compiled versus
its eager preflight: max E/F/W differences 0.0035043 eV / 5.1016e-5 eV/Å /
0.0041372 eV. Repeat spread: 0.0011345 eV / 3.0191e-5 eV/Å / 0.0025168 eV.
These are fixed-coordinate measurements, **not full MD-step throughput**.
The fp64 geometry significantly reduces the mixed-precision force error on
this fixture; it does not establish long-time energy conservation.

Official compatibility checks:

- Polar isolated water: dE=1.255e-7 eV, max dF=1.343e-7 eV/Å, against frozen
  upstream goldens (`regressions/mace-polar-official-water.py`).
- Job 2063791: 4,995-atom periodic PEO official per-term energy regression
  passes, total dE=3.01e-5 eV (unchanged 1e-3 eV tolerance).
- Actual MatPES converted weights plus JSON: 30 checkpoint tests passed,
  including parameter parity. 193-atom fp64-state/fp32-model three-step NVE
  smoke passed on CPU and fused GH200. Its short, initially unrelaxed path
  is deliberately **not** a drift acceptance test. This box requires
  `--skin 0.1`; default skin 1 exceeds half its cell width and correctly raises.
- Shared MACE/GTO fast suite: 337 passed, 1 skipped; both energy-core
  fullgraph tests passed. Final affected suite after trajectory precision and
  snapshot fixes: **565 passed, 7 skipped, 18 deselected**, including the
  flat-force live-edge leaf regression and the complete MACE package.
- Whole-tree Ruff check/format and `git diff --check` pass. Repository `ty`
  completes with exit zero under its configured warning policy, but reports
  468 diagnostics (including dynamic TensorDict/nn.Module typing); it is not
  a warning-clean type audit.

The x86 editable molrs Python tree had advanced beyond its binary. Tests used
an existing coherent 0.14.0 wheel extracted into `.scratch/mace-env-qxNHBt`,
via PYTHONPATH, without overwriting shared source/binaries or hiding version
metadata. Keep scratch wheels, checkpoints, job logs and benchmark outputs out
of commits. All four new GPU jobs completed successfully.

MatPES compiled-entry follow-up 2067955: three GPU NVE steps complete with
`fullgraph=True, dynamic=True`. The 113.9 s includes cold compilation and is
not a steady-state speed measurement. Compared with eager job 2063791,
saved total-energy difference is 1.16e-7 in MD energy units. This comparison
exposed `TrajectoryHook` forcing positions/velocities/forces to fp32 at save
time, even for fp64 MD. The writer now preserves the input dtype by default;
`dtype=` / MatPES `--trajectory-precision` select explicit fp32/fp64 storage.
CPU capture also makes an independent copy so frames cannot alias a live MD
buffer. Mirrored tests exercise sub-fp32 digits and post-capture mutation.
The three-step MatPES CPU recheck saved `pos`, `vel`, `forces` and `etot` in
float64, with distinct position frames and the correct `.run.json` contract.
The final shard merge still materializes the whole trajectory; capture's
`flush_every` is not a total host-memory bound. No long-run memory claim is made.

Input SHA256 fingerprints captured during validation:

| Asset | SHA256 |
| --- | --- |
| Polar original `MACE-POLAR-1-M.model` | `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a` |
| `polar_cueq_state.pt` | `19a8927b67a58a8d38a3454b5b560c1825fb1374def5ad0b41ddaa65606681a5` |
| `matpes_r2scan_cueq_state.pt` | `91b603831562a126f902c197b00fd51c12b1e8816a0d0e16c19f61a112b625f0` |
| `matpes_r2scan_config.json` | `739f4ee166b5064774d620976198dc915258dbcc6a143f24f1929b4f091043c7` |
| 8,850-atom `dp25_n50.pt` | `db09d2b6a18641abe7501253d06c050387b6988a427f95a4ecd3122327cb5ee9` |

Commit review groups (not yet staged or committed):

1. Existing generic scatter/reciprocal-grid changes: retain ownership and
   review separately; they predate this cleanup.
2. cuEq receiver-block scheduling, recomputation, uvu reassociation and mirrored
   tests; no new local TP/CG kernel.
3. Shared precision, strict official loader, runners/regressions, documentation
   and tests. Public usage: `docs/molzoo/user-guide/mace.md`.

Do not label these commits production-trained, long-NVE-validated or
multi-GPU-capable. Those acceptance gates remain open; multi-GPU is the design
above, not an implemented distributed execution strategy.
