# MACE: precision and official checkpoints

MatPES, OMol, OFF, LES and Polar share one construction, one precision
boundary and one weight-loading path. cuEquivariance owns the equivariant
kernels; no ASE or mace-torch runtime is needed.

## One spec, one product, two methods

A product is constructed from its typed spec, and every default lives on the
spec:

```python
from molzoo.mace import MACEMatpesSpec, MaceMatpes

spec = MACEMatpesSpec(atomic_numbers=[1, 6, 8], atomic_energies=[-13.6, -1029.0, -2041.0])
model = MaceMatpes(spec, wrt=("atoms.pos",))   # forward writes graphs.energy and atoms.forces
```

A product has two public methods. `energy(frame)` reads the post-collate
batch (`atoms.{Z,pos,batch}`, `pairs.pair_index`, the imaged `pairs.pair_diff`,
optional `atoms.mask`, `graphs.{total_charge,total_spin}`, `graphs.cell`) and
writes `atoms.energy` and `graphs.energy`; `forward` is the
`molix.autograd.Derivative` built from `wrt=` at construction and adds the
derivatives (`atoms.forces`, `graphs.virial`). `Derivative(model.energy, wrt=...)`
composed outside is the same op. MACE-LES and Polar carry a graph-level
electrostatic term: their `graphs.energy` is the sum of `atoms.energy` plus
that term.

## Choose precision before construction and compilation

```python
import torch
from molix import config
from molzoo.mace import MaceMatpes

# fp32 parameters; the matmul policy stays PyTorch's default (no TF32).
config.set_precision("fp32")
model = MaceMatpes.load(
    "matpes_r2scan_cueq_state.pt",
    config="matpes_r2scan_config.json",
    energy_dtype=torch.float64,
    input_dtype=torch.float32,   # fp64 MD state drives the fp32 model
    wrt=("atoms.pos",),
).cuda().eval()
model.requires_grad_(False)  # inference only; leave enabled for training
```

The caller may keep positions, velocities and cell in float64. `energy` casts
the geometry to `input_dtype` at its top, differentiably: forces and virial
still differentiate the original MD state and come back at its dtype. After
one eager forward, `Compiler(...)(model)` compiles the whole op. Choose all
precision settings first.

| Control | Meaning |
| --- | --- |
| `config.set_precision("fp64")` before loading | Construct parameters in fp64; `fp32` selects fp32 parameters |
| `energy_dtype=torch.float64` (default) | Accumulate per-atom and graph energies in fp64, preserving small interactions alongside large atomic baselines |
| `energy_dtype=torch.float32` / `None` | Explicit fp32 / model-dtype accumulation |
| `input_dtype=torch.float32` | fp32 model inputs with an independently chosen MD-state dtype |

fp64 accumulation does **not** turn an fp32 model into an fp64 model. Casting
stored fp32 weights or coordinates to fp64 cannot recover lost digits. For a
precision reference, construct/load in fp64 from the highest-precision source
available and use fp64 MD state. No policy here promises bitwise deterministic
GPU scatter reductions. Assess fixed-coordinate repeat spread and long NVE
energy drift with declared tolerances; NPT thermostat/barostat runs cannot
establish energy conservation.

Both runners expose `--precision {fp64,fp32}`, `--potential-precision` and
`--energy-precision {model,fp32,fp64}`; the MatPES runner adds
`--autocast-bf16`, `--compile` and `--trajectory-precision {state,fp32,fp64}`,
the Polar runner `--config`, `--compile-core` and `--benchmark-repeats` /
`--benchmark-warmup` / `--benchmark-output`. The fp32 matmul policy is not a
flag; it stays PyTorch's default.

```sh
# MatPES NVE reference; choose skin consistent with half the cell width.
python scripts/matpes_port/run_nve.py --system system.pt \
  --weights-dir weights --out nve.pt --precision fp64 \
  --potential-precision fp64 --energy-precision fp64

# Polar bounded E/F/virial benchmark, fp64 state with strict fp32 model.
python scripts/polar_port/run_npt.py --system peo_box.pt \
  --cueq-state polar_cueq_state.pt --config polar_config.json \
  --precision fp64 --potential-precision fp32 --energy-precision fp64 \
  --compile-core --benchmark-repeats 5
```

MatPES saves requested settings and library versions in `.run.json`, including
for interrupted runs. Preserve that sidecar with restart checkpoints; restart
files alone do not validate matching weights, precision or neighbor policy.
Trajectory position/velocity/force storage defaults to the MD-state dtype
(`--trajectory-precision state`). Select `fp32` explicitly to reduce disk use;
`TrajectoryHook(dtype=None)` preserves precision and captures independent CPU
snapshots. Energies retain their accumulation dtype regardless of this option.
`flush_every` bounds capture buffers, not the final full-trajectory shard merge;
choose stride and retained-frame count to fit host memory for long tests.
BF16 autocast remains an explicit experimental MatPES option, not an accepted
precision-sensitive MD policy. MatPES compiled MD runs on static shapes: the
MD driver hands the op the neighbour list's full capacity buffer every step,
so `Compiler(cuda_graphs=True)` replays one recorded graph
(`docs/molix/user-guide/md.md`). Polar compiled MD remains gated behind
benchmark mode.

## Official weights: convert once, load once

Preferred deployment is the official **cuEq-converted tensor state plus JSON
configuration** (or a tensor-safe `state_dict`/`config` envelope). Perform the
official basis conversion in a separate, version-pinned environment, then
retain the original/converted hashes, converter versions and numerical parity
record. That conversion environment may require upstream packages; inference
and training in MolNex do not.

`MaceXxx.load(path, config=None, *, wrt, energy_dtype, input_dtype)` is three
steps every product spells the same way, and a caller may compose them
itself: `MaceArchive().read(path, config)` returns the weights and a typed
`MaceOfficialConfig`; the family's spec builds itself from that config with
`from_official` (holding the family's rejection rules — a MatPES config does
not build a MACE-OFF model); `CheckpointRemap(keys, skip=MACE_SKIP).load`
renames the weights in. A pickled Polar dump whose weights sit in a separate
converted state is the composed form:

```python
from molzoo.checkpoint import CheckpointRemap
from molzoo.mace import MACEPolarSpec, MacePolar
from molzoo.mace.checkpoint import MACE_SKIP, polar_keys
from molzoo.mace.utils import MaceArchive

state, cfg = MaceArchive(converted_weights="polar_cueq_state.pt").read("MACE-POLAR-1-M.model")
model = MacePolar(MACEPolarSpec.from_official(cfg), energy_dtype=torch.float64)
CheckpointRemap(polar_keys, on_unexpected="return", skip=MACE_SKIP).load(model, state)
```

Only converted weights are installed. Unconverted (pre-cuEquivariance) contraction weights are rejected:
concatenating them is not the required CG basis projection. Nothing in molnex
converts the upstream basis — run the version-pinned official converter out of tree.
Pickled modules are read through the shared stub unpickler
(`molzoo.checkpoint.OfficialArchive`); it is an import aid, not a sandbox —
read only archives you trust.

Loading preflights missing parameters, key collisions and layout mismatches
before writing tensors. Only scalar ranks and cuEq Linear's singleton shared
weight axis are normalized; arbitrary same-numel reshapes are rejected.
The saved Polar `kspace_cutoff` overrides the constructor heuristic because
official dumps can contain a stale `kspace_cutoff_factor`. The two GTO grids
receive that same explicit cutoff. Unsupported Polar architecture switches
fail rather than silently changing the fitted energy surface.

## Memory and multi-GPU boundary

The convolution is the full indexed cuEq kernel; there is no memory-bounded
alternative layout (the receiver-blocked one was removed on 2026-09-09: slower
at every size, and its memory saving did not survive `torch.compile`).

Current measurements use **8,850**, not exactly 10,000, PEO atoms on GH200.
They validate bounded energy/force/virial evaluation, not long-trajectory
stability or force-loss training. Multi-GPU domain decomposition is a
[design](../../../.claude/notes/mace-polar-10k-review.md), not a shipped backend:
DDP does not split one large periodic system, and FSDP does not solve its
activation/edge memory. A future atom-domain backend needs differentiable halo
exchange, global Fukui normalization and structure-factor reduction before
Hartree energy, while retaining the same official parameter keys.
