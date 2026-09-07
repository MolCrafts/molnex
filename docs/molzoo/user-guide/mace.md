# MACE: precision and official checkpoints

MatPES and Polar use the same precision/weight-loading boundary. cuEquivariance
owns the equivariant kernels; no e3nn, ASE or mace-torch runtime is needed.

## Choose precision before construction and compilation

```python
import torch
from molix import config
from molzoo.mace import MaceMatpes

# Strict fp32 model calculation; does not enable TF32 GEMMs.
config.set_precision("fp32", matmul_precision="highest")
model = MaceMatpes.load(
    "matpes_r2scan_cueq_state.pt",
    config="matpes_r2scan_config.json",
    energy_dtype=torch.float64,
    compute_forces=False,
).cuda().eval()
model.requires_grad_(False)  # inference only; leave enabled for training
model.configure_input_dtype(torch.float32)
```

The caller may keep positions, velocities and cell in float64. The model
boundary casts geometry differentiably; forces and virial still differentiate
the original MD state. `energy_core` is the lower-level, already-cast kernel;
use `forward`, `energy_forces` or the Molpot interface for the precision boundary.
After one eager validation, `model.compile_core(Compiler(...))` compiles that
same kernel. Choose all precision settings first.

| Control | Meaning |
| --- | --- |
| `config.set_precision("fp64")` before loading | Construct parameters in fp64; `fp32` selects fp32 parameters |
| `matmul_precision="highest"` | Strict fp32 matmul; `high` allows reduced internal precision |
| `energy_dtype=torch.float64` (default) | Sum graph energies in fp64, preserving small interactions alongside large atomic baselines |
| `energy_dtype=torch.float32` / `None` | Explicit fp32 / model-dtype accumulation |
| `configure_input_dtype(torch.float32)` | fp32 model inputs with independently chosen MD-state dtype |

fp64 accumulation does **not** turn an fp32 model into an fp64 model. Casting
stored fp32 weights or coordinates to fp64 cannot recover lost digits. For a
precision reference, construct/load in fp64 from the highest-precision source
available and use fp64 MD state. No policy here promises bitwise deterministic
GPU scatter reductions. Assess fixed-coordinate repeat spread and long NVE
energy drift with declared tolerances; NPT thermostat/barostat runs cannot
establish energy conservation.

Both runners expose `--precision`, `--potential-precision`,
`--energy-precision {model,fp32,fp64}` and `--matmul-precision {highest,high}`:

```sh
# MatPES NVE reference; choose skin consistent with half the cell width.
python scripts/matpes_port/run_nve.py --system system.pt \
  --weights-dir weights --out nve.pt --precision fp64 \
  --potential-precision fp64 --energy-precision fp64 --matmul-precision highest

# Polar bounded E/F/virial benchmark, fp64 state with strict fp32 model.
python scripts/polar_port/run_npt.py --system peo_box.pt \
  --cueq-state polar_cueq_state.pt --config polar_config.json \
  --precision fp64 --potential-precision fp32 --energy-precision fp64 \
  --matmul-precision highest --cueq-target-block-size 0 \
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
precision-sensitive MD policy. MatPES compilation uses dynamic edge counts:
the MD adapter trims live-prefix edges, so this runner does not promise static
CUDA graph replay. Polar compiled MD remains gated behind benchmark mode.

## Official weights: convert once, load once

Preferred deployment is the official **cuEq-converted tensor state plus JSON
configuration** (or a tensor-safe `state_dict`/`config` envelope). Perform the
official basis conversion in a separate, version-pinned environment, then
retain the original/converted hashes, converter versions and numerical parity
record. That conversion environment may require upstream packages; inference
and training in MolNex do not.

For a verified local Polar dump with architecture metadata:

```python
from molzoo.mace import MacePolar

model = MacePolar.load(
    "MACE-POLAR-1-M.model",
    converted_weights="polar_cueq_state.pt",
    trusted=True,
    energy_dtype=torch.float64,
)
```

Only converted weights are installed. Raw e3nn contraction weights are rejected:
concatenating them is not the required CG basis projection. Nothing in molnex
converts an e3nn basis — run the version-pinned official converter out of tree.
Pickle loading requires `trusted=True`; import stubs are not a security sandbox.
Use tensor-only state plus JSON for deployment from untrusted sources.

Loading preflights missing parameters, key collisions and layout mismatches
before writing tensors. Only scalar ranks and cuEq Linear's singleton shared
weight axis are normalized; arbitrary same-numel reshapes are rejected.
The saved Polar `kspace_cutoff` overrides the constructor heuristic because
official dumps can contain a stale `kspace_cutoff_factor`. The two GTO grids
receive that same explicit cutoff. Unsupported Polar architecture switches
fail rather than silently changing the fitted energy surface.

## Memory and multi-GPU boundary

Full indexed cuEq convolution is fastest in the tested near-10k case when it
fits. `model.configure_cueq_target_blocks(4096)` trades speed for lower peak
memory using official kernels and checkpoint recomputation. The neighbor list
must use the same maximum receiver block size; arbitrary collated COO cannot
be passed to a receiver-blocked model. Runtime scheduling adds no weight keys.

Current measurements use **8,850**, not exactly 10,000, PEO atoms on GH200.
They validate bounded energy/force/virial evaluation, not long-trajectory
stability or force-loss training. Multi-GPU domain decomposition is a
[design](../../../.claude/notes/mace-polar-10k-review.md), not a shipped backend:
DDP does not split one large periodic system, and FSDP does not solve its
activation/edge memory. A future atom-domain backend needs differentiable halo
exchange, global Fukui normalization and structure-factor reduction before
Hartree energy, while retaining the same official parameter keys.
