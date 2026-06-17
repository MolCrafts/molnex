# `pair_style molnex` — one LAMMPS pair style for every molnex potential

A single generic LAMMPS pair style that evaluates **any** molnex potential
exported via `molix.lammps.export_for_lammps`. There is no per-model C++: the
pair style reads everything model-specific (cutoff, native units, compute dtype,
supported species, capabilities) from the export directory's
`<name>.meta.json` `lammps` block, and runs the AOT-Inductor `.so` through the
shared `molnex::interface::ModelRunner`.

```
pair_style molnex <model_dir> [name]
pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN>
```

* `<model_dir>` — an export directory produced by `export_for_lammps`
  (contains `<name>.so`, `<name>.pt`, `<name>.meta.json`).
* `[name]` — artifact basename inside `<model_dir>` (default `model`).
* `pair_coeff` maps each LAMMPS atom type to an atomic number `Z`.

## The contract (what makes a model "molnex-LAMMPS compatible")

The protocol boundary is the **AOTI export calling convention**, not a model's
in-memory `forward` — a `TensorDict` cannot cross into C++. The exported `.so`
must take flat tensors and return flat tensors:

| direction | tensors |
|-----------|---------|
| inputs  | `Z (N,)` int64 · `pos (N,3)` float · `edge_index (E,2)` int64 (sorted by `(src,tgt)`) |
| outputs | `energy ()` scalar float · `forces (N,3)` float |

`N` (atoms) and `E` (edges) are exported as **dynamic** dimensions, so one `.so`
serves every MD frame. Units, cutoff, dtype, and species travel in `meta.json`.

## Exporting a model (Python side)

### molnex-native potential (PiNet, etc.)

```python
from molix.lammps import export_for_lammps

export_for_lammps(
    pinet_potential,                 # nested-TensorDict forward → {"energy","forces"}
    "pinet_aspirin",
    species=[1, 6, 7, 8],
    cutoff=pinet_potential.encoder.config.r_max,
    units="real",                    # model native: real=kcal/mol, metal=eV
    device="cuda",                   # or "cpu"
)
```

### third-party model

Anything that isn't a molnex potential just needs a Python-side adapter — no C++.
Two cases:

1. **Already flat** (`forward(Z, pos, edge_index) -> (energy, forces)`): use the
   built-in `flat` adapter.
   ```python
   export_for_lammps(my_model, "out", species=[1,8], cutoff=5.0, adapter="flat")
   ```
2. **Different input/output**: subclass `LammpsAdapter` (~15 lines) — map the flat
   `(Z, pos, edge_index)` onto your model's inputs and read back `(energy, forces)`:
   ```python
   from molix.lammps import LammpsAdapter, export_for_lammps

   class MyNetAdapter(LammpsAdapter):
       name = "mynet"
       def build_inputs(self, model, Z, pos, edge_index):
           return MyGraph(z=Z, r=pos, edges=edge_index)      # your model's input
       def read_outputs(self, out):
           return out.total_energy, out.forces               # () scalar, (N,3)

   export_for_lammps(my_model, "out", species=[1,8], cutoff=5.0, adapter="mynet")
   ```

The adapter name is stamped into `meta.json` for provenance.

## Units

`export_for_lammps(..., units=...)` records the model's native unit system. At
run time `pair_style molnex` reads LAMMPS' active `units` and converts
energy/forces (distance is Å in both `real` and `metal`):

| model `units` | energy   | converted to `units real` | converted to `units metal` |
|---------------|----------|---------------------------|----------------------------|
| `real`        | kcal/mol | ×1                        | ×0.0433641                 |
| `metal`       | eV       | ×23.0605                  | ×1                         |

## Building the plugin

`pair_style molnex` is a **runtime plugin** (`plugin load`), never compiled into
LAMMPS. It is arch+ABI specific, so build it once **from source** against the
exact LAMMPS it will be loaded into — link `liblammps.so` from a
`-DPKG_PLUGIN=on -DBUILD_SHARED_LIBS=on` **source build**, not a binary wheel.
(A wheel that ships `PKG_PLUGIN` can still *load* the plugin at runtime, but the
`.so` itself must be compiled against matching headers/lib — there is no
universal binary.) Requires LibTorch (the active venv's torch) and a C++17
compiler; `molnex_interface` (CPU-only or CUDA, auto-detected) builds from the
parent `interface/` directory.

The output is **arch-tagged** — `<MOLNEXPLUGIN_OUTPUT_DIR>/molnexplugin.so`,
defaulting to `<build>/<arch>/` — so x86_64 and aarch64 plugins coexist (mirrors
the arch-suffixed op lib in `src/molix/op`).

```bash
TORCH_CMAKE=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share/cmake'))")
LMP_SRC=/path/to/lammps-src/src                 # source tree headers (pair.h, lammpsplugin.h, version.h)
LMP_LIB=/path/to/your/source-build/liblammps.so # YOUR build, not the wheel

cmake -S interface/lammps -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DLAMMPS_HEADER_DIR="$LMP_SRC" \
  -DLAMMPS_LIB="$LMP_LIB" \
  -DLAMMPS_MPI_INCLUDE="$LMP_SRC/STUBS"         # serial LAMMPS (BUILD_MPI=off)
cmake --build build -j                          # → build/<arch>/molnexplugin.so

# at runtime
export LD_LIBRARY_PATH="$(dirname $LMP_LIB):$LD_LIBRARY_PATH"
# in the LAMMPS input script:
#   plugin load build/<arch>/molnexplugin.so
#   pair_style molnex pinet_aspirin
#   pair_coeff * * 1 6 7 8
```

A complete, runnable from-source build + benchmark (LAMMPS + plugin, arch-tagged,
no wheel) lives in `work/lammps-iface-bench/` (`build_lammps.sh` + `build_plugin.sh`).

## Scope and roadmap

This version targets the single-molecule regime that the original
`pinet-quant/lammps_native/pair_pinet` proved out:

* single MPI rank, non-periodic (`boundary f f f`) ⇒ no ghost atoms;
* full neighbour list within the cutoff;
* no virial output (`virial_fdotr_compute` fallback for pressure).

The `meta.json` flags `supports_pbc` / `supports_virial` are written (default
`false`) so the C++ can grow domain-decomposition / ghost reverse-communication
and a third virial output **without a protocol change** — a PBC-capable export
just flips the flag and adds `cell` / `edge_shift` inputs to the convention.

> **CPU caveat.** The functorch force path of some molnex potentials (notably
> PiNet) is currently miscompiled by the torch 2.12 CPU inductor backend; use a
> CUDA export for those models, or the eager `fix external` driver for CPU
> prototyping. This is an upstream torch issue, independent of this interface.
