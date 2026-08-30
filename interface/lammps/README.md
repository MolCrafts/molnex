# `pair_style molnex` — one LAMMPS pair style for every molnex potential

Classical Class-I / OPLS parameter tables do **not** go through this pair
style. Compile `ForceField.to_ir()` with `LammpsAdapter` and write
`system.data` / `system.in.settings` via `LammpsWriter` — see
[Force-field export](../../docs/molix/user-guide/ff-export.md). This
document is the AOTI neural-potential path only.

A single generic LAMMPS pair style that evaluates **any** molnex potential
exported via `molix.engine.export_for_lammps`. There is no per-model C++: the
pair style reads everything model-specific (cutoff, native units, compute dtype,
capabilities) from the export directory's `<name>.meta.json` `lammps` block, and
runs the AOT-Inductor `.pt2` package through the shared
`molnex::interface::ModelRunner`. The block also records the model's supported
species, which today is provenance — the check of your `pair_coeff` atomic
numbers against it lands with `lammps-iface-02-view`.

```
pair_style molnex <model_dir> [name]
pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN>
```

* `<model_dir>` — an export directory produced by `export_for_lammps`
  (contains `<name>.pt2` and `<name>.meta.json`).
* `[name]` — artifact basename inside `<model_dir>` (default `model`).
* `pair_coeff` maps each LAMMPS atom type to an atomic number `Z`.

## The contract (what makes a model "molnex-LAMMPS compatible")

The protocol boundary is the **AOTI export calling convention**, not a model's
in-memory `forward` — a `TensorDict` cannot cross into C++. The exported `.pt2`
must take flat tensors and return flat tensors:

| direction | tensors |
|-----------|---------|
| inputs  | `Z (N,)` int64 · `pos (N,3)` float (Å) · `edge_index (E,2)` int64 (sorted by `(src,tgt)`) |
| outputs | `energy ()` scalar float · `forces (N,3)` float — both in the model's native units |

Two optional slots widen that table, and the meta `inputs` / `outputs` lists say
which of them a given export uses: a fixed-shape (`execution: "cuda-graph"`)
export takes a fourth input `mask (E,)` bool, and a `supports_virial=true` export
returns a third output `virial (6,)` ordered `(xx, yy, zz, xy, xz, yz)`.

On the default (`execution: "dynamic"`) path `N` (atoms) and `E` (edges) are
exported as **dynamic** dimensions, so one `.pt2` serves every MD frame; the
fixed-shape path pins them instead (`n_atoms` / `e_max` in the meta) because a
CUDA graph can only be captured over static shapes. Units, cutoff, dtype, and
species travel in `meta.json`.

## Exporting a model (Python side)

### molnex-native potential (PiNet, etc.)

```python
from molix.engine import export_for_lammps

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
2. **Different input/output**: subclass `EngineAdapter` (~15 lines) — map the flat
   `(Z, pos, edge_index)` onto your model's inputs and read back `(energy, forces)`:
   ```python
   from molix.engine import EngineAdapter, export_for_lammps

   class MyNetAdapter(EngineAdapter):
       name = "mynet"
       def build_inputs(self, model, Z, pos, edge_index):
           return MyGraph(z=Z, r=pos, edges=edge_index)      # your model's input
       def read_outputs(self, out):
           return out.total_energy, out.forces               # () scalar, (N,3)

   export_for_lammps(my_model, "out", species=[1,8], cutoff=5.0, adapter="mynet")
   ```

The adapter name is stamped into `meta.json` for provenance.

## Units

`export_for_lammps(..., units=...)` records the model's native unit system
(`real` = kcal/mol, `metal` = eV). The export **also stamps an `energy_conv`
table** into the meta `lammps` block: one entry per LAMMPS unit style, holding
the value of one model energy unit expressed in that style. Those numbers are
evaluated on the Python side by `molix.units` (pint), which is the only place
the conversion is ever computed — `energy_conv[units]` is exactly `1.0` by
construction.

At run time `pair_style molnex` looks up LAMMPS' active unit style
(`update->unit_style`) in that table, and refuses — listing every style the
export serves — when there is no entry for it. **No conversion factor is written
in the C++, and none is written in this README**: the numbers your model is
actually scaled by are in your own export's `<name>.meta.json`, which is the one
copy that cannot drift from the model it describes.

Forces reuse the energy factor, and that reuse is *checked* rather than assumed.
`[F] = [E]/[L]`, so the identity `force_conv = energy_conv` holds only while the
model measures length the way LAMMPS does. The meta therefore also carries
`length_unit`, and the pair style asserts it is `angstrom` (LAMMPS `real` and
`metal` both use Å) before setting the force factor. A model exported in Bohr
fails loudly instead of being silently mis-scaled by the Bohr/Å ratio; serving
one would take a schema v3 with a length table, which is deliberately not
pre-built.

## Schema versions

The meta `lammps` block is versioned by `schema_version`, currently **2**. The
plugin and the exports it loads are upgraded as a pair.

**v1 → v2 (breaking).** A v2 export carries `schema_version`, `contract`,
`execution`, `energy_conv` and `length_unit`, and **drops** the `cuda_graph`
boolean. Execution strategy is now `execution: "dynamic" | "cuda-graph"`, and
`contract: "cutoff-graph" | "fixed-graph"` declares where the model's edge set
comes from (`fixed-graph` has no implementation yet and is refused explicitly,
never quietly run down the cutoff path).

The two mixing directions are not symmetric:

* **v2 plugin + v1 export** — refused loudly. The contract validates
  `schema_version == 2` while loading and tells you to re-export.
* **v1 plugin + v2 export** — *unsupported, and undetectable from inside the old
  binary*. A v1 plugin read `get_bool(meta, "cuda_graph", false)`; with the key
  gone it falls back to `false` and runs a CUDA-graph export down the dynamic
  path without a word. Rebuild the plugin when you move to v2 exports.

## Building the plugin

`pair_style molnex` is a **runtime plugin** (`plugin load`), never compiled into
LAMMPS. It is arch+ABI specific, so build it once **from source** against the
exact LAMMPS it will be loaded into — link `liblammps.so` from a
`-DPKG_PLUGIN=on -DBUILD_SHARED_LIBS=on` **source build**, not a binary wheel.
(A wheel that ships `PKG_PLUGIN` can still *load* the plugin at runtime, but the
`.so` itself must be compiled against matching headers/lib — there is no
universal binary.) Requires LibTorch (the active venv's torch) and a C++17
compiler; `molix.engine` (CPU-only or CUDA, auto-detected) builds from the
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

A complete from-source build + benchmark workspace (LAMMPS + plugin, arch-tagged,
no wheel — `build_lammps.sh` + `build_plugin.sh`) is kept **outside this
repository** as developer scratch and is not shipped with molnex; the recipe
above is the supported path.

## Scope and roadmap

### Open-boundary contract (`supports_pbc=false`)

Single-molecule / one-domain regime:

* single MPI rank, non-periodic (`boundary f f f`) ⇒ no ghost atoms;
* full neighbour list within the cutoff;
* `virial_fdotr_compute` fallback for pressure unless the model emits virial.

### Ghost / PBC path (`supports_pbc=true`)

Export with `export_for_lammps(..., supports_pbc=True)`. At run time the pair
style:

* builds `Z` / `pos` over **local + ghost** atoms (ghost coordinates already
  carry the minimum-image placement from LAMMPS);
* includes ghost neighbours in `edge_index` (still sorted by `(source,target)`);
* scatters forces onto ghosts and **reverse-communicates** them to owners;
* allows multi-rank domain decomposition.

Models that recompute `edge_diff = pos[target] - pos[source]` from positions
need no separate `shifts` input in this cut. A future export may still add
explicit `cell` / `edge_shift` tensors for models that require them.

A fixed-shape export — `export_for_lammps(..., cuda_graph=True)` on the Python
side, `execution: "cuda-graph"` in the meta — is **incompatible** with ghosts:
the ghost count changes as atoms move between domains, while the captured graph
needs a fixed `N`. The pair style refuses the combination while loading the
model rather than at the first step. Re-export with `cuda_graph=False` (meta
`execution: "dynamic"`) for periodic / multi-rank MD.

### Virial (`supports_virial=true`)

If the AOT package returns a third tensor of shape `(6,)`
`(xx, yy, zz, xy, xz, yz)` in the model energy unit, it is converted and
accumulated into the global virial. Otherwise LAMMPS falls back to
`virial_fdotr_compute` when fdotr virial is requested.

> **CPU caveat.** The functorch force path of some molnex potentials (notably
> PiNet) is currently miscompiled by the torch 2.12 CPU inductor backend; use a
> CUDA export for those models, or the eager `fix external` driver for CPU
> prototyping. This is an upstream torch issue, independent of this interface.
