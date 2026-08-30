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
species, and every `pair_coeff` atomic number is checked against that list at
`pair_coeff` time — an element the model was never exported for is refused
rather than fed forward as an embedding index the network has no row for.

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

## `special_bonds` requirement (molecular systems)

**A molecular system must set `special_bonds lj/coul 1.0 1.0 1.0`.** The pair
style refuses anything else at `init` time, and the refusal is not pedantry:

```
# in your input script, for any system with bond topology
special_bonds lj/coul 1.0 1.0 1.0
```

LAMMPS' default is `special_bonds lj/coul 0.0 0.0 0.0`, and a coefficient of
`0.0` does not scale a pair down — it **deletes that pair from the neighbour
list entirely**. Every 1-2, 1-3 and 1-4 pair, i.e. exactly the bonded nearest
neighbours, would then be invisible to the cutoff graph this pair style builds.
A many-body ML potential would be evaluated on a graph missing its closest
edges: wrong energies, wrong forces, and — before this check existed — no
diagnostic anywhere.

A coefficient that is neither `0.0` nor `1.0` is refused for the same reason
from the other direction: it says LAMMPS expects those interactions to be
scaled, and molnex applies no special-bond scaling of its own, so the run would
silently disagree with the input script.

The check fires only when the system has bond topology (`atom->molecular != 0`).
An `atom_style atomic` system has no special pairs at all, so it runs whatever
its unused coefficients say.

## Per-atom energy and stress are refused, not zeroed

`compute pe/atom` and `compute stress/atom` (and anything that pulls them in,
such as a per-atom dump or `fix ave/atom` over them) **error out** under
`pair_style molnex`.

The exported models return a total energy, per-atom forces and an optional
global virial — there is no per-site decomposition to hand back. Until this
version, LAMMPS' `ev_init` allocated `eatom` / `vatom`, this pair style never
wrote them, and the user received a legal-looking all-zero field: a per-atom
energy map that was silently, uniformly wrong. The pair style now says so
instead. Global thermo output (`etotal`, `press` via the model virial or
`virial_fdotr_compute`) is unaffected.

A contract that produces per-site energies can fill them in later; the refusal
is what a contract that cannot must do in the meantime.

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

### Weight hot-reload (`num_models`) — the opening, on record

`molnex::interface::ModelRunner`'s constructor takes a `num_models` count: the
number of constant buffers the underlying AOTI runner allocates. That parameter
is the designed seam for **double-buffered hot reload** (`update_weights`), in
which the active buffer keeps serving `run()` calls while the inactive one is
rewritten and a single atomic swap flips them — no inference pause, no
caller-visible lock. It is what an on-line re-parameterisation loop (train a
little, swap the weights, keep integrating) would be built on.

**`pair_style molnex` passes `num_models = 1` today, and there is no keyword to
change it.** The pair style has no hot-reload consumer, and a command-line
grammar with nothing to drive it is syntax nobody earned: it would have to be
parsed, validated, documented and kept compatible before a single caller
existed. The grammar therefore ships with the spec that introduces the consumer
(a `fix` that reloads weights during a run), not before. This paragraph exists
so the seam is *recorded* rather than rediscovered — the C++ side needs no
change to support it.

Two version notes bind that future keyword, and are the reason it cannot be a
free-floating option:

* **torch ≥ 2.12 makes the two modes mutually exclusive.** `num_models > 1`
  requires `run_single_threaded = false`; the runner refuses the combination
  `num_models > 1` with `run_single_threaded = true` outright.
* **Hot reload therefore excludes CUDA-graph capture.** Capture requires the
  single-threaded runner (the multi-threaded `run()` does a capture-illegal
  allocation/synchronisation, PyTorch #158834), so a hot-reloadable run is an
  `execution: "dynamic"` run. An export with `execution: "cuda-graph"` and a
  hot-reload request is a contradiction the future keyword must reject rather
  than resolve.
