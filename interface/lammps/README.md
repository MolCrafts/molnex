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
must take flat tensors and return flat tensors, **positionally**, in exactly the
order the meta's `inputs` / `outputs` lists give:

| position (dynamic / cuda-graph) | input | shape | dtype | present when |
|---|-------|-------|-------|--------------|
| 0 / 0 | `Z` | `(N,)` | `int64` | always |
| 1 / 1 | `pos` | `(N,3)` | model dtype (Å) | always |
| 2 / 2 | `edge_index` | `(E,2)` | `int64` | always |
| — / 3 | `edge_mask` | `(E_max,)` | `bool` | `execution: "cuda-graph"` only |
| 3 / 4 | `local_mask` | `(N,)` | `bool` | always — **last** in either list |

| # | output | shape | dtype | present when |
|---|--------|-------|-------|--------------|
| 0 | `energy` | `()` | model dtype | always |
| 1 | `forces` | `(N,3)` | model dtype | always |
| 2 | `atomic_energy` | `(N,)` | model dtype | always |
| 3 | `virial` | `(6,)` | model dtype | `supports_virial: true` only |

So the input list is `["Z", "pos", "edge_index", "local_mask"]` (4) on the
dynamic path and `["Z", "pos", "edge_index", "edge_mask", "local_mask"]` (5) on
the cuda-graph path — `local_mask` stays last and `edge_mask` is inserted before
it. `LammpsContract` refuses any other arity while loading (`kDynamicInputs` /
`kCudaGraphInputs` in `lammps_contract.h`).

Energies, forces and virial are all in the **model's** native units; the pair
style converts. `virial` is ordered `(xx, yy, zz, xy, xz, yz)`.

**Two masks, two different things.** `local_mask (N,)` marks the atoms this rank
OWNS, and the exported energy is `(atomic_energy * local_mask).sum()` — the
partition sum that makes multi-rank and periodic runs correct. `edge_mask
(E_max,)` exists only on the fixed-shape path and marks which rows of a padded
`edge_index` are real. v2 called that one just `mask`; v3 renames it to
`edge_mask` in the same bump that introduces `local_mask`, because two
differently-meaning masks sitting in one positional list under the names `mask`
and `local_mask` is a mis-wiring waiting to happen.

**Edge order is not promised by default.** The C++ side builds `edge_index` in
the LAMMPS neighbour list's own order and does **not** sort, because every
in-tree model aggregates with `scatter` / `index_add`-class reductions, which are
permutation-invariant. A model that needs rows grouped by index (segmented or
CSR aggregation) must declare `requires_sorted_edges=True` at export; then, and
only then, does the C++ side sort by `(source, target)`.

On the default (`execution: "dynamic"`) path `N` (atoms) and `E` (edges) are
exported as **dynamic** dimensions, so one `.pt2` serves every MD frame; the
fixed-shape path pins them instead (`n_atoms` / `e_max` in the meta) because a
CUDA graph can only be captured over static shapes. Units, cutoff, dtype,
species, `receptive_field_layers` and `requires_sorted_edges` travel in
`meta.json`.

## Exporting a model (Python side)

### molnex-native potential (PiNet, etc.)

```python
from molix.engine import export_for_lammps

export_for_lammps(
    pinet_potential,                 # nested-TensorDict forward
                                     # → {"energy", "forces", "atomic_energy"}
    "pinet_aspirin",
    species=[1, 6, 7, 8],
    cutoff=pinet_potential.encoder.config.r_max,
    units="real",                    # model native: real=kcal/mol, metal=eV
    receptive_field_layers=5,        # T — required, no default (see below)
    device="cuda",                   # or "cpu"
)
```

`receptive_field_layers` is `T`, the number of message-passing layers, and it has
**no default**: a site energy depends on everything within `T * cutoff`, and the
LAMMPS side sizes its ghost shell from that number. Defaulting it to 1 would let
a 5-layer model pass a check computed at a fifth of the radius it needs, so
omitting it is a `ValueError` naming the keyword — not a plausible-looking guess.

### third-party model

Anything that isn't a molnex potential just needs a Python-side adapter — no C++.
Two cases:

1. **Already flat** (`forward(Z, pos, edge_index) -> (energy, forces,
   atomic_energy)`): use the built-in `flat` adapter.
   ```python
   export_for_lammps(my_model, "out", species=[1,8], cutoff=5.0,
                     receptive_field_layers=1, adapter="flat")
   ```
   A model that returns only `(energy, forces)` is refused: schema v3 needs the
   per-site decomposition, and a two-tuple cannot be widened into one.
2. **Different input/output**: subclass `EngineAdapter` (~15 lines) — map the flat
   `(Z, pos, edge_index)` onto your model's inputs and read back
   `(energy, forces, atomic_energy)`:
   ```python
   from molix.engine import EngineAdapter, export_for_lammps

   class MyNetAdapter(EngineAdapter):
       name = "mynet"
       def build_inputs(self, model, Z, pos, edge_index):
           return MyGraph(z=Z, r=pos, edges=edge_index)      # your model's input
       def read_outputs(self, out):
           # () scalar, (N,3), (N,)
           return out.total_energy, out.forces, out.site_energy

   export_for_lammps(my_model, "out", species=[1,8], cutoff=5.0,
                     receptive_field_layers=1, adapter="mynet")
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
one would take a length table in the meta, which no schema version pre-builds —
not even v3, which had every opportunity and no such model to serve.

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

## Per-atom energy is supported; per-atom stress is refused, not zeroed

**`compute pe/atom` works** as of schema v3. A v3 export returns `atomic_energy
(N,)`, the pair style keeps the entries for the atoms this rank owns, and
`eflag_atom` tallies them into LAMMPS' `eatom`. The total energy LAMMPS reports
is the sum of exactly those numbers — `eng_vdwl` and `eatom` are derived from one
tensor, so `compute reduce` over `pe/atom` agrees with `pe` by construction
rather than by coincidence.

**`compute stress/atom` still errors out**, along with anything that pulls it in
(a per-atom dump, `fix ave/atom` over it). A model returns at most a *global*
virial; there is no per-site decomposition of it to hand back. That refusal is
loud on purpose: LAMMPS' `ev_init` allocates `vatom`, and a pair style that
never writes it hands the user a legal-looking all-zero field — a per-atom
stress map that is silently, uniformly wrong. Per-atom energy was refused for
the same reason until it had a producer; per-atom virial still has none.

Global thermo output (`etotal`, `press` via the model virial or
`virial_fdotr_compute`) is unaffected either way.

## Schema versions

The meta `lammps` block is versioned by `schema_version`, currently **3**. The
plugin and the exports it loads are upgraded as a pair.

### v2 → v3 (breaking) — every earlier export is invalid

**Every `.pt2` exported before schema v3 must be re-exported.** This is not a
compatibility shim away: the v3 model takes a different NUMBER of inputs, in a
different order, and returns a different number of outputs in a different order.
Loading a v2 package under a v3 plugin would read a `(N,)` per-atom energy where
a `(6,)` virial is expected. `LammpsContract` therefore refuses any
`schema_version != 3` while loading, naming the re-export.

Re-export with the same arguments you used before, plus `T`:

```python
from molix.engine import export_for_lammps

export_for_lammps(
    model,
    export_dir,
    species=[...],
    cutoff=...,
    receptive_field_layers=<T>,      # NEW, required, no default
    units="real",
    supports_pbc=True,               # for periodic / multi-rank runs
    # requires_sorted_edges=False,   # NEW, default; True only for CSR/segmented
)
```

What v3 adds:

| change | kind | why |
|---|---|---|
| `atomic_energy (N,)` output | new output, index 2 | the partition energy `Σ_{i∈local} E_i` cannot be formed from a scalar; also makes `compute pe/atom` work |
| `local_mask (N,)` input | new input, always last | marks the atoms this rank owns; the exported energy is `(atomic_energy * local_mask).sum()` |
| `mask` → `edge_mask` | rename (cuda-graph only) | two masks in one positional list needed two distinguishable names |
| `receptive_field_layers` | new meta field, **required** | `T`; the LAMMPS side checks `ghost cutoff >= T * cutoff + skin` |
| `requires_sorted_edges` | new meta field, default `false` | declares whether the model needs sorted `edge_index`; when false the C++ side skips a per-step sort |
| `virial` moves to output index 3 | reorder | it now sits behind `atomic_energy` |

**Per-atom energy is authoritative in v3.** The exported energy is *defined* as
`(atomic_energy * local_mask).sum()`; the model's own scalar `energy` is not read
by the export path at all. A model whose total energy contains a term that
cannot be attributed to a site cannot be partitioned across ranks, and is
therefore outside the v3 contract rather than silently mis-decomposed.

Two eager checks run on the example system **before** anything is traced (they do
not enter the exported graph, so `torch.export` is unaffected):

* **decomposability** — `|energy − Σ atomic_energy|` within rel `1e-6`, else
  `ValueError`. This is what catches a non-decomposable total.
* **force consistency** — with an all-ones mask, the wrapper's derived forces
  `-∂(Σ E_i)/∂pos` against the model's own, within rel `1e-5`: the masked-energy
  rewrite must not have changed what is differentiated.

**v1 → v2 (breaking).** A v2 export carries `schema_version`, `contract`,
`execution`, `energy_conv` and `length_unit`, and **drops** the `cuda_graph`
boolean. Execution strategy is now `execution: "dynamic" | "cuda-graph"`, and
`contract: "cutoff-graph" | "fixed-graph"` declares where the model's edge set
comes from (`fixed-graph` has no implementation yet and is refused explicitly,
never quietly run down the cutoff path).

The two mixing directions are not symmetric:

* **v2 plugin + v1 export** — refused loudly. That contract validated
  `schema_version == 2` while loading and told you to re-export; the current
  plugin validates `== 3` and refuses a v2 export the same way.
* **v1 plugin + v2 export** — *unsupported, and undetectable from inside the old
  binary*. A v1 plugin read `get_bool(meta, "cuda_graph", false)`; with the key
  gone it falls back to `false` and runs a CUDA-graph export down the dynamic
  path without a word. Rebuild the plugin when you move to v2 exports.

## Runbook — parity evidence, and when it expires

Some correctness claims here cannot be proven by any test that runs without a
LAMMPS build. They are proven by a **recorded run** instead, and a recorded run
has an owner and an expiry — otherwise the evidence quietly becomes a memory of
a machine nobody has any more.

### Periodic multi-rank parity

| | |
|---|---|
| **claim** | one rank, two ranks and a direct Python evaluation of the same periodic configuration agree on total energy and forces within rel `1e-6` |
| **proves** | the local-only energy partition (no ghost site energy counted twice) **and** the single reduction path (no ghost force communicated twice). Single-rank alone cannot show the second — only a decomposed run has an inter-rank reduction to double. It is also the full-scale edge-order-invariance check: C++ builds edges in raw neighbour-list order while the Python reference builds them sorted, and they agree. |
| **tool** | `scripts/lammps_iface_parity.py` |
| **owner** | the implementer of the `lammps-iface` chain |
| **re-run trigger** | **any change under `interface/` or `src/molix/engine/`** |
| **status** | **PENDING — while no run is attached, this branch must not merge.** The tool exits non-zero without a LAMMPS build; it never skips. |

The tool's Provenance block records the LAMMPS version, rank count,
`atom_style`, `comm_modify` settings, plugin commit, torch version,
device/precision, the capture command and the date. A run whose provenance no
longer matches the tree is expired evidence, not evidence.

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

* single MPI rank, non-periodic (`boundary f f f`) ⇒ no ghost atoms — and that
  is enforced, not assumed: a run that presents ghosts to a `supports_pbc=false`
  export is refused before anything is staged;
* full neighbour list within the cutoff, over every atom in the system, so the
  receptive field is complete at any `T` with no `comm_modify` line;
* `local_mask` is all-ones, and since there are no ghosts the local sum IS the
  total — the same code path as the periodic one, with nothing to exclude;
* newton may be on or off (there is no ghost force to reduce);
* `virial_fdotr_compute` fallback for pressure unless the model emits virial.

### Ghost / PBC path (`supports_pbc=true`)

Export with `export_for_lammps(..., supports_pbc=True)`. At run time the pair
style:

* builds `Z` / `pos` over **local + ghost** atoms (ghost coordinates already
  carry the minimum-image placement from LAMMPS);
* walks ghost centres as well as local ones, so every local atom's complete set
  of incoming edges is in `edge_index` (in neighbour-list order unless the
  export declared `requires_sorted_edges=True`);
* sums site energies over **local atoms only** — `E_rank = Σ_{i∈local} E_i` —
  using `local_mask`, so no ghost's site energy is counted twice;
* scatters forces onto ghosts and leaves them in `atom->f` for the
  **integrator's** global reverse communication to carry home;
* allows multi-rank domain decomposition.

Models that recompute `edge_diff = pos[target] - pos[source]` from positions
need no separate `shifts` input in this cut. A future export may still add
explicit `cell` / `edge_shift` tensors for models that require them.

#### `newton on` is required

A `supports_pbc` run is refused at `init` time unless newton is on (it is
LAMMPS' default). Ghost forces are the derivative of *local* site energies with
respect to a ghost's coordinates, and the only thing that returns them to the
owning atom is `Verlet::run`'s global reverse communication — which LAMMPS
performs **only** `if (force->newton)`. With newton off nothing reduces them and
every periodic cross-term is silently discarded, so the pair style refuses
rather than integrate forces that are quietly incomplete. The pair style issues
no reverse communication of its own; doing both is exactly the double-counting
this path was fixed to remove.

#### `comm_modify cutoff` for `T >= 2` (the fat-ghost condition)

**A `T >= 2` model in a periodic run needs a widened ghost shell.** A `T`-layer
message-passing network's site energy depends on every atom within `T * cutoff`,
and this pair style evaluates the model in a single shot — so each rank must
hold a copy of all of them. LAMMPS' default ghost cutoff is `cutoff + skin`,
which is enough only at `T == 1`.

```
# T = 5, cutoff = 5.0 Å, default skin for real/metal units = 2.0 Å
comm_modify cutoff 27.0        # T * cutoff + skin
```

The rule is

```
comm_modify cutoff  >=  receptive_field_layers * cutoff + neighbor skin
```

and the pair style **checks it and fails closed** — never a warning. The
diagnostic states the measured value, the required value and the exact
`comm_modify cutoff <value>` line to add, because the alternative to failing is
a run whose outer graph sites are quietly missing neighbours: confident, wrong
forces with no diagnostic anywhere.

The check is scoped to `supports_pbc` exports, since it is a question about
ghosts. A `supports_pbc=false` export has none — it requests none and refuses to
run with any — so all of its atoms are graph nodes, its receptive field is
complete at any `T`, and no `comm_modify` line is needed for a single molecule
however deep the model.

A wider ghost shell costs memory and communication that grow with `T * cutoff`;
that is the price of the single-shot scheme, and the fix is a shallower model or
a smaller cutoff, not a smaller `comm_modify`.

A fixed-shape export — `export_for_lammps(..., cuda_graph=True)` on the Python
side, `execution: "cuda-graph"` in the meta — is **incompatible** with ghosts:
the ghost count changes as atoms move between domains, while the captured graph
needs a fixed `N`. The pair style refuses the combination while loading the
model rather than at the first step. Re-export with `cuda_graph=False` (meta
`execution: "dynamic"`) for periodic / multi-rank MD.

### Virial (`supports_virial=true`)

If the AOT package returns a **fourth** tensor of shape `(6,)`
`(xx, yy, zz, xy, xz, yz)` in the model energy unit — index 3, behind
`atomic_energy`, since v3 — it is converted and accumulated into the global
virial. Otherwise LAMMPS falls back to `virial_fdotr_compute` when fdotr virial
is requested. That virial must be the virial of the **local** energy, the same
quantity `local_mask` selects; a virial computed over local + ghost sites would
over-count exactly as an unpartitioned energy would.

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
