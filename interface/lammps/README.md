# `pair_style molnex` — one LAMMPS pair style for every deployed molnex model

Classical Class-I / OPLS parameter tables do **not** go through this pair
style. Compile `ForceField.to_ir()` with `LammpsAdapter` and write
`system.data` / `system.in.settings` via `LammpsWriter` — see
[Force-field export](../../docs/molix/user-guide/ff-export.md). This
document is the neural-potential path only.

## Layers

The C++ side of deployment is two layers with a one-way dependency:

| layer | where | knows |
|---|---|---|
| `molnex_runtime` | `interface/runtime/` | the model: `Metadata` (`{name}.json`), `ModelArtifact::open` (format → loader; `AotiArtifact` for `aoti`), `OpLibraries` (native operator libraries), `AtomEnergyExec` (the calling convention). LibTorch only — no engine, no Python. |
| LAMMPS adapter | `interface/lammps/` | LAMMPS: `bridge/` (atom arrays, neighbour list, forces, reverse comm), `impls/` (`CutoffGraphImpl`), and the `pair_molnex` facade, the only file that includes a LAMMPS header. |

The runtime never references LAMMPS; another engine links `molnex_runtime` and
writes its own adapter. One generic pair style evaluates any model deployed
with `molix.deploy` — there is no per-model C++.

```
pair_style molnex <model_dir> [name] [mode fold|ghost]
pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN>
```

* `<model_dir>` — holds `<name>.json` and the package (`<name>.pt2` for the
  `aoti` format).
* `[name]` — basename inside `<model_dir>` (default `model`).
* `[mode]` — how periodic neighbours reach the model (default `fold` on one
  rank, `ghost` on several; see [Two modes](#two-modes-fold-and-ghost)).
* `pair_coeff` maps each LAMMPS atom type to an atomic number `Z`. Every Z is
  checked against the metadata's `species` — an element the model was not
  built for is refused.

## The calling convention

`molix.deploy.AtomEnergyModel` is a structural `Protocol`; a module satisfies it
when `forward` has this signature. Nothing subclasses it.

```text
Z (N,) int64, pos (N, 3), pair_index (E, 2) int64, shifts (E, 3), local_mask (N,) bool
    -> energy (), forces (N, 3), atom_energy (N,), virial (3, 3)
```

| # | tensor | shape | dtype |
|---|--------|-------|-------|
| in 0 | `Z` | `(N,)` | `int64` |
| in 1 | `pos` | `(N, 3)` | model dtype, in the model's length unit |
| in 2 | `pair_index` | `(E, 2)` | `int64`. `[:, 0]` is the source, `[:, 1]` the target |
| in 3 | `shifts` | `(E, 3)` | model dtype. Lattice translation of the target image: the edge is `pos[t] - pos[s] + shifts` |
| in 4 | `local_mask` | `(N,)` | `bool`. True for atoms this rank owns |
| out 0 | `energy` | `()` | model dtype. Equals `(atom_energy * local_mask).sum()` |
| out 1 | `forces` | `(N, 3)` | model dtype. Equals `-d(energy)/d(pos)`, ghosts included |
| out 2 | `atom_energy` | `(N,)` | model dtype |
| out 3 | `virial` | `(3, 3)` | model dtype. `+d(energy)/d(strain)`, symmetric — molnex's `graphs.virial` sign |

The runtime refuses any other output count (`AtomEnergyExec`). `N` and `E` are
dynamic (`molix.deploy.contract.DYNAMIC_SHAPES`), so one package serves every
frame. Edges are **not** sorted; a model that needs `(source, target)` order
sorts inside its own `forward`.

## Two modes: `fold` and `ghost`

The same package serves both; they differ in what the pair style stages.

| | `fold` (default on one rank) | `ghost` (default on several ranks) |
|---|---|---|
| graph nodes | the `nlocal` owned atoms | owned atoms + every ghost within `T * cutoff + skin` |
| a periodic neighbour | folded onto the atom it copies, `shift = x[image] - x[owner]` | a node of its own, `shift = 0` |
| `local_mask` | all true | true for the owned prefix |
| forces written | owned atoms | owned and ghost rows; ghosts go home by the integrator's reverse comm |
| global virial | the model's `virial`, negated to LAMMPS' `-dE/deps` and converted like the energy | `virial_fdotr_compute()` |
| ghost shell needed | `cutoff + skin` (LAMMPS' default) | `T * cutoff + skin` (`comm_modify cutoff`) |
| `newton` | not required | `newton on` required |

**Why fold.** On one GPU the ghost mode evaluates the model on the whole ghost
shell every step. For a 35 Å PEO box (4600 atoms) at `T * cutoff + skin = 14 Å`
that is about 5-6 times the atoms: MACE-OFF24-M measured 927 ms/step against
179 ms in molnex's own periodic path, and Allegro-OAM-L / MACE-OMol-0 ran out
of memory. The folded graph is exactly the periodic graph molnex itself
evaluates in Python (`molix.md.NeighborList` + shifts), so nothing is computed
twice.

**How the fold is built** (`GraphBridge::build_folded`). The pair style walks
the full neighbour list of the owned centres only (no `REQ_GHOST`). Each
neighbour `j` within the cutoff becomes the pair `(i, owner(j))` with
`shift = x[j] - x[owner(j)]`. `owner(j)` is the owned atom carrying `j`'s atom
ID, read off a per-step ID table of the owned atoms (`atom_modify map` is not
needed; LAMMPS' own map would agree — it gives owned atoms precedence over
images). Because the shift is the difference of two LAMMPS coordinates it is
exactly the lattice translation LAMMPS applied to make the image, for
orthogonal and triclinic boxes alike. In a box under twice the cutoff an atom
can be its own neighbour's original: that is the pair `(i, i)` with a non-zero
shift, which is the periodic self-interaction the model must see.

The fold requires every ghost to be an image of an owned atom, which holds on
one rank and not on several, so `mode fold` with more than one rank is refused
at `init`; a ghost whose ID no owned atom carries is refused at `compute`. It
also needs atom IDs (`atom_modify id no` is refused).

A long-range term (MACE-POLAR's electrostatics, MACE-LES's latent Ewald sum) is
not a sum of atom energies, so those models cannot be deployed through this
convention: `AtomEnergyAdapter.check` refuses them before export.

## The metadata

`<name>.json` is `molix.deploy.DeployMetadata`, read by
`molnex::runtime::Metadata`. It is engine-neutral: the model's own units, not a
LAMMPS unit table.

```json
{
  "format": "aoti",
  "cutoff": 5.0,
  "species": [1, 6, 7, 8],
  "dtype": "float64",
  "receptive_field_layers": 5,
  "energy_unit": "eV",
  "length_unit": "angstrom",
  "op_libraries": {}
}
```

A missing or mistyped field is refused — the message names it; a field this
reader does not know is ignored. The device is read from the package itself
(`AOTI_DEVICE_KEY`), not from this file.

`op_libraries` names the native libraries that register the package's
non-ATen operators (a cuEquivariance model needs `molnex_cueq_ops`), each with
the upstream requirement its binding was built against. `ModelArtifact::open`
loads every one before the package: `lib<name>.so` from the directories of
`MOLNEX_OP_LIBRARY_PATH` (colon-separated) or, when that is unset, the dynamic
loader's own search. A library that is missing, or that reports another
requirement through `molnex_op_library_requirement()`, is refused.

## Deploying a model

Three primitives, composed by the caller (see `molix.deploy`):

```python
from molix.deploy import DeployMetadata, AtomEnergyAdapter, example_inputs
from molix.deploy.contract import DYNAMIC_SHAPES
from molix.export import aoti

model = AtomEnergyAdapter(potential).cuda()
inputs = example_inputs([1, 6, 8], 6.0, torch.float32, "cuda")
model.check(*inputs[:4])                       # atom-energy partition + force checks
written = aoti.package(model, inputs, "out/model.pt2", dynamic_shapes=DYNAMIC_SHAPES)
DeployMetadata(
    format="aoti", cutoff=6.0, species=(1, 6, 8), dtype="float32",
    receptive_field_layers=2,                  # T — required, no default
    op_libraries=DeployMetadata.op_libraries_for(written.op_namespaces),
).write("out")
```

A module from another library subclasses nothing: its `forward` takes the five
inputs and returns the four outputs, and `aoti.package` + `DeployMetadata`
deploy it the same way.

`receptive_field_layers` is `T`, the number of message-passing layers, and it
has **no default**. An atom's energy depends on everything within `T * cutoff`.

`AtomEnergyAdapter.check` runs an example system eagerly before export: the
potential's energy must equal the sum of its atom energies (rel 1e-6) and,
when the potential emits forces, those forces must match
`-d(sum of atom energies)/d(pos)` (rel 1e-5).

## Units

The metadata states the model's own units (`energy_unit`, `length_unit`). The
pair style maps the energy unit onto the LAMMPS unit style that uses it — `eV`
is `metal`, `kcal/mol` is `real` — and converts into the active style with
LAMMPS' own table, `utils::get_conversion_factor(utils::ENERGY, METAL2REAL /
REAL2METAL)`. **No conversion constant is written in molnex**, Python or C++.
An energy unit with no LAMMPS style, or a run in a style LAMMPS cannot convert
to (anything but `real` / `metal`), is refused at `pair_style`.

Forces reuse the energy factor, and that reuse is *checked*: `[F] = [E]/[L]`,
so it holds only while the model measures length the way LAMMPS does.
`CutoffGraphImpl` refuses a `length_unit` other than `angstrom` (LAMMPS `real`
and `metal` both use Å), so a Bohr model fails loudly instead of being
mis-scaled by the Bohr/Å ratio.

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

## Per-atom energy is supported; per-atom stress is refused

`compute pe/atom` reads `atom_energy (N,)`. The pair style keeps the entries
for the atoms this rank owns and tallies them into LAMMPS' `eatom`. The total
energy is the sum of those local entries, re-summed from the same tensor, so
`compute reduce` over `pe/atom` agrees with `pe`.

`compute stress/atom` errors out, along with anything that pulls it in (a
per-atom dump, `fix ave/atom` over it), in both modes. The model returns no
per-atom virial. LAMMPS' `ev_init` allocates `vatom`, and a pair style that
never writes it would hand back a legal-looking all-zero field. The refusal
names the compute and does not mention a file version.

Global pressure: in the `ghost` mode `virial_fdotr_compute()` when LAMMPS asks
for the force-times-distance form (`vflag_fdotr`). In the `fold` mode the
model's `virial` output (`+dE/deps`), negated into LAMMPS' `-dE/deps`,
converted with the energy factor and tallied as `xx yy zz xy xz yz`; `fdotr` is
disabled for that mode (`no_virial_fdotr_compute`), because with the images
folded onto their owners it would miss every periodic contribution.

## Runbook — parity evidence, and when it expires

Some correctness claims here cannot be proven by any test that runs without a
LAMMPS build. They are proven by a **recorded run** instead, and a recorded run
has an owner and an expiry — otherwise the evidence quietly becomes a memory of
a machine nobody has any more.

### Periodic multi-rank parity

| | |
|---|---|
| **claim** | one rank, two ranks and a direct Python evaluation of the same periodic configuration agree on total energy and forces within rel `1e-6` |
| **proves** | the local-only energy partition (no ghost atom energy counted twice) **and** the single reduction path (no ghost force communicated twice). Single-rank alone cannot show the second — only a decomposed run has an inter-rank reduction to double. It is also the full-scale edge-order-invariance check: C++ builds edges in raw neighbour-list order while the Python reference builds them sorted, and they agree. |
| **tool** | `scripts/lammps_iface_parity.py` — **tool removed from the tree** (F5 / `lammps-iface-post-chain`): it drives LAMMPS from `scripts/`, which Decision 4 confines to `interface/`, and it has not yet been moved. Until it is, this row's evidence cannot be regenerated. |
| **owner** | the implementer of the `lammps-iface` chain |
| **re-run trigger** | **any change under `interface/`, `src/molix/deploy/` or `src/molix/export/`** |
| **status** | **RECORDED 2026-08-30** — all six comparisons OK at ≤1.6e-15 (gate rel 1e-6); provenance below. The tool exits non-zero without a LAMMPS build; it never skips. |

The block below is the export API of 2026-08-30, kept as the provenance of
that run. That API (`molix.engine.export_for_lammps`, a LAMMPS unit table in the
JSON) has since been replaced by `molix.deploy` + `molix.export.aoti`, and the
calling convention by the atom-energy one (shifts + virial) with the `fold` / `ghost`
modes; the run must be re-recorded against the current layers, now as four
comparisons — Python, one rank `fold`, one rank `ghost`, two ranks `ghost` —
plus the `fold` virial against Python's `graphs.virial`.

Recorded run (SLURM job 1805410, NAISS Arrhenius GH200 node):

```
system          : water box, 27 molecules, 81 atoms
atom_style      : atomic (molecular == 0: no bonds, so no special_bonds pairs)
boundary        : p p p                    box: 9.318 A cubic
ranks           : 1 and 2                  mpirun: OpenMPI 5.0.8 -np 2
comm_modify     : cutoff 10.000000 (T 2 * cutoff 4.000 + skin 2)
receptive_field : 2                        ghost_atoms: 2160 (Python reference)
lammps          : 22 Jul 2025 source build (-DBUILD_MPI=on -DBUILD_SHARED_LIBS=on -DPKG_PLUGIN=on)
plugin_commit   : a6b0a03143e45db195a31278980332c367246ccd-dirty
torch_version   : 2.13.0                   device: cuda   precision: float64
model           : random-weight PiNet (seed 2666, hidden 16, depth 2), energy-only export,
                  supports_pbc=true, units real, contract cutoff-graph, execution dynamic
results         : E python/1rank/2ranks = -15.770846444542112 / same / -15.77084644454211
                  worst E rel 1.126e-16; worst F rel 1.588e-15 (gate rel 1e-6)
capture_command : scripts/lammps_iface_parity.py --case periodic
                  --model-dir <ws>/exports/pinet_pbc --lammps <ws>/build-lammps/lmp
                  --plugin <ws>/build-plugin/aarch64/molnexplugin.so
                  --mpirun 'mpirun -np {ranks}' --ranks 2
workspace       : /nobackup/proj/disk/teoroo/personal/jicli594/work/lammps-gh200 (<ws>);
                  full log logs/lammps-parity-1805410.out
date            : 2026-08-30    tolerance: rel 1e-06 (energy |dE|/|E|; forces max_i|df_i|/max_j|f_j|)
```

The aspirin goldens (`ac-013`) were captured and re-verified in the same job —
energy rel 0.0, forces 8.707e-16 — and live as the `Goldens(...)` block inside
`scripts/lammps_iface_parity.py`, pinned by deck and model hashes with their
own embedded provenance.

The tool's Provenance block records the LAMMPS version, rank count,
`atom_style`, `comm_modify` settings, plugin commit, torch version,
device/precision, the capture command and the date. A run whose provenance no
longer matches the tree is expired evidence, not evidence.

### Classical periodic parity

| | |
|---|---|
| **claim** | a Class-I force field **exported** by `molix.ff_export` means the same energy and forces once LAMMPS, not molnex, evaluates it — bonded terms, switched Lennard-Jones, Ewald-split Coulomb under a lattice, and 1-2 / 1-3 / 1-4 exclusions on both classes |
| **proves** | the export *conventions*: `½k` → `K`, radians → degrees, the quintic taper vs `lj/mdf`, the model's declared combining rule on every `pair_coeff i j` (`MixingRule`; Lorentz–Berthelot for the bare `ForceField` both cases build), `special_bonds` carrying `NonbondedScaling`, and excluded pairs subtracted against the reciprocal sum rather than attenuated in real space. Nothing here involves `pair_style molnex`: no plugin, no exported model. Both sides are converged first and the report says so, so a residual cannot be mesh error wearing a convention's clothes. |
| **tool** | `interface/tests/parity_classical.py` (ctest `parity_classical_butane`, `parity_classical_periodic`; exit 77 = no `lmp`) |
| **owner** | the implementer of the `composable-a0` chain |
| **re-run trigger** | **any change under `src/molix/ff_export/` or `src/molpot/forcefield/`** |
| **status** | **RECORDED 2026-09-08 (condensed types)** — butane relative energy 2.583e-12, worst force component 3.506e-12; periodic relative energy 9.912e-09, worst force component 6.876e-06 kcal/mol/Å (gates 1e-6 and 1e-5). Re-run unchanged after `ForceField.assign_types` replaced identity typing (F25): the periodic deck went from 545 to 116 lines — data 160 → 94, bonded include 29 → 7, pair block 356 → 15 — with **every number identical**, which is what a pure type-condensation must do. The deck is consumed exactly as `LammpsWriter` writes it — the tool patches nothing. Provenance below. |

Recorded run (local x86_64 workstation, `/home/jicli594/.local/x86_64/bin/lmp`):

```
lammps          : LAMMPS (30 Mar 2026 - Development - patch_30Mar2026-1074-g7f680de296)
                  packages EXTRA-PAIR (lj/mdf) + KSPACE + MOLECULE
torch           : 2.14.0+cpu    platform: Linux-5.14.0-687.29.1.el9_8.x86_64
butane          : C4 chain, bonded-only OPLS form (bond/angle/OPLS torsion),
                  boundary f f f, deck-local pair_style zero
                  E molnex = 0.1309516320367863, lammps = 0.130951632036448
                  rel 2.583e-12
                  max|dF| 3.506e-12; 200-step NVE both sides, |dPE_final| 2.583e-10
                  molnex 0.42 ms/step, lammps 1.01 ms/step
periodic        : 8 TIP3P-geometry waters + Na+ + Cl-, 26 atoms, boundary p p p,
                  cubic L = 12.0 A, seed 20260908, 352 placed edges
                  lj/cut switch 4.5-5.5 A; coulomb smearing 1.0 A, real-space
                  cutoff 5.5 A; special_bonds lj 0 0 0.5 coul 0 0 0.5
                  deck: pair_style hybrid/overlay lj/mdf 4.5 5.5 coul/long 5.5
                        + kspace_style pppm 1e-10 + kspace_modify gewald 0.7071068
                  types: 4 atom (O, H, Na+, Cl-), 1 bond, 1 angle - condensed by
                        parameter equality, so 10 numeric pair_coeff lines
                        (4*5/2) + `pair_coeff * * coul/long`, one bond_coeff and
                        one angle_coeff. Identity typing wrote 351 / 16 / 8.
                  files: system.data 94 lines, system.in.settings 7,
                        system.in.pair 15 (identity: 160 / 29 / 356)
                  E molnex = -47.23502966626038, lammps = -47.235029198050
                  per term (molnex - lammps): bond -2.309e-14, angle -1.127e-11,
                        vdW -3.309e-08, coulomb -4.351e-07
                  rel 9.912e-09; max|dF| 6.876e-06 kcal/mol/A
                  the vdW residual is the .7g rendering of the 10 mixed
                  eps_ij / sigma_ij pair_coeff values, nothing else - identical
                  to the identity-typed run, since condensing repeats no value
deck statement  : every pair_style / pair_coeff / kspace_style / kspace_modify /
                  special_bonds / bonded line is INCLUDED from what LammpsWriter
                  wrote, unmodified. The deck adds only `pair_modify table 16`
                  and the output-format lines - numerics and printing, no model.
convergence     : splitting a = 1/(sigma*sqrt2) = 0.7071068 1/A, carried by the
                  EXPORTED `kspace_modify gewald` line so LAMMPS solves molnex's
                  decomposition instead of picking its own a from the requested
                  accuracy; the erfc(a*r_c) = 3.8e-08 real-space residue then
                  cancels instead of appearing in the parity;
                  molnex side is the converged ewald k-vector sum (lambda_min
                  0.25 A, unchanged when halved), because the pme mesh does not
                  reach 1e-8 at any affordable spacing in this box:
                    mesh 0.25 A   |dE|/|E| 7.3e-05  max|dF| 1.1e-02
                    mesh 0.125 A  |dE|/|E| 4.2e-06  max|dF| 1.4e-03
                    mesh 0.0625 A |dE|/|E| 2.5e-07  max|dF| 1.7e-04   (~7 s)
                    mesh 0.03125A |dE|/|E| 1.7e-08  max|dF| 2.0e-05   (~60 s)
                  the pme evaluation still runs every time and is reported as the
                  mesh diagnostic it is. LAMMPS side: pppm accuracy 1e-10 (grid
                  192^3) and `pair_modify table 16` — measured against an
                  independent analytic real-space sum, the default 12-bit erfc
                  table leaves ~1e-6 kcal/mol in ecoul and `pair_modify table 0`
                  (the Abramowitz-Stegun polynomial LAMMPS' own warning
                  recommends) is an order of magnitude WORSE.
floor           : k_e differs between the two codes by 9.9e-09 relative
                  (molnex 332.0637132991921, LAMMPS qqrd2e 332.06371 measured),
                  which is the hard floor of any Coulomb comparison here.
goldens         : the molnex side of both cases is pinned at rel 1e-10 inside the
                  tool and checked on every run, so a molnex regression is caught
                  even when it is the LAMMPS binary that moved.
capture_command : PYTHONPATH=src python interface/tests/parity_classical.py \
                      --case periodic --lmp <lmp> --capture
date            : 2026-09-08
```

**What this run fixed in the exporter.** The first recorded attempt could not
be read by LAMMPS at all, and the four defects it exposed are fixed in the
library rather than worked around in the tool — the deck above is consumed
byte-for-byte as written:

1. `LammpsWriter.write_pair_block` now emits an explicit `pair_coeff i j` for
   every `i <= j`, mixed by the rule **the model declares** — the `mixing`
   option of its `pair:lj/cut` style, carried on `NonbondedMethodRecord` and
   applied through `molpot.potentials.vdw.lj126.MixingRule.mix`, the same call
   `Potentials` evaluates with. That is `"lorentz-berthelot"` (σ arithmetic, ε
   geometric) for a bare `ForceField` — what both parity cases here use — and
   `"geometric"` (both, the Jorgensen 1996 OPLS-AA rule) for
   `OPLSAAForceField` since 2026-09-08; before that date molnex mixed
   Lorentz–Berthelot regardless of template and said so nowhere (ledger finding
   **F24**). Diagonal lines alone were **silently zero** for every cross-type LJ
   pair under `hybrid/overlay` (its `pair_coeff * * coul/long` maps every type
   pair to the Coulomb sub-style, and LAMMPS never mixes an LJ sub-style onto a
   pair it was not explicitly assigned), and LAMMPS's own default (`pair_modify
   mix geometric`) is a third rule again — so the deck states every mixed pair
   rather than letting either side choose. Cost: `T(T+1)/2` lines for `T`
   distinct atom types — `N(N+1)/2` while types were identity types, which is
   what ledger debt **F25** recorded and what condensing (item 2) closed, not a
   return to LAMMPS-side mixing.
2. Types are **condensed on parameter equality** by
   `ForceField.assign_types`, the one owner of the mapping: `realize` registers
   a molpy type per class from it and every `LammpsWriter` call labels from the
   same object, so the data file's labelmap and the include's coeff names cannot
   drift apart. Two atoms with the same (mass, charge, ε, σ) are one atom type
   and two interactions between the same atom types with the same constants are
   one bonded type — 26 water/ion atoms are 4 types, not 26. Until 2026-09-08
   molnex exported *identity* types (one per atom), which was correct and
   quadratic. Labels are `t1`/`b1`/`a1`/`d1`/`i1`-style: a LAMMPS type label may
   not begin with a digit, so the original `1-2` / `1-2-3` names stopped
   `read_data` at `ERROR: Invalid type label 1-2` and no exported system
   carrying bonds could be read. They are **zero-padded to the type count**,
   because molrs numbers a data file's types by sorting their labels as *text*:
   unpadded `t1 … t26` makes LAMMPS type 2 be the tenth type and the numeric
   `pair_coeff` lines address the wrong ε/σ — measured `E_vdwl = -0.2251`
   against molnex's `-0.8163`, no warning. Invisible below ten types, which is
   why only a real box caught it.
3. `LammpsWriter.write_settings` passes `skip_units=True`; the include is sourced
   after `read_data`, where LAMMPS refuses a `units` command.
4. `ForceField.nonbonded_scaling` mirrors into molpy's `set_special_bonds`, and
   `LammpsWriter.to_frame` writes the molecule column as `mol_id` (the spelling
   molrs reads to infer `atom_style full`). Without the first, an exported
   bonded include carried `special_bonds … 1.0` — every 1–4 pair at full
   strength; without the second, the data file came out `Atoms # charge` and
   LAMMPS refused it under the `atom_style full` the include declares.

## Building the plugin

`pair_style molnex` is a **runtime plugin** (`plugin load`), never compiled into
LAMMPS. It is arch+ABI specific, so build it once **from source** against the
exact LAMMPS it will be loaded into — link `liblammps.so` from a
`-DPKG_PLUGIN=on -DBUILD_SHARED_LIBS=on` **source build**, not a binary wheel.
(A wheel that ships `PKG_PLUGIN` can still *load* the plugin at runtime, but the
`.so` itself must be compiled against matching headers/lib — there is no
universal binary.) Requires LibTorch from the molnex venv and a C++20
compiler. `molnex_runtime` (CPU or CUDA, detected from the toolkit) and the
adapter's `bridge/` / `impls/` layers are added from their own directories.

The LAMMPS-free part of the suite — runtime and adapter layers, no LAMMPS build
needed — runs from `interface/`:

```bash
cmake -S interface -B build/interface-$(uname -m) -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DPython_EXECUTABLE="$PY"
cmake --build build/interface-$(uname -m) -j
ctest --test-dir build/interface-$(uname -m) -E '^parity_classical_'
```

Every build is out of source under the repository's `build/`, one directory
per architecture — `build/interface-<arch>/` (this suite), `build/plugin-<arch>/`
(the plugin below), `build/csrc-<arch>/` (the native op libraries) — so x86_64
and aarch64 builds coexist and the source tree holds no artefacts.

```bash
source scripts/env/activate.sh
TORCH_CMAKE=$("$PY" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share/cmake'))")
LMP_SRC=/path/to/lammps-src/src                 # source tree headers (pair.h, lammpsplugin.h, version.h)
LMP_LIB=/path/to/your/source-build/liblammps.so # YOUR build, not the wheel

cmake -S interface/lammps -B build/plugin-$(uname -m) \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DLAMMPS_HEADER_DIR="$LMP_SRC" \
  -DLAMMPS_LIB="$LMP_LIB" \
  -DLAMMPS_MPI_INCLUDE="$LMP_SRC/STUBS"         # serial LAMMPS (BUILD_MPI=off)
cmake --build build/plugin-$(uname -m) -j        # → build/plugin-<arch>/molnexplugin.so

# at runtime
export LD_LIBRARY_PATH="$(dirname $LMP_LIB):$LD_LIBRARY_PATH"
# op libraries the metadata lists: build/csrc-<arch>/{molix,cueq} (activate.sh sets this)
export MOLNEX_OP_LIBRARY_PATH=/path/to/build/csrc-<arch>/molix:/path/to/build/csrc-<arch>/cueq
# in the LAMMPS input script:
#   plugin load build/plugin-<arch>/molnexplugin.so
#   pair_style molnex pinet_aspirin
#   pair_coeff * * 1 6 7 8
```

A complete from-source build + benchmark workspace (LAMMPS + plugin, arch-tagged,
no wheel — `build_lammps.sh` + `build_plugin.sh`) is kept **outside this
repository** as developer scratch and is not shipped with molnex; the recipe
above is the supported path.

## The `ghost` mode

Ghosts are requested (`Neighbor` full list plus ghost centres). `newton
on` is required; it is LAMMPS' default. Ghost forces are the derivative of
local atom energies with respect to a ghost's coordinates, and the only thing
that returns them to the owning atom is `Verlet::run`'s global reverse
communication, which LAMMPS performs only when `force->newton` is set. With
newton off those cross-terms are discarded, so `init` refuses the script. The
pair style issues no reverse communication of its own: doing both would send
the same contribution home twice. When a rank has no ghosts, `local_mask` is
all-true and that reverse communication is a no-op.

`pair_index` is the neighbour list's own order. The pair style does not sort.

### `comm_modify cutoff` (ghost mode)

A `T`-layer network's atom energy depends on every atom within `T * cutoff`,
and the pair style evaluates the model in one shot, so each rank must hold a
copy of all of them. LAMMPS' default ghost cutoff is `cutoff + skin`, which
meets the check only at `T == 1`. When the user has not set a ghost cutoff,
LAMMPS reports `0` and the pair style treats that as `cutoff + skin`.

```
# T = 5, cutoff = 5.0 Å, default skin for real/metal units = 2.0 Å
comm_modify cutoff 27.0        # T * cutoff + skin
```

The rule, checked at `init` and never downgraded to a warning, is

```
ghost cutoff  >=  receptive_field_layers * cutoff + neighbor skin
```

The diagnostic names `T`, the effective cutoff, the required cutoff, and the
`comm_modify cutoff` line to add. A wider shell costs memory and communication
that grow with `T * cutoff`.

On a single non-periodic rank there are no ghosts to widen, `local_mask` is
all-true, and the same check still runs against whatever ghost cutoff LAMMPS
reports.

The `fold` mode has no such condition: only the owned atoms are nodes, and a
neighbour within the cutoff is all it reads, which LAMMPS' default ghost
cutoff (`cutoff + skin`) always supplies.

## Weight reload

`AotiArtifact::update_weights` calls `AOTIModelPackageLoader::load_constants`
with `use_inactive=false`. That rewrites the **active** constant buffer. A
`shared_mutex` is held exclusive across the rewrite and shared across `run`:
the AOTI container does not keep an in-flight `run` off the buffer being
rewritten, and without the lock the overlap is an intermittent SIGSEGV.
`load_constants` translates state-dict fully-qualified names itself. The
package must have been exported with
`aot_inductor.use_runtime_constant_folding` (an `inductor_configs` entry of
`molix.export.aoti.package`).

`pair_style molnex` opens the package with `num_models=1`. There is no
pair-style keyword for a second buffer. `num_models > 1` is accepted by
`AotiArtifact` only together with `run_single_threaded=false` (torch >= 2.12
rejects the other combination). That flag is not a CUDA-graph switch; this
pair style does not capture a CUDA graph.
