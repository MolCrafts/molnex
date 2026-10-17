I found about 60 real violations across the four repos. The example you gave is already fixed: `LJCut` is not in `molrs::md` (Rust) or `molrs.md` (Python). `md/mod.rs:43` re-exports nothing from ff or core, and `molrs.md.__all__` is just `MD, Langevin, MDState, MaxwellBoltzmann, VelocityVerlet`. Only stale doc lines still say md owns LJ (row R16).

Nothing was edited, built or committed. The molrs Rust crate I audited myself. The bindings, molpy and molpack were each audited by a separate read-only agent, and I re-checked their key evidence lines. The molrs audit covers the `ir/p-registry` checkout only, not the uncommitted md/ff work in the `dev` checkout (`molrs/`).

**Module jobs (molrs crate)**
- `op`: numeric base (vec3, linalg, rigid motions, superpose, SO(3) sampling).
- `core`: data model (store, system), spatial (box, regions, neighbour lists, periodic images), math, units.
- `perceive`: chemical facts derived from the graph (rings, aromaticity, bond types, stereo, SMARTS matching).
- `io`: file formats into and out of Frame or Trajectory, plus SMILES.
- `ff`: force-field IR, ForceField, potentials (kernels), typifiers, charge models, parameter tables, force-field file readers/writers.
- `optimize`: the optimizer trait and L-BFGS.
- `md`: integrators and force providers.
- `conformer`: ETKDG conformer generation.
- `compute`: trajectory analysis.
- `builder`: structure builders.
- `signal`: FFT, ACF, windows.
- `stream`: WebSocket and MessagePack streaming.

## 1. molrs Rust crate (`molrs-p-registry/molrs/src`)

**P1: wrong owner or a duplicate implementation**

| # | path:line | symbol / evidence | current | why | correct owner | fix (public path change) |
|---|---|---|---|---|---|---|
| R1 | `md/pairs.rs:1377`; `ff/potential/mod.rs:634` (doc: "Feed it to `SpecialWeights::new`(crate::md::SpecialWeights::new)"); `molrs-python/src/ff/mod.rs:1863` | `SpecialWeights`: "The weights a force field puts on close non-bonded neighbours" | md | Force-field special_bonds logic lives in md. The ff binding has to construct an md type. | `ff::potential`, next to `PairWeights` | Move it; `PairWeights::special_weights()` returns `SpecialWeights` directly. `molrs::md::SpecialWeights` → `molrs::ff::potential::SpecialWeights` |
| R2 | `compute/util.rs:29` `pub enum MicHelper { Free, Ortho{len,inv_len,pbc}, Triclinic }` (used in 15 files) | minimum-image helper | `compute::util` (public) | Copy of `core::spatial::simbox::Mic` (`simbox.rs:1142`, same three variants). `util` is also a grab-bag module name. | `spatial::simbox::Mic` | Delete `MicHelper` and use `SimBox::mic()`. Move `get_positions_ref` next to `FrameAccess` or make it `pub(crate)`. `molrs::compute::util` removed |
| R3 | `compute/dynamics/persist.rs:88` `mic_dist2`; `compute/voronoi/integrate.rs:394` `min_image`; `optimize/soft.rs:200` `harm_term` (`l * (d[a]/l).round()`) | hand-rolled orthorhombic minimum image | compute, optimize | More copies of `Mic::Ortho` | `spatial::simbox::Mic` | Replace with `Mic::apply`. No path change |
| R4 | `ff/potential/cmap/charmm.rs:266-290` private `sub`, `cross`, `dot` | vector math | ff cmap kernel (new in #107) | Copies of `op::vec3` and `ff::potential::geometry::sub3` | `op::vec3` | Import those instead. No path change |
| R5 | `md/maxwell.rs:101` and `md/integrators.rs:487` `fn standard_normal` | Gaussian draw | md, twice | The same numerical helper exists twice | `op` (or `rand_distr::StandardNormal`) | Keep one. No path change |
| R6 | `conformer/etkdg/etmin.rs:519` `pub fn minimize` ("steepest descent with adaptive step"), used at `etkdg/mod.rs:354,364,381`, while `etkdg/mod.rs:444` already calls `crate::optimize::minimize_lbfgs_rms` | minimizer | conformer | A second public minimizer outside `optimize` | `optimize` | Use `minimize_lbfgs_rms` (check RDKit parity). `molrs::conformer::etkdg::etmin::minimize` removed |
| R7 | `conformer/distgeom/uff.rs:34,73,121` (`UffParams{r1,xi}` table, `atom_label`, `rest_length`) | UFF parameters, typing and rest-length formula | conformer | Duplicates `ff::params::uff` and `ff::typifier::uff::bond_rest_and_k` (`typifier/uff/mod.rs:683`). The `conformer` feature already enables `ff`. | `ff::typifier::uff` / `ff::params::uff` | Expose a `pub(crate)` rest-length function from `ff::typifier::uff` and delete `distgeom/uff.rs`. `molrs::conformer::distgeom::uff::*` removed |
| R8 | `ff/mmff/hybrid.rs:15` `pub enum Hyb`; `ff/typifier/uff/mod.rs:508` `enum Hyb`; `conformer/distgeom/mol_features.rs:23` `pub enum Hybridization` | hybridization perception | ff, conformer | Three implementations; `perceive` has none | `perceive` (new `hybridization.rs`) | Single `perceive::Hybridization`. `molrs::ff::mmff::hybrid::Hyb` and `conformer::distgeom::mol_features::Hybridization` removed |
| R9 | `ff/mmff/` (`mod.rs`: "the front end of the typifier, and nothing else"; `aromaticity`, `atomtype`, `charges`, `resolve`, `topo`, `da`, `hybrid`) beside `ff/typifier/mmff/` | MMFF typing | `ff::mmff` | Typifier logic sits outside `typifier`: one typifier split across two modules | `ff::typifier::mmff` | Move `ff/mmff/*` under `typifier/mmff/` (private). `molrs::ff::mmff::{MmffVariant, MmffMolProperties, da::*, hybrid::*, topo::*}` removed or made private |
| R10 (superseded by ruling 11: the typing-metadata XML readers are `io::forcefield::xml`'s) | `ff/forcefield/xml.rs:137` `read_mmff_params_xml_str`, `:190` `read_opls_typing_xml_str`; imports `:40-41` `use crate::ff::typifier::mmff::{MMFFAtomProp, MMFFParams}; use crate::ff::typifier::opls::{OplsTypeRow, OplsTypingMeta};` | typifier table readers | `ff::forcefield::xml` | Typifier logic in forcefield. The only callers are `typifier/mmff/engine.rs:72` and `typifier/opls/mod.rs:103` | `ff::typifier::{mmff, opls}` | Move both readers into those typifiers. `molrs::ff::forcefield::xml::read_{mmff_params,opls_typing}_xml_str` → `ff::typifier::{mmff,opls}::…` (or `pub(crate)`) |
| R11 | `ff/typifier/am1bcc.rs:41,182,514` `BccParameterSet`, `BCCCorrectionTable`, `BCCCorrector::apply` (writes `charge`); `ff/typifier/mod.rs:26` re-exports them; `ff/charge/mod.rs:90` re-exports `BccParameterSet` | BCC charge correction | `ff::typifier` | Charge-model logic in typifier. There are two public ways to apply BCC: `BCCCorrector::apply` and `BccModel::correct` | `ff::charge` (bcc.rs) | Move the table, parameter set and corrector into `ff::charge`; delete `BCCCorrector` or make it private. Keep only `BCCAtomChargeTypifier` in typifier. `ff::typifier::{BccParameterSet, BCCCorrectionTable, BCCCorrector}` → `ff::charge::…`; `BCCCorrector` removed |
| R12 | `perceive/equivalence.rs:353` `pub fn average_charges` (sole user: `ff/charge/model.rs:19,138`) | class-mean charge averaging | perceive | A charge-model step inside perception. The module's own doc says "Perception ends at the classes." | `ff::charge` | Move it (`pub(crate)`). `molrs::perceive::equivalence::average_charges` removed |
| R13 | `io/smiles/mod.rs:154` `pub use parser::parse_smarts;` and `perceive/smarts/parser.rs` (doc at `io/smiles/mod.rs:18-21`: "that engine has its own parser and never consumes this one") | SMARTS grammar | io and perceive | Two parsers for one language | One parser: `io::smiles` parses, `perceive::smarts` compiles from `SmilesIR` | Make the matcher consume `SmilesIR`; delete `perceive/smarts/parser.rs` |
| R14 | `io/data/top.rs:240,247,480,488` `read_top`, `read_top_frame`, `write_top`, `TopReader`, `TopFrameWriter` vs `ff/forcefield/readers/gromacs/mod.rs:370,379` `read_system[_str]` and `ff/forcefield/writers/gromacs.rs:922` `write_system_str` | GROMACS `.top` molecules | io and ff | Two readers and two writers for one format, with opposite index bases (1-based vs 0-based) and different `#include` handling | ff GROMACS system reader/writer | Delete `io::data::top`. `molrs::io::data::top::*` removed |
| R15 | `io/data/prmtop_tables.rs:76,117,166,228` `decode_{bond,angle,dihedral,nonbond}_params` (doc: "mirror the historical molpy … helper surface"); only consumer is `molrs-python/src/io/mod.rs` | per-row parameter decoding | io | Force-field decoding inside io, parallel to `ff/forcefield/readers/prmtop.rs:355,386,515`, which decodes on its own. molpy compat surface with no callers | ff prmtop reader | Delete the four, plus `parse_pointers`/`parse_a4_names` if unbound. `molrs::io::data::prmtop_tables::decode_*` removed |

**P2: two public paths, aliases, dead or compat code**

| # | path:line | evidence | fix (path change) |
|---|---|---|---|
| R16 | `lib.rs:119` "In-process MD: velocity-Verlet / Langevin and shifted Lennard-Jones."; `Cargo.toml:73` "In-process MD (integrators, shifted LJ)"; `md/mod.rs:13-15,28-30` explain pair-kernel naming | Docs still say md owns LJ | Reword: "integrators and force providers; kernels are in `ff::potential`" |
| R17 | `ff/potential/registry.rs:540-591` `register_kernel`, `register_kernel_with`, `lookup_kernel`, `lookup_typed_kernel`, `lookup_param_source`, `lookup_row_source` (doc: "thin shims over it"), re-exported at `ff/potential/mod.rs:27-30` | Second registration/lookup API over `ff::ir::{register_style, with_global}` | Delete the shims; keep `KernelRegistry::builtin` (the built-in constructor table). `ff::potential::{register_kernel*, lookup_*}` → `ff::ir::register_style` / `with_global` |
| R18 | `ff/ir/mod.rs:85-86` `pub use crate::ff::potential::generic::{CompoundForm, ParamCols, ScalarForm}; pub use crate::ff::potential::registry::{KernelConstructor, ParamSource, RowSource, SpecialClass};` | Two or three paths each (`ff::ir::X`, `ff::potential::X`, `ff::potential::registry::X`) | Move `ParamSource`, `RowSource`, `SpecialClass` and `KernelConstructor` into `ff::ir::registry` (they are registry vocabulary) and drop the `potential` re-exports. Keep `*Form` in `potential::generic` and drop the `ir` re-export |
| R19 | `ff/mod.rs:24-43` (`pub use forcefield::readers::{…LammpsFfReader…}`, `writers::{…}`, `typifier::cmap::assign_cmaps`, `typifier::gaff::GaffParameterSet`, `charge::{…}`, `scale_lj::{…}`) | Every symbol has a second path at the ff root; `assign_cmaps` has three (`ff::`, `ff::typifier::`, `ff::typifier::cmap::`) | Delete the `ff` root re-exports. Owners: `ff::forcefield::{readers,writers}`, `ff::typifier`, `ff::charge` |
| R20 | `ff/potential/soft.rs` `SoftPotential` + `optimize/soft.rs:33` `SoftSpec` (holds sigma, a_rep, k_bond…; resolves exclusions and pairs) + `optimize/potential.rs:268` `SoftLbfgs` beside `:23` `LBFGS` | Potential parameterization in optimize, and two L-BFGS front doors | Move `SoftSpec` into `ff::potential::soft` and make `SoftLbfgs` = `LBFGS` over a pair-rebuilding potential (see borderline). `optimize::SoftSpec` → `ff::potential::soft::SoftSpec` |
| R21 | `io/data/frcmod.rs:91` `write_frcmod` (raw section text, "Round-trips the historical molpy … surface") vs `ff::write_amber_frcmod` | Two frcmod writers | Delete the io writer, keep `parse_frcmod` (raw reader) only if something needs it; otherwise add an ff frcmod reader. `molrs::io::data::frcmod::{write_frcmod, format_frcmod}` removed |
| R22 | `io/data/prmtop_tables.rs:711,713` `SCEE_DEFAULT = 1.2`, `SCNB_DEFAULT = 2.0` vs `ff/params/amber.rs:30,37` `AMBER_SCEE`, `AMBER_SCNB`; `prmtop_tables.rs:280` `CHAMBER_COULOMB` | Duplicate constants | Keep the `ff::params::amber` constants |
| R23 | `io/trajectory/lammps_dump.rs:188` `fn canonical_column_name(name) { canonical_dump_column(name) }` | Alias | Delete |
| R24 | `perceive/mod.rs:21-25` ("The free functions it wraps remain available (and re-exported at the crate root)") | Two ways to perceive (`Perceive` builder and the free functions); the crate-root claim is false | Make the free functions `pub(crate)` or delete the builder; fix the doc |
| R25 | Systemic `pub mod x; pub use x::Y;`: `lib.rs:78,80` `pub mod core; pub use crate::core::*;` (so `molrs::core::store::frame::Frame` = `molrs::store::frame::Frame` = `molrs::Frame`); `lib.rs:90` builder flattened to the root; `core/types.rs:10` `pub use crate::op::types::{F,…}`; `io/smiles/mod.rs:113,155` (`io::smiles::smiles::X` = `io::smiles::X`); `ff/potential/pair/mod.rs:232,244` (`pair::lj_cut::LJCut` = `pair::LJCut`); `md/mod.rs:36-50`; `compute/mod.rs` | Every re-exported item has two or more paths | Decide once: leaf modules private (`mod x;`), keep the facade. Make `core` `pub(crate)` or hidden, since the root is the documented flat API. Drop the builder root flattening (no other subsystem is flattened) |

## 2. molrs-python and the other binding crates

**P1**

| # | path:line | evidence | why | fix (path change) |
|---|---|---|---|---|
| B1 | `src/md.rs:38,242,245,257,307,314,329` (`check_nx3`, `ErrSlot`, `take_err`, `SubclassPotential`, `Members`, `take_members`, `take_potential`); used by `src/ff/potential.rs:38 use crate::md::check_nx3;`, `src/ff/ir.rs:43 use crate::md::ErrSlot;`, `src/ff/mod.rs:855,903` | Potential-adaptation machinery lives in the md binding | ff depends on md: the dependency points the wrong way | Move to `src/ff/potential.rs`. No Python change |
| B2 | `src/ff/mod.rs` (2553 lines): typifiers (:100, :373, :1133, :1253, :1299), `OptReport` (:493, `module="molrs.optimize"`), `LBFGS` (:963), `Potentials`/`TypedPotentials`/`PotentialCompiler` (:553, :581, :1737), 17 force-field readers/writers (:1317-2480), `scale_lj` (:649-732) | One pyo3 file binds four subsystems; optimize is bound inside ff | Split into `src/optimize.rs`, `ff/typifier.rs`, `ff/potential.rs`, `ff/forcefield.rs`, `ff/scale_lj.rs` |
| B3 | `python/molrs/ff/__init__.py:73-87` (`from .._lib import PotentialCompiler`, `Potentials`, `intramolecular_pairs`) | Owned by `ff::potential` in Rust, but only reachable as `molrs.ff.*`; `TypedPotentials` is only reachable as `_lib` | `molrs.ff.{Potentials, PotentialCompiler, intramolecular_pairs}` → `molrs.ff.potential.*`; add `molrs.ff.potential.TypedPotentials` |
| B4 | `python/molrs/ff/__init__.py:91-102` (`from .charge import BccModel, GasteigerModel, MullikenModel`; `from .potential import Potential`; `from .typifier import AtdTypifier, …, Typifier`) | 11 symbols with two paths; contradicts `molrs/__init__.py:32` ("exactly one spelling") | Delete. `molrs.ff.{BccModel,…}` → `molrs.ff.charge.*`; `molrs.ff.Potential` → `molrs.ff.potential.Potential`; `molrs.ff.{Typifier, Match, *Typifier}` → `molrs.ff.typifier.*` |
| B5 | `python/molrs/fields.py:31-122` (PDB, LAMMPS, XYZ, MOL2 column renames), used at `io/__init__.py:185-202` | Python redoes format mapping. Rust readers already emit canonical names (`pdb.rs:409-431`, `lammps_data.rs:287`); only MOL2 does not | Move the MOL2 rename into Rust and delete `molrs.fields` |
| B6 | `python/molrs/io/raw.py:19-63` vs `io/__init__.py:91-165`; pass-through wrappers at `io/__init__.py:253,266,309,318` | 22 functions with two paths | Delete `molrs.io.raw` and the wrappers |
| B7 | `src/io/mod.rs:1115-1214` (`prmtop_parse_pointers`, `prmtop_parse_a4_names`, `prmtop_decode_{bond,angle,dihedral,nonbond}_params`, `read_amber_prmtop_sections`) | Python face of R15; no callers in tests or molpy | Unbind all seven |
| B8 | `src/io/mod.rs:1413` `read_top` vs `src/ff/mod.rs:2047` `read_gromacs_system` | Python face of R14 | Remove `molrs.io.read_top` (and `write_top`) |
| B9 | `src/io/mod.rs:2505-2512` `write_smiles` ("Equivalent to `SmilesIR.from_atomistic(mol, **flags).write_smiles()`") | Self-declared alias | Remove `molrs.io.write_smiles` |
| B10 | `src/core/system/molgraph.rs:2026` `#[pyclass(module = "molrs", name = "Reaction")]`; `python/molrs/__init__.py:75` | Rust owner is `perceive::smarts::Reaction`; Python puts it at the core top level | `molrs.Reaction` → `molrs.perceive.Reaction` |
| B11 | `src/core/system/molgraph.rs:43,1805,1864,2973` (`SmartsPattern`, `SmartsMatch`, `RingInfo`); `:1517` `Atomistic.max_ring_system_size` | Perceive bindings in the core file; perceive logic as a method on a core type | Move to `src/perceive.rs`; delete `Atomistic.max_ring_system_size` (use `RingInfo`) |

**P2**
- `src/helpers.rs` is a grab-bag: simbox parsing (:103-135), SMILES errors (:163-217), stream format (:237), compute collectors (:251, :271), pickle (:36-87). Error mapping is also split across `error.rs`, `helpers.rs` and `store.rs:24`. Move each piece to its single user; merge error mapping into `error.rs`.
- `src/compute/extra.rs` (organized by "0.0.16→0.0.17 cycle", 6 domains), `compute/fitting.rs` (binds fitting, spectroscopy and transport although `transport.rs` exists) and root-level `src/spectrum_checks.rs`: re-file one file per compute domain.
- Public `protocol`/`driver` modules give second paths:
  - `molrs.compute.protocol.Compute`, `molrs.ff.potential.protocol.Potential`, `molrs.md.driver.MD`.
  - `driver.py:14` also leaks `ForceField`, `PotentialCompiler`, `Potentials`, `NeighborList` and `VerletSkin` under `molrs.md.driver`.
  - Fix: rename to `_protocol.py` / `_driver.py`.
- Dead shim at `python/molrs/__init__.py:122-136`: lazy `md` `__getattr__` "against a compiled `_lib` that predates the `md` submodule", but `src/lib.rs:447` always registers `md`. Delete it.
- Stale stubs and docs:
  - `_lib.pyi:4387-4394` declares `MD_ENERGY`, `kb_md`, `energy_to_md` and `preset_energy_to_md` under md; none are registered, and unit conversion would belong to units anyway.
  - `_lib.pyi:3115` documents `Potentials.set_energy_scale`, which is not registered.
  - `src/ff/mod.rs:3-13` still names `molrs.intramolecular_pairs`.
  - `python/molrs/md/__init__.py:9-18` teaches `md.VerletSkin(md.NeighborList(...))`, but md exports neither.
  - `compute/__init__.py:4-6` falsely says flat `molrs.<Class>` spellings work.
- Wrong `__module__`: `src/ff/mod.rs:649` `module="molrs"` for `FragmentScaling` (not exported there); `:643` `molrs._lib` for `ForceField`. Set both to `molrs.ff`.

**Other binding crates**
- molrs-ffi and molrs-capi are clean.
- `molrs-cxxapi/src/lib.rs` (1911 lines, one file) binds compute (:157-316), io (:589-828), Frame (:885-1318), ff charge (:1365) and regions (:1423-1558). Split it per subsystem.
- molrs-wasm:
  - `compute.rs:133,400,522,622` holds neighbour search. `LinkedCell` (its own doc says the preferred spelling is `new NeighborList(3.0)`) and `BruteForce` duplicate `NeighborList`; delete them and move the rest to `core/spatial.rs`.
  - `compute.rs:1686,1866,1891` holds `Topology` and `findRings`/`TopologyRingInfo`, duplicating `perceive.rs:35`.
  - `ff.rs:198,298` binds `LBFGS`/`OptReport`; move to `optimize.rs`.
  - `smiles.rs` belongs under `io/`; `generate.rs` should be named `builder.rs`.

## 3. molpy (`molpy-016/src/molpy`)

| # | path:line | evidence | why | fix (path change) |
|---|---|---|---|---|
| M1 | `io/data/lammps.py:154,275` (`molrs.io.read_lammps_data(path)`, then `_type_labels_from_meta`, `_adapt_frame`, `_drude_flags`…) | `mp.io.read_lammps_data` / `write_lammps_data` / `LammpsDataResult` | A second LAMMPS data-format layer under the same name as `molrs.io`, with a different return type | Move into the native reader; `mp.io.X is molrs.io.X`; `LammpsDataResult` removed |
| M2 | `io/readers.py:34,41,95`; `io/writers.py:30` | `read_xyz` (post-joins columns), `read_amber_inpcrd(frame=)`, `read_amber_ac` (renames `q`→`charge`), `write_pdb` (element fallback) | Same names as the native functions, different behaviour | Move each into molrs; identity re-export. `read_amber_ac` → `read_ac` |
| M3 | `io/data/lammps_bond_react.py:78`; `io/writers.py:54,150` | `LammpsBondReactWriter`, `BondReactTemplate`, `write_bond_react_map`, `write_lammps_bond_react_system` | A whole file-format writer in molpy; molrs has none | Port to `molrs.io` |
| M4 | `io/__init__.py:30-40` (`read_forcefield_xml as read_xml_forcefield`, `read_gromacs_top_ff as read_gromacs_forcefield`, and the two writers) | Renaming aliases | Use the molrs names |
| M5 | `potential/__init__.py:44-55` (`from molrs.ff.ir import IrError, Param, StyleSpec, …` and `from molrs.ff.potential import LJCut, Potential, kernel`) | `mp.potential` mixes two owners | Split IR into `mp.ff.ir` |
| M6 | `__init__.py:182-204` root `Potentials`, `PotentialCompiler`; `md/__init__.py:5-6` | Collection only on the root, kernels on `mp.potential` (follows B3) | `mp.Potentials` → `mp.potential.Potentials` |
| M7 | `core/ops/__init__.py:10-16`, `core/ops/scale_lj.py:15` | `scale_lj` (different signature from `molrs.ff.scale_lj`), `compute_k_ij`, `fragment_scaling_data`, `intramolecular_pairs` | Force-field transforms in "core" | Delete `core/ops`; re-export from ff |
| M8 | `io/emit/{__init__,lammps,gromacs}.py` vs `engine/openmm.py:303`, `engine/lammps.py:316` | Engine decks in io; LAMMPS deck built twice | Move to `mp.engine`, one `generate_inputs` per engine. `mp.io.emit.*` → `mp.engine.*` |
| M9 | `core/region.py:73,110,124` (`class BoxRegion(molrs.Cuboid, …)`, `SphereRegion(molrs.Sphere)`, `Cube`, `And/Or/NotRegion`); `core/selector.py:118,164` | Two classes per shape; geometric selectors repeat regions | Add `mask`, `bounds` and composition natively; delete. `mp.BoxRegion` → `mp.Cuboid`, `SphereRegion` → `Sphere` |
| M10 | `core/unit.py:13-62` (`_NATIVE_PRESETS`, `openmm` preset, redefined k_B) | Second unit-preset registry | Add the presets natively; `mp.UnitSystem` → `mp.UnitRegistry` / `UnitPreset` |
| M11 | `builder/virtualsite.py:34` `load_polarizability`; `data/forcefield/alpha.ff`, `clp.xml` | CL&Pol data and parser split from `molrs/src/ff/params/clpol.rs` | Embed in `ff::params::clpol` |
| M12 | `builder/crystal.py`, `builder/symmetry.py:85` | `frac_to_cart` repeats `Box.to_frac`/`to_cart` | Use Box (lattice ownership: see borderline) |

**P3 (molpy)**
- `mp.compute.signal` should be `mp.signal`.
- `builder/__init__.py` names also exist under `mp.builder.polymer.*` and `.nanostructure.*`: make the subpackages private.
- Dead code to delete: `DistributionIR` (`builder/polymer/distributions.py:22`), `core/logger.py`, the "historical error shape" shim (`io/data/lammps.py:127`), and the `env_manager` aliases `pip`/`virtualenv` (`wrapper/env.py:38`).
- Subclasses that change the return type or only add a guard: `GrapheneBuilder`/`CarbonTubeBuilder` and `Conformer`. Upstream the change and re-export by identity.
- Two column vocabularies (`mp.fields.X` vs `mp.keys.X`): keep one.
- `mp.Script` belongs in `mp.engine.Script`.
- The stale comment in `builder/virtualsite.py:27` points at a deleted molpy potential.

## 4. molpack (`molpack`, branch dev)

Already clean: neighbour cells use molrs `CellGrid`, minimum image uses `SimBox::mic`, rotations use `op::rigid`, rotatable bonds and exclusions come from molrs, and there is no local L-BFGS.

| # | path:line | evidence | fix (path change) |
|---|---|---|---|
| K1 | `src/script/io.rs:42,90` (`pub use io::{read_frame, write_frame}` at `script/mod.rs:48`); `python/src/script.rs:167-184`; `python/molpack/cli.py:63-84` | Three copies of extension-based reader/writer dispatch | Add `molrs::io::{read_frame, write_frame}`; delete all three. `molpack::script::{read_frame,write_frame}` → `molrs::io::…` |
| K2 | `src/handler.rs:192-260` `XYZHandler` (hand-written extended XYZ) | Duplicates `molrs::io::data::xyz::XYZFrameWriter` | Use it (behind the `io` feature) or delete |
| K3 | `src/assemble.rs:82-158` `topology_frame` | Frame concatenation with endpoint offsets (as in `Frame::replicate`) | Add `Frame::concat` in molrs |
| K4 | `src/entry/setup.rs:179` `6.022_140_76e23` | Hard-coded Avogadro | Use `molrs::units::constants::AVOGADRO` |
| K5 | `src/restraint/geometric/mod.rs:33-42` `plane_repeats_along` (doc: "The same test molrs's `HalfSpace` applies") | Copy | Use `HalfSpace::repeats_along` |
| K6 | `src/grow/internal.rs:402-435` `unit`, `angle_at`, `dihedral`, `nerf`; `restraint/collective/geometry.rs:24` `unit`; `target.rs:612-631`, `optimizer/torsion_mc.rs:242` centroid | Geometry primitives that belong in `molrs::op` | Add `op::vec3::{angle, dihedral}` and `op::rigid::nerf` (absorbing `ff::potential::geometry::compute_angle/compute_dihedral`, lines 40 and 82); use `op::superpose::centroid` |
| K7 | `src/lib.rs:138-140,147` (`pub use molrs::{BondDistanceWeights, Element}`, `types::F`, `optimize::{OptReport, Optimizer}`), `lib.rs:200-234` `prelude` | Re-exports molrs symbols molpack doesn't own; the prelude duplicates the root | Delete both |
| K8 | Leaf `pub mod` plus root re-exports in `context/`, `grow/`, `optimizer/torsion_mc`, `pipeline/{combinators,engine}` | Three paths per type | Make the leaves private |
| K9 | `objective.rs:166,208,411` public `compute_f/fg/g`, plus inherent `PackContext::evaluate`, plus `Objective::evaluate`; `eval.rs:18` `EvalMode::RestMol` (alias of `FAndGradient`); `grow/moves.rs:145` `uniform` → `uniform01` | Duplicate APIs and aliases | `compute_*` → `pub(crate)`; delete the inherent method, `RestMol` and `uniform` |
| K10 | `python/molpack/version.py:20,34,65` `check_molrs_version`/`MOLRS_MINOR` vs `python/src/interop.rs:166` `check_abi`; two `molpack` CLIs (`pyproject.toml:75` and Rust `[[bin]]`) with different syntax; `cli.py:92` `_version_string` | Two gates and two CLIs for one job | Keep the ABI check and one CLI |
| K11 | `src/numerics.rs` (grab-bag; `objective_small_floor` ≡ `residual_small_floor`); `python/src/helpers.rs` (really errors; `NpF = f64` alias); `python/src/constraint.rs` binds `restraint`; `src/grow_error.rs` outside `grow/`; `Cargo.toml:54` `ff = ["molrs/ff"]` pass-through feature | Grab-bags and misfiled modules | Move each to its user; rename to `errors.rs` and `restraint.rs`; move to `grow/error.rs`; drop the feature |

molpack still pins molrs 0.15 at `../molrs/molrs` (`Cargo.toml:31`, `python/Cargo.toml:24,31`, `version.py:20`). That needs the 0.16 bump; the versions agent may already have it.

## Public-path changes for migration.md

**molrs Rust**
- `md::SpecialWeights` → `ff::potential::SpecialWeights`
- `optimize::SoftSpec` → `ff::potential::soft::SoftSpec`
- `ff::typifier::{BccParameterSet, BCCCorrectionTable}` → `ff::charge::…`; `BCCCorrector` removed
- `ff::forcefield::xml::read_{mmff_params,opls_typing}_xml_str` → `ff::typifier::{mmff,opls}`
- `ff::mmff::*` → private under `ff::typifier::mmff`
- `ff::potential::{register_kernel, register_kernel_with, lookup_*}` removed (use `ff::ir::register_style` / `with_global`)
- `ff::potential::{KernelConstructor, ParamSource, RowSource, SpecialClass}` → `ff::ir::…` only; `ff::ir::{CompoundForm, ParamCols, ScalarForm}` → `ff::potential::generic` only
- All `ff::` root re-exports (readers, writers, charge, `assign_cmaps`, `GaffParameterSet`, `scale_lj`) → their submodules
- Removed:
  - `perceive::equivalence::average_charges`
  - `io::data::top::*`
  - `io::data::prmtop_tables::decode_*`
  - `io::data::frcmod::{write_frcmod, format_frcmod}`
  - `compute::util` (`MicHelper`)
  - `conformer::etkdg::etmin::minimize`
  - `conformer::distgeom::uff`
  - `ff::mmff::hybrid::Hyb` (new `perceive::Hybridization`)
  - `io::smiles::smiles::*` (leaf path)
  - `molrs::core::*` paths, if `core` goes private
  - root builder flattening (`molrs::GrapheneBuilder` → `molrs::builder::GrapheneBuilder`)

**Python and JS bindings:** the full list is in the B rows of section 2.

**molpy and molpack:** as listed in the M and K rows.

## Borderline: needs your decision

1. **Force-field file I/O inside `ff`.** `ff::forcefield::{readers,writers}` (prmtop, LAMMPS `.ff`, GROMACS, OPLS XML, frcmod) parse files outside `io`, and `ff` imports `io::data::prmtop::parse_flag_sections`. This is a deliberate "structure in io, parameters in ff" split. Decide whether force-field formats count as io.
2. **Shared numerics in `core::math`.** `core::math::pair_form::lj_ab_to_sigma_epsilon` (an LJ parameter identity) sits in core so that io and ff can both reach it, and `core::math::Virial` is used by ff kernels and md. Both are reasonable shared base code, but one is pair-potential knowledge.
3. **ETKDG objective terms.** `conformer/etkdg/etmin.rs` `FirstStageField` and `ExpTorsionField` are internal distance-geometry objectives (RDKit parity). They could stay private to conformer or count as kernels outside ff.
4. **`core::store::record_v1`.** It converts force-field parameters per style (k halving, theta0 radians to degrees) inside core, while `ForceFieldSection` claims to know nothing of kernels. The alternative owner is `ff` (`ForceField::from_section`) or `io::mrec`.
5. **`ForceFieldSection` Python path.** Rust owner is `core::store`, Python path is `molrs.io.mrec`, and the binding file is in `src/ff/`. Pick one owner.
6. **`SoftLbfgs` vs `LBFGS`.** Keep `SoftLbfgs` as a second optimizer, or merge into `LBFGS` over a pair-rebuilding potential (R20).
7. **`perceive::coarsen::Coarsener`.** It is a coarse-graining mapping operator, arguably `builder` or `system::coarsegrain` rather than perception.
8. **Semantic type aliases.** `Bead = Atom`, `Bond = Relation` and `AtomId = NodeId` (`core/system/atomistic.rs:53-70`, `molgraph.rs:565`) are domain aliases; they are covered by the "delete aliases" rule only if you treat them as such.
9. **molpy namespace for ff.** Add an `mp.ff` that mirrors `molrs.ff`, or keep a flat root (decides M5-M7). Also decide whether molpy owns the crystal `Lattice`/`SpaceGroup` builder.
10. **molpack bit-parity.** The Packmol restraint kernels (`restraint/geometric/*`), the diamond-lattice walk, the separate RNG streams and the centroid order are kept for bit-identical Packmol results. Replacing them with molrs equivalents (regions, `SelfAvoidingWalk`, `op::superpose::centroid`) changes golden outputs.
## Rulings on the borderline items (coordinator, 2026-10-06; binding)

1. ~~Force-field file formats stay in `ff::forcefield::{readers,writers}` (they map files to the IR); `io` owns structure/trajectory formats. A shared low-level parser (`io::data::prmtop::parse_flag_sections`) may be used by ff. Document this split in each module's doc.~~ **Superseded by ruling 11** (user correction, 2026-10-07).
2. `lj_ab_to_sigma_epsilon` is pair-potential knowledge → `ff::potential` (pair). `core::math::Virial` stays in core (shared numerics).
3. ETKDG objective terms stay private to `conformer` (internal DG objectives, not IR kernels) — make them non-`pub`.
4. `core::store::record_v1` (v1→v2 force-field parameter conversion) moves out of core: per-style parameter semantics are ff knowledge → `ff::forcefield` (called by the mrec reader).
5. `ForceFieldSection`: Rust owner `core::store` (record data model) stays; Python path stays `molrs.io.mrec`; move its pyo3 binding file from `src/ff/` to `src/io/`.
6. Merge `SoftLbfgs` into `LBFGS` over a pair-rebuilding potential; `SoftSpec` → `ff::potential::soft`. One optimizer front door.
7. `perceive::coarsen::Coarsener` → `builder` (it constructs a new system; perception only derives facts).
8. Pure type aliases giving one type two public names are deleted (`Bead = Atom`, `AtomId = NodeId`), keeping the name used by the documented API. Relation-kind names (`Bond`, `Angle`, …) are distinct kinds, not aliases, if they are real classes/kind markers; if `Bond` is literally `type Bond = Relation`, keep exactly one public spelling per concept and document it.
9. molpy gets `mp.ff` mirroring `molrs.ff` by identity (M5–M7 follow); crystal `Lattice`/`SpaceGroup` builder stays in molpy (molrs has no crystal builder) but uses `Box` for frac↔cart.
10. molpack keeps its Packmol-faithful restraint kernels, lattice walk, RNG streams and centroid order (bit parity with Packmol is molpack's job) — document why in each; every other K item is fixed.
11. **User correction, 2026-10-07 (binding; supersedes ruling 1 and R10's target).** Verbatim: "你改错了啊！作为工厂函数，一定是io.read_foo/io.foo.FooReader的形式啊！" — a factory that reads or writes a file format has exactly one shape: a function at the top of `io` (`io.read_<fmt>[_<what>]` / `io.write_<fmt>[_<what>]`) or a class `io.<fmt>.<Fmt>Reader` / `<Fmt>Writer`. Force-field file formats are therefore `io`'s, not `ff::forcefield`'s: Rust `molrs::io::forcefield::{readers, writers, xml, lammps_units}` (the OPLS / MMFF typing-metadata XML readers included); `ff` is the data model and never depends on `io` outside `#[cfg(test)]` engine checks (enforced by `ff::io_boundary`). Python: every force-field reader/writer is `molrs.io.read_*` / `write_*`; `molrs.ff.forcefield` holds the `ForceField` data model only. Typifiers read no file (`OPLSAATypifier::new(meta, ff)`, `MMFF94Typifier::from_parts(params, ff)`). A class of one format lives in that format's submodule (`molrs.io.smiles`, `molrs.io.log`, `molrs.io.lammps_bond_react`, `molrs.io.trajectory`, `molrs.io.mrec`); the top of `molrs.io` holds functions only.
