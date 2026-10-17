Most of the audit is fixed. Of the roughly 60 rows and 10 rulings, 7 items are only partly fixed, the mechanical checks are clean apart from one blind spot in the public-paths test, and the fix waves introduced about 15 new problems. Two of the remaining issues are real bugs:
- **molpy GROMACS:** `GROMACSEngine.generate_inputs` writes a `.top` that `grompp` cannot use (details below).
- **molpack build:** molpack-016 builds against the wrong molrs (details below).

Nothing in the repos was edited; all three checkouts are clean at the stated commits. The rustdoc JSON, the wheel build and the runtime walks ran on compute node n220, and the allocation is cancelled and its scratch removed. The per-row checks were grep-based reads split across four read-only agents; I re-checked the main claims myself.

## Row status

| Area | Fixed | Not fully fixed |
|---|---|---|
| **molrs Rust, R1–R25** | All 25. Example evidence: `ff/potential/mod.rs:681` SpecialWeights; `grep MicHelper` → 0; `perceive/hybridization.rs:29` is the only Hybridization; `ff/mod.rs` has 0 `pub use`; `lib.rs:90` makes `core` private | R3 and R15 have leftovers (in the issues list). |
| **Rulings 1–8** | All 8. Examples: `record_v1` is now `ff/forcefield/record_v1.rs` (`pub(crate)`); `SoftLbfgs` → 0; Coarsener is in `builder/coarsen.rs`; `Bead`/`AtomId` → 0 | — |
| **Bindings, B1–B11 and the P2 list** | All of them. `ff/mod.rs` is 41 lines; `helpers.rs`, `extra.rs`, `spectrum_checks.rs`, `fields.py`, `io/raw.py` and the md `__getattr__` shim are gone; protocol/driver modules are now `_protocol`/`_driver`; stale stubs → 0 | — |
| **Ruling 5** (`ForceFieldSection` binding) | Moved to `src/io/mrec/section.rs` | — |
| **cxxapi** | Code split per subsystem | PARTIAL: the bridge is still one ~390-line string in `build.rs:9`, and the generated `src/bridge.rs` is committed although `.gitignore:66` ignores it. |
| **wasm** | LinkedCell, BruteForce, Topology/findRings, LBFGS and smiles all fixed. `generate.rs` became `conformer.rs`, not `builder.rs` (target changed) | — |
| **ffi / capi** | ffi is clean | capi is not clean, contrary to the audit (in the issues list). |
| **molpy, M1–M12, P3 and ruling 9** | 21 of 24, including M1–M10, M12, ruling 9 (`mp.ff` mirrors `molrs.ff`) and 5 of the 7 P3 bullets | M11 PARTIAL: `data/forcefield/clp.xml` is still shipped by molpy. P3 subclasses and P3 private submodules are PARTIAL (in the issues list). |
| **molpack, K1–K11** | K1, K2, K3, K5, K7, K8, K9, K10 and K11 | K4 PARTIAL: Avogadro is still hard-coded at `examples/pack_peo/main.rs:721`. K6 PARTIAL: `restraint/collective/geometry.rs:26` `unit` is still a wrapper. The 0.16 pin is PARTIAL (see molpack issues). |
| **Ruling 10** (molpack bit-parity) | Reason documented for the restraint kernels, lattice walk, RNG streams and `target.rs` centring | PARTIAL: no note for `context/rigid_view.rs:266-275` (a centroid) or `euler.rs`. |

## Mechanical checks

**(a) Rust public paths** (rustdoc JSON with full, serde, stream and filesystem features): 1581 public items, 137 public modules, **0 items with more than one public path**, 0 hidden duplicate paths, 0 re-exports of other crates. `pub mod x` alongside `pub use x::…` occurs **0** times.

**(b) Python public paths** (wheel built and imported):
- The repo's `test_public_paths.py` passes (11 tests).
- My own walk, which uses `dir()` rather than only `__all__`, covered 27 modules and 364 classes/functions:
  - **0 objects at two paths.**
  - **0 classes** with the wrong `__module__`.
  - **103 public functions report a module that is not their public path:** 89 say `molrs._lib`; 7 in `ff.ir` say `ir`; 2 in `op` say `op`; 5 in `store.schema` say `schema`.
  - 13 typing/stdlib names leak outside `__all__` in `molrs.io` and `molrs.ff.ir`.
- Blind spots in the test:
  - It walks only `__all__`.
  - It never checks a function's `__module__` (which is how the 103 get through).
  - Every symbol is still reachable through `molrs._lib.*`.
  - It does not check constants, export names that differ from `__name__`, or removed attributes such as `molrs.md.LJCut`.

**(c) Module jobs:**
- `molrs::md` holds 16 items, all integrators, force providers or state; its only `impl Potential` blocks are in tests.
- `molrs.md` exports exactly `Langevin, MD, MDState, MaxwellBoltzmann, VelocityVerlet`.
- wasm has no md module, and `ff` has no integrator (grep → 0).
- `io` is clean except `prmtop_tables` and `prmtop.rs:856` (in the issues list).
- Grab-bag modules: there are no util/utils/helpers/misc modules, but two "common" modules remain: `io/data/vasp_common.rs` (public) and `io/lammps/common.rs` (`pub(crate)`).

**(d) molpy against molrs:**
- **342** molpy public names are the molrs object itself.
- **2** molpy classes shadow a molrs class by subclassing it: `mp.Box` (over `molrs.spatial.Box`) and `mp.Trajectory` (over `molrs.store.Trajectory`).
- Smaller wrappers: `mp.io.read_smiles` (adds a guard), `mp.io.read_amber`, two `_write_mol2` copies, `_FORMAL_CHARGE`, and `metric_readers` LammpsLog/Mrec readers.
- **6** molrs objects have a second molpy path because public `engine.*` submodules or `core.trajectory` leak their imports, e.g. `molpy.engine.lammps.read_lammps_data` == `molpy.io.read_lammps_data`, and `molpy.core.trajectory.Frame`.
- Molpy's own classes have second paths through the public `core.*`, `engine.*`, `wrapper.*` and `adapter.base` submodules: about 30 cases, e.g. `mp.Box` == `mp.core.box.Box`.
- There is no `mp.store`, so root flattening does not create `mp.X` / `mp.store.X` pairs.

**(e) molpack:** 0 `pub use molrs` re-exports, 0 Python re-exports of molrs objects, 0 double public paths, and no format-dispatch IO left. Hand-rolled geometry not covered by ruling 10 remains in:
- `restraint/collective/com.rs:28` `centroids`
- `geometry.rs:34,85,102`
- `examples/pack_peo/main.rs:313-335`, plus a hand-rolled minimum image at `:460`

## Remaining issues

**molrs Rust**
1. Pure aliases that ruling 8 says to delete: `op/types.rs` defines `F3` = `FN` (`Array1<F>`) and `F3x3` = `FNx3` (`Array2<F>`).
2. Force-field meaning still in io:
   - `io::data::prmtop_tables` is still `pub` and holds ff-only helpers (`amber_phase` at `:314`, `canonical_terms`, `chamber_urey_bradleys`).
   - `io/data/prmtop.rs:856` turns SCEE/SCNB into 1-4 weights.
3. Duplicates:
   - Fractional-to-Cartesian exists three times: `vasp_common.rs:107` (public), `chgcar.rs:335` and `SimBox::to_cart`.
   - `SimBox::mic_kernel` (`simbox.rs:617`) repeats `Mic::apply`.
   - 14 private copies of `invalid_data`/`err_mapper` across `io/`.
   - `forcefield_section.rs:98` `unit_preset` repeats the `UnitPreset` tables.
4. Compat alias: `io/reader.rs:355` `open_file` calls itself "a compatibility wrapper".
5. Public modules with no public items: `perceive::aromaticity` and `ff::forcefield::lammps_codecs`.
6. New from the fix waves:
   - `ff/potential/soft.rs:227,231` add local `sub`/`norm` copies of `geometry::sub3` / `op::vec3::norm`.
   - `perceive/hybridization.rs:261,272` `min_valence` duplicates `Element::default_valences`, and `default_valence` is a one-line alias of it.
   - Ruling 4 created an io → ff dependency while ff already depends on io; `record_io.rs:694` adds a `cfg(not(ff))` stub enum to cover it.
7. Second public entry points:
   - `optimize::minimize_lbfgs_rms` next to `LBFGS`; only `conformer` calls it.
   - `compute_gasteiger_charges` next to `GasteigerModel`.

**Bindings**
1. 103 functions report the wrong `__module__` (see check b).
2. Doc examples that no longer work, because the flat names were removed:
   - `src/core/store/block.rs:55`, `simbox.rs:77` and `frame.rs:617` still use `from molrs import Block/Box/Frame`.
   - `md/_driver.py:339` describes MD units, but the module is unit-agnostic.
3. `molrs.io.TrajectoryReader` and `molrs.io.mrec.TrajectoryReader` are two different classes with one name.
4. mrec is split between `molrs.io.*mrec*` functions and `molrs.io.mrec`.
5. `SmartsPattern.find_matches(mapped=True)` is a self-described "legacy shortcut" that duplicates `SmartsMatch.mapping`.
6. `ff/forcefield/forms.rs:15` `ir_err` is a one-line wrapper of `ff::ir::refuse`.
7. `_lib.pyi` has no `schema` submodule stub.
8. capi:
   - `molrs-capi/src/forcefield.rs:704,794` is a second ForceField JSON format, about 300 lines, beside the core `to_section`/`from_section`; it was extended during the fix window.
   - `capi/src/lib.rs:76` declares its own `pub type F = f64`.
9. wasm:
   - The whole-content readers in `io/reader.rs` are labelled "legacy" and duplicate the `*Stream` readers.
   - `covalentRadius` is bound at the crate root instead of in core.
   - `optimize.rs:34-46` builds its own O(N²) pair list.

**molpy**
1. Bug: `engine/gromacs.py:167` writes the `.top` with `write_gromacs_top_ff`, which writes directives only, so `grompp` cannot use it; it should call `write_gromacs_system`. The test only checks for `[ defaults ]`.
2. `mp.Box` and `mp.Trajectory` subclass molrs classes instead of being the same objects.
3. Second paths: `core`, `engine.*`, `wrapper.*` and `adapter.base` are public; only `builder` was made private.
4. `builder/_virtualsite.py:21,133` hard-codes `FOUR_PI_EPS0` and `/ 4.184` instead of using `molrs.units`; the `/ 4.184` was added in 803df38.
5. Stale docs: `core/__init__.py:4` cites the deleted `mp.Config`, and `docs/tutorials/naming-conventions.md:73` still uses the removed `read_xml_forcefield`.
6. Seven empty leftover directories (`potential/`, `conformer/`, `typifier/`, `builder/nanostructure/`, `core/ops/`, `io/data/`, `io/emit/`) hold only ignored `__pycache__`, so `import molpy.potential` still succeeds locally.

**molpack**
1. Build bug: `Cargo.toml:31,58` and `python/Cargo.toml:24,31` use `path = "../molrs/molrs"`. From molpack-016 that is the dev checkout, which is version 0.15.1 and lacks `read_frame`, `Frame::concat`, `nerf` and `vec3::dihedral`. There is no `[patch]` pointing at molrs-p-registry, so molpack-016 cannot build against molrs 0.16 as it stands.
2. Molecule-ID numbering is computed three times: `handler.rs:290`, `assemble.rs:97` and `assemble.rs:156`. `XYZHandler::snapshot` largely repeats `coords_only_frame`.
3. `pack/gencan/mod.rs:6-10` still has `pub mod` under a private parent.
4. Stale references: `python/src/interop.rs:14` says molpack is built "`ff`-only" (the feature is gone), and several docs cite `src/gencan/` instead of `src/pack/gencan/`.

The per-area agent reports have further `path:line` evidence for every row if you need it.
## Coordinator rulings for fix wave 2 (binding)

- Pure aliases `F3`=`FN`, `F3x3`=`FNx3`: keep ONE name per type (the one with more uses across molrs+molpack), delete the other everywhere.
- No io↔ff dependency cycle and no `cfg(not(ff))` stub: the v1→v2 record migration must live where it needs no reverse dependency. Preferred: the mrec reader (io) returns the stored section + version untouched; `ff` performs the force-field-section conversion (`ForceField::from_section` handles version 1), and the frame-column conversion is driven by data the io layer owns or is applied by ff when a force field is present. Pick a design without a cycle, document it.
- `prmtop_tables` ff-only helpers and SCEE/SCNB→1-4 weights move to the ff prmtop reader; `prmtop_tables` becomes `pub(crate)` or loses those items.
- Exactly one frac↔cart (`SimBox`), one minimum image (`Mic`), one `invalid_data` error helper for io, one unit-preset table (`UnitPreset`), one valence table (`Element`), `soft.rs` uses `geometry::sub3`/`op::vec3::norm`.
- `open_file` compat alias: delete. Empty public modules (`perceive::aromaticity`, `ff::forcefield::lammps_codecs`): private.
- `optimize::minimize_lbfgs_rms`: conformer uses `LBFGS`; delete or make pub(crate). `compute_gasteiger_charges`: delete, `GasteigerModel` is the door.
- Python: every function's `__module__` = its public path; `test_public_paths.py` extended to walk `dir()`, check functions/constants, and assert removed names (e.g. `molrs.md.LJCut`) are absent. Doc examples use the new paths. One `TrajectoryReader` name per class (rename or unify the mrec one). ~~All mrec doors live in `molrs.io.mrec`.~~ **Superseded below** (user correction, 2026-10-07): the mrec doors are `molrs.io.read_mrec*` / `write_mrec*`, and the mrec store classes are `molrs.io.mrec.MrecReader` / `MrecWriter`. Delete `find_matches(mapped=True)` shortcut, the `ir_err` wrapper. Add the `schema` stub.
- capi: one ForceField serialization — use the core section (`to_section`/`from_section` JSON), delete the second JSON format; use `molrs::op::types::F`.
- cxxapi: the generated `src/bridge.rs` is either generated at build time and not committed, or committed and not ignored — choose the one Atomiverse's build supports, make it consistent; the bridge definition is a real source file, not a 390-line string in build.rs, if feasible.
- wasm: delete the "legacy" whole-content readers (streams are the door); `covalentRadius` into core; `optimize.rs` uses `NeighborList`.
- `vasp_common` / `lammps/common`: rename by what they are (no "common" grab-bag), private where possible.
- molpy `mp.Box`/`mp.Trajectory` subclasses: upstream their correct additions into `molrs.spatial.Box`/`molrs.store.Trajectory` so molpy re-exports by identity.
- molpy: GROMACS engine uses `write_gromacs_system` (+ a grompp test if gmx is available); `core`, `engine.*`, `wrapper.*`, `adapter.base` submodules private; constants from `molrs.units`; stale docs; remove empty dirs.
- molpack: K4/K6/ruling-10 notes leftovers, remaining hand-rolled geometry → `op`, one molecule-ID numbering, `gencan` pub mod, stale refs. The `../molrs/molrs` path is the CI layout — not a bug; build locally via the scratch layout.

## User correction (2026-10-07, binding; supersedes the rulings above where they conflict)

Verbatim: "你改错了啊！作为工厂函数，一定是io.read_foo/io.foo.FooReader的形式啊！" Every file-format factory has exactly one shape: a function at the top of `io` (`io.read_<fmt>[_<what>]` / `io.write_<fmt>[_<what>]`) or a class `io.<fmt>.<Fmt>Reader` / `<Fmt>Writer`. Superseded: audit ruling 1 (force-field formats in `ff::forcefield`), the wave-2 bullet "All mrec doors live in `molrs.io.mrec`", and the placement of the frame-bytes codec under `molrs.stream`. Now:

- Rust: force-field file formats are `molrs::io::forcefield::{readers, writers, xml, lammps_units}`; `ff` is the data model and names `io` only in `#[cfg(test)]` code (the engine checks `equivalence_check`, `engine_codec_check`, cmap `lammps_check`, … are all test-only), enforced by `ff::io_boundary`. The OPLS / MMFF typing-metadata XML readers are `io::forcefield::xml`'s; typifiers take parsed parts. `io::mrec::{FrameSequence, FrameSequenceWriter}` → `io::mrec::{MrecReader, MrecWriter}`; `io::log::parse_lammps_log_text` → `read_lammps_log_str`; new `io::smiles::read_smiles`. The frame-bytes codec stays in `molrs::stream` (its MessagePack/JSON wire types are shared with `ControlCommand` and pull deps `io` must not take); only its Python doors moved.
- Python: `molrs.io` top level = functions only — every structure/trajectory/force-field reader and writer, `read_mrec*` / `write_mrec*`, `read_frame_bytes` / `write_frame_bytes`, `read_smiles`, `read_lammps_log[_str]`. Classes live in their format's submodule: `molrs.io.trajectory.TrajectoryReader`, `molrs.io.mrec.{MrecReader, MrecWriter, SequenceSchema, ForceFieldSection, section_names, pack, schema}`, `molrs.io.smiles` (SmilesIR, SmilesError, CGsmiles records), `molrs.io.log` (Lammps* records), `molrs.io.lammps_bond_react.BondReactTemplate`. `molrs.ff.forcefield` = data model only; `molrs.stream` = `Publisher`, `ControlCommand`. No aliases. `test_public_paths.py` enforces it.
- WASM: the mrec cursor's JS class is `MrecReader` (was `TrajectoryReader`).
