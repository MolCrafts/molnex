# Naming and placement audit (2026-10-07)

This is the full list behind the user's ruling: "molrs也不应该有store这种名字，所有的名字都是有意义的！再过一遍检查".

**Scope**
- molrs `ir/p-registry` @ dfc0e8c1: Rust, Python, wasm, capi, cxxapi and ffi.
- molpy `ir/molpy-016` @ b8f588d.
- molrec `ir/w3d-compat` @ 5ba4c09.
- molpack `compat/molrs-016` @ 4003859.

## Rulings in force
- **R1 File doors.** A function at the io top is `io.read_<fmt>[_<what>]` / `io.write_<fmt>[_<what>]`. A class is `io.<fmt>.<Fmt>Reader` / `<Fmt>Writer`. Per-format classes live in `io.<fmt>`. The `read_`/`write_` prefixes are reserved for file doors.
- **R2 One job per module.** Potentials only in `ff.potential`, `md` holds integrators only, geometry lives in `op`, perception in `perceive`, and SMARTS lives wholly in `perceive.smarts`.
- **R3 One public path per symbol.** No aliases or shims. Leaf modules are private.
- **R5 Constants.** Constants live in `units::constants`.
- **R6 No duplicate implementations.**
- **N1 No container words.** Never `store`, `data`, `generic`, `common`, `helpers`, `utils`, `misc`, `base`, `types`, `support`, `shared`, `info`, `entry`, `holder` or `lib`.
- **N2 Names state the job.** No vague verbs and no undecodable abbreviations.
- **N3 Same concept, same name** across Rust, Python, JS and C. One name never means two concepts.
- **core.** molrs has a single `molrs.core` / `molrs::core` and no `store`/`system`/`spatial`/`units` top modules. The molpy root holds subsystems plus the promoted core data classes only.

## A. Decided, not yet done
- **A1.** `store`/`system`/`spatial`/`units`/`math` become `molrs.core`. Today: `lib.rs:91` re-exports `core::{…}` at the root, and Python has `molrs.store/system/spatial/units`.
- **A2.** SMARTS moves to `perceive::smarts`, and the shared SMILES/SMARTS grammar goes to a private notation module. Delete `write_smarts`, `write_local_smarts` and `SmilesIR.write_smarts`, replaced by `SmartsPattern.from_environment` and `str(pattern)`. The ASTs are duplicated: `io/smiles/chem/ast.rs:303,416,473` vs `perceive/smarts/ast.rs:172,215,299`.
- **A3.** Follow-ups from molrec/molpack:
  - `scripts/ci-conformance.py` must use `molrec.suite.ConformanceSuite` and `molrec.report.*`.
  - `SECTION_PRESETS` (`forcefield_section.rs:91`) needs `openmm`.
  - Add a cm³→Å³ constant.
  - Expose Python `molrs.units.constants`.
  - Bump the partner pins.

## B. Still present from the first audit (verified at dfc0e8c1)

**io layout and door names**
- io is split by content kind (`io::data/trajectory/log/mesh`) rather than per format.
- Reader/writer classes use the wrong shape: `PDBReader`, `SDFReader`, `XYZReader`, `LAMMPSDataReader`, `*FrameWriter`, `LAMMPSTrajReader` vs `LAMMPSDumpWriter`.
- Rust and Python door names don't match: pyclass `*TrajReader` vs Rust `DcdReader`, `read_gro` (single vs multi), `read_xyz_traj`.
- Bytes/text doors use other shapes: `parse_frame_bytes` ×8, `parse_frame_with_decoder_context`, `parse_ac`, `parse_prep`, `format_prep`, `parse_stl`.
- csv is named `block_from_csv` / `block_to_csv` in Rust and `read_block_csv` in Python.

**Misplaced subsystems**
- Ring perception lives in `system` (`Topology::find_rings`, `TopologyRingInfo`).
- Geometry ops live in `spatial::geometry`, which makes a spatial↔system cycle.
- ETKDG has its own energy kernels (`conformer/etkdg/etmin.rs`).
- `ForceFieldSection` lives in `store` and duplicates vocabularies.
- `md::Comm`, `md::kinetic_energy` and `com_velocity` are in `md`; Python recomputes kinetic energy in `md/_driver.py:388`.
- The torsion form family sits in `ff/forcefield/torsion.rs`, an ir↔forcefield cycle.

**Duplicates and aliases**
- `ff::potential::geometry` aliases `op::vec3`; `KernelRegistry` is public.
- Constants live outside `units::constants`:
  - `ff::constants` (`MDYNE_A_TO_KCAL`, `VACUUM_DIELECTRIC`, `DEG2RAD`);
  - spectroscopy (`C_MS`, `FS_TO_S`, `1.438777`);
  - `uff::G = 332.06`;
  - `preset.rs:353` (`4.184/10.0`).
- Three ACF implementations: `signal::acf_fft`, `compute::Acf`/`autocorrelation`, `transport::unbiased_cartesian_acf`. `forward_fft_onesided` sits in `fitting`.
- `dihedral_deg` is re-implemented in `perceive/equivalence.rs:575` and `ff/potential/cmap/charmm.rs:268`.
- `LammpsFfUnits` (`io/forcefield/lammps_units.rs`) restates the `UnitPreset` tables.

**Keys and public modules**
- Keys are declared outside `store::keys`: `COEFFS_TEXT_META`, `UNITS_META`, `REACT_ID`, `EQUIV_CLASS`, `BCC_BOND_TYPE`, `PORTS`, `FRAG_ID`, and pyo3's `VSITE` and `BEAD_ATOMS`.
- Leaf modules are public: `store::{keys,precision,schema,type_labels,typed_json}`, `io::data::*`, `io::trajectory::*`, `op::*`, `perceive::*`, `spatial::{neighbors,region}`, `typifier::{cmap,mmff}`, `math::*` and `conformer::distgeom`. `io::trajectory::xdr` is public, and so are the PDB/XYZ/DCD/prmtop parser internals.
- `pub const VERSION` sits at the crate root.

**Cross-language naming**
- `Box`/`SimBox` and `Graph`/`MolGraph` differ.
- `ff.ir` naming: `Param`/`CategoryInfo`/`StyleInfo` vs `*Spec`; Python's `StyleSpec` is something else; Python `Arity`/`Dim` exceptions clash with the Rust types.
- Python compute names differ from Rust: `Onsager`, `Persist`, `CenterOfMassResult`, `Dielectric`, `Angle`, `Dihedral`, `TypedPotentials`.
- `BondOrder` is defined three times: `compute::BondOrder`, `mmff::topo::BondOrder`, `CGBondOrder`.
- `fragment_scaling_data` vs `builtin_fragment_scaling`.
- `MOLREC_VERSION` and `RESERVED_META_KEYS` have different Rust and Python paths.

**Bindings**
- WASM:
  - re-implements pair exclusions (`optimize.rs:163-237`);
  - does its own format dispatch with non-core names (`writeFrame`, `writeFrameBytes`, `readFrameBytes`);
  - names dtypes `f64` where the core says `float`;
  - uses mixed reader casing (`CIFReader`/`AcReader`, `parseSMILES`, `readSTL`);
  - calls its streams `LAMMPSStream`/`LAMMPSTrajStream`;
  - calls `TriMesh` `Mesh`;
  - names the module `io/zarr` and exports `RecordReader` to JS as `MrecReader`;
  - duplicates the zip reader;
  - keeps both `schemaJson` and `schemaDocument`;
  - keeps the `rdf.*`/`voronoi.*` id shim;
  - does its own LAMMPS-log sniffing and PMFT quaternion math;
  - has LBFGS defaults of 200 vs 500;
  - doesn't mark x/y/z ↔ N×3 copies (`Frame::coords`) with `js_name`.
- cxx:
  - `write_frame` writes mrec, plus `read_first_frame`/`xyz_read_first_frame`;
  - compute names `MsdCompute`/`RdfCompute`/…;
  - `frame_column_u32` moves u64;
  - BCC parameter table duplicated with Python;
  - Einstein D written twice;
  - stale bridge comments.
- capi:
  - `molrs_frame_from_smiles` sits under `frame` and its docs name the wrong columns;
  - `molrs_ff_*` vs `MolrsForceFieldHandle`;
  - `read_meta`/`put_meta`;
  - re-types the `DType::name` table by hand;
  - `CStore::clear` leaks regions (bug).
- ext-example: `const D` (deg2rad).
- scripts: copies of `4.184`, `332.06371`, `read_tsv`, the element table, union-find and the LAMMPS-log parser (×5).

**Docs**
- `interop.md:67,181,194-196,236`;
- `attribution.md` stale paths;
- READMEs say "Upgrading from 0.14";
- `quickstart-wasm.md:33`;
- `py:ff/__init__.py:7` still says `ff.forcefield` holds the file doors.

## C. New findings (second pass, molrs)

### C1. One name with two meanings, or two names for one concept (N3/R6)

| # | Where | Now | Proposed |
|---|---|---|---|
| 1 | `io/forcefield/xml.rs:48`, `writers/xml.rs:926` | `read_forcefield_xml` reads 3 layouts; `write_forcefield_xml` writes OpenMM only | `read_molrs_xml_forcefield` / `read_openmm_xml_forcefield` / `write_openmm_xml_forcefield` |
| 2 | `io/forcefield/readers/opls.rs:114` | `OplsXmlReader` reads OpenMM XML | `io::openmm_xml::OpenmmXmlReader` |
| 3 | `io/forcefield/writers/xml.rs:100` | `XmlForceFieldWriter` | `OpenmmXmlWriter` |
| 4 | `system/bond.rs:30` vs `ff/forcefield/mod.rs:168` | `BondType`: chemical bond class vs force-field parameter row | chemical side renamed (`BondClass`/`BondKind`) |
| 5 | `block/column.rs:180`, `entity_table.rs:80`, `potential/generic/mod.rs:47` | `Column` ×3 | `Column`, `EntityColumn`, `StyleParamColumn` |
| 6 | `compute/error.rs:12` | `compute::NodeId` duplicates `system::NodeId` | delete |
| 7 | `compute/traits.rs:113` vs `op/superpose.rs:98` | `Fit` ×2 | `op::Superposition` |
| 8 | `ff/ir/dim.rs:25`, `units/dimension.rs:22`, `schema/column.rs:42` | three dimension types | one `units::Dimension` |
| 9 | `ir/spec.rs:61` vs `forcefield/mixing.rs:22` | `Mix` vs `Mixing` | `ParamCombination` / `CombiningRule` |
| 10 | `typifier/mmff/params.rs:7,26` vs `params/mmff.rs:47` | `MMFFAtomProp`/`MMFFParams` vs `MmffProp` | one row type |
| 11 | `forcefield/lammps_codecs.rs` | codec structs share kernel names (`DihedralCharmm`, `SignedCosine`, …), and are placed in `forcefield` | `*Codec`, under `ff::ir::engine_codec::lammps` |
| 12 | `forcefield_section.rs:237` vs `ir/registry.rs:235` | `StyleEntry` ×2 | `SectionStyleTable` / `RegisteredStyle` |
| 13 | `ff/ir/registry.rs` vs `ff/potential/registry.rs` | two `registry` modules | `ff::ir::style_registry`; potential table private |
| 14 | `ff/one_four.rs` (test) vs `ff/forcefield/one_four.rs` | `one_four` ×2 | `one_four_lammps_check` |
| 15 | `ff/ir/expression.rs` vs `ff/ir/expr/` | two modules for one concept | merge |
| 16 | `py:store/schema.py` vs `py:io/mrec/schema.py` | `schema` ×2 | `io.mrec.validation` |
| 17 | `py:io/trajectory.py:22` vs `io/reader.rs:207` | `TrajectoryReader`: a Python class vs a Rust trait | Python `ConcatTrajectoryReader` |
| 18 | `stream` `frame_to_bytes`/`bytes_to_frame` vs Python `io.read_frame_bytes` | one job, two names | one pair |
| 19 | `io/reader.rs:357` vs `io/writer.rs:66` | `validated` vs `check_before_write` | `check_read_frame` / `check_write_frame` |
| 20 | `ff/potential/*` | `BondHarmonic`, `UffBond`, `MMFFBondStretch`, `LJCut`, `PairBuck`, `MMFFVdW`, `*_ctor`, `stbn`, `oop` | `<Category><Style>` everywhere (`PairLjCut`, `BondUff`, `AngleMmffStretchBend`) |
| 21 | compute exports | `MSD`/`MsdMode`, `RDF`/`RdfMode`, `VACF`, `COMResult`, `RgResult`, `IRFlux`, `PMFTXY` | one casing (`Msd`, `Rdf`, `CenterOfMassResult`, …) |
| 22 | typifier/charge | `BCCAtomChargeTypifier`, `OPLSAATypifier`, `UFFTypifier` vs `GaffTypifier`, `BccModel` | one casing |
| 23 | `io/data/*` | `_from_reader` / `_to_writer` / `write_X_frame(W)` / `_into` / `format_prep` | one suffix family |
| 24 | `perceive/*` | `find_*`, `detect_rotatable_bonds`, `judge_bond_orders`, `assign_stereo_from_3d`, `ring_classes`, `hybridizations` | one verb |
| 25 | `perceive/builder.rs:70` | `Perceive` builder duplicates the free functions; module `builder` clashes with `crate::builder` | delete it, or `perceive::perceiver::Perceiver` |
| 26 | `io/writer.rs:41` | `ToFrame for Atomistic` duplicates `Atomistic::to_frame` | keep one |
| 27 | `optimize` | `MinResult` vs `OptReport` | one `OptimizationReport` |
| 28 | `ffi:store.rs:247-417` vs `ffi:shared.rs:216-343` | duplicate column accessors | delete the `Store` set |
| 29 | capi/cxx/wasm | meta verbs (`read_meta`/`put_meta`/`getMeta`); `n_*` vs `num_*`; H matrix as `molrs_box_h`/`hMatrix`/`frame_box` | one name each |
| 30 | wasm | `RecordReader`↔`MrecReader`; snake_case methods without `js_name`; `OptReport` fields, `energyForces`, `generate3D`, … | match core |

### C2. Names that say nothing (N1/N2)

| # | Where | Now | Proposed |
|---|---|---|---|
| 31 | `core/mod.rs:46` | `core::store` | dissolved into `core` (A1) |
| 32 | `ffi:store.rs`, `ffi:shared.rs` | `Store`, `SharedStore`, `new_shared` | `FrameArena`, `FrameArenaCell` |
| 33 | `capi:store.rs` | `CStore`, `GLOBAL_STORE`, `lock_store` | `HandleRegistry` |
| 34 | `io/zarr/store.rs`; `io::mrec` `*_store`; wasm `HostStore` | Zarr "store" | `*_storage` |
| 35 | `py:_lib`, `pyo3:lib.rs:119` | `molrs._lib` | `molrs._native` |
| 36 | `ff/potential/generic/` | `generic` | `form_kernel` |
| 37 | `op/types.rs`, `md/types.rs`, `wasm:core/types.rs` | `types` ×3 | `op::scalar`, `md::state`, `wasm ndarray` |
| 38 | `compute/traits.rs`, `compute/result.rs` | `traits`, `result` | `compute::analysis_contract` |
| 39 | `*/test_support.rs`, `test_util`, `pair::testing` | support / util | `fixtures` |
| 40 | `store/trajectory.rs:164` | `ObservableData` | `ObservableValues` |
| 41 | `block/column.rs:63` | `ColumnHolder` (+13 `from_*_holder`) | `ColumnArray` |
| 42 | `perceive/rings.rs:38`, `topology.rs:807`, `ir.StyleInfo`/`CategoryInfo` | `*Info` | `RingSet`, `RegisteredStyle` / `RegisteredCategory` |
| 43 | `io/streaming.rs:32`, `cxx:bridge.rs:153` | `FrameIndexEntry`, module `streaming`, `MetaEntry` | `io::frame_index::FrameOffset`, `KeyedMetaValue` |
| 44 | `typifier/estimate/mod.rs:47` | `TypifierParameterContext` | `EstimationInputs` |
| 45 | `ff/ir/engine.rs:486` | `no_extra` | name what is absent |
| 46 | `py:ff/scale_lj.py:10` | `fragment_scaling_data()`, `frag_data=` | `ff.params.clpol_fragment_scaling()`, `fragment_table=` |
| 47 | `pyo3:io/data.rs`, `io/json.rs`, `py:io/_trajectory_doors.py` | data / json / doors | `structure.rs`, `json_to_py.rs`, `_trajectory_readers.py` |
| 48 | `ff/potential/mod.rs:483,592,600` | `Member`, `TypedKernel`, `TypedMember` | `ForceTerm`, `ScaledTerm`, `WeightedTerm` |
| 49 | `ff/potential/instances.rs:45`, `need.rs`, `generic/form.rs:27` | `Instances`, `need`, `ParamCols` | `ExplicitTerms`, `param_reads`, `ParamColumns` |
| 50 | `ff/ir` root | `Value`, `Sample`, `Refusal`, `Metric`, `Residual`, `with_global`; module `engine` | `ParamValue`, `ConformanceSample`, `FormRefusal`, `FitMetric`, `FitResidual`, `with_global_registry`; `engine_codec` |
| 51 | `typifier/mod.rs:66,70` | `Typifier::match → Match`, `library()` | `assign → TypeAssignment`, `forcefield()` |
| 52 | `typifier/mmff/{frame_builder,topo,embedded}`, `opls/{embedded,deps,meta}` | names misstate the content | `typing`, `topology_snapshot`, `shipped_params`, `dependency`, `typing_metadata` |
| 53 | `ff/params/gaff_equiv.rs` | holds PARMCHK.DAT | `ff::params::parmchk` |
| 54 | `optimize/potential.rs`, `Optimizer::run` | module `potential` holds LBFGS | `optimize::lbfgs_driver`, `minimize` |
| 55 | `md/forces.rs:155`, `integrators.rs:489`, `md/pairs.rs` | `Direct`, `scalar_mass`, `pairs` | `SelfPairedForces`, `uniform_masses`, `periodic_copies` |
| 56 | `op/rigid.rs:97,112,143,188` | `about`, `alignment`, `frame`, `nerf` | `rotation_about`, `alignment_axis_angle`, `orthonormal_frame`, `place_from_internal_coords` |
| 57 | `perceive/bond_type.rs` | holds kekulization plus BCC bond types | `perceive::kekule` + `perceive::bcc_bond_class` |
| 58 | `perceive/ring_class.rs:40,65` | `RingSlot`, `RingFacts` | `AntechamberRingMembership`, `AntechamberRingSummary` |
| 59 | `conformer/graph.rs`, `distgeom/{knowledge,matrix,smooth,mol_features}`, `mol_features::perceive` | vague, and clash with `crate::perceive` | `topological_distance`, `basic_knowledge_torsions`, `bounds_matrix`, `triangle_smoothing`, `dg_perception` |
| 60 | `builder/strategy.rs` vs `builder/walk.rs` | trait in `walk`, impls in `strategy` | `builder::growth_strategy` |
| 61 | `compute/ml/`, `density/spatial.rs`, `dynamics/persist.rs`, `environment/match_env.rs`; `Pca2`, `PersistResult`, `AnyObservable`, `DistKind`, `NetworkResult`, `LifetimeResult`, `compute_qlm`, `compute_current_density` | vague | `clustering`/`decomposition`, `spatial_distribution`, `pair_survival`, `environment_match`; `Pca`, `PairSurvivalResult`, `HBond*Result`, `steinhardt_qlm`, `current_density` |
| 62 | `stream/message.rs`, `MessageFormat`; `signal/grid.rs`; `region/region.rs`; `store/meta.rs`; `core/test_rayon.rs` | vague or stuttering | `stream::control`, `FrameEncoding`, `signal::frequency_grid`, `region::primitives`, `metadata`, `test_thread_pool` |
| 63 | `io/format.rs`, `io/smiles/smiles/`, `from_atomistic` / `to_atomistic` / `frame_reader::parse_atomistic` | vague or stuttering | `io::by_extension`, `io::smiles::graph`, `smiles_from_atomistic` / `atomistic_from_smiles` |
| 64 | `io/data/xyz.rs:18-44`, `lammps_units.rs:28` | `Primitive`, `ExtValue`, `PropType`; `parse_style` | `ExtxyzScalarKind`, …; `parse_lammps_units_style` |
| 65 | cxx `frame_column_u32`, capi `molrs_block_set_F/I/U`, `molrs_sizeof_F/I/U` | wrong or undecodable | `_u64`, `_f64` / `_i32` / `_u64` |
| 66 | Python `NeighborQuery.free`, `LJCut.eval*`, `ff.potential.kernel()`, `forcefield.Type`, `Box.isin` | vague | `unbounded`, `energy_forces_*`, `potentials_for_style`, `ForceFieldType`, `contains` |

### C3. File doors (R1), placement (R2), paths (R3), constants (R5)

| # | Where | Now | Proposed |
|---|---|---|---|
| 67 | `io/trajectory/xdr.rs` | public XDR `read_i32` … `write_string` | private `io::xdr` |
| 68 | `io/trajectory/*` | `open_dcd`, `open_trr`, `open_xtc`, `open_lammps_dump` (a third door shape) | `DcdReader::open(path)` |
| 69 | `io::mrec` (Rust) | `read_frame_file`, `read_record_file`, `pack`, `open_packed`, `column_dtype`, `*_store` | `read_mrec_frame`, …, `pack_mrec_zip`, `open_mrec_zip` |
| 70 | Python io | `write_bond_react_map`; `read_amber_prmtop_ff`, `read_gromacs_top_ff`, `write_gromacs_top_ff`; `read_block_csv`; `io.log` | `write_lammps_bond_react_map`; `…_forcefield`; `read_csv_block`; `io.lammps_log` |
| 71 | `SmilesIR.write_smiles`, `write_fragment_smiles`, `write_atomistic_smiles` (return strings) | `write_` on non-file operations | `to_smiles` |
| 72 | cxx | `xyz_read_first_frame`, `read_first_frame`, `write_frame_xyz_typed` | `read_xyz_frame`, `read_mrec_frame`, `write_xyz_frame` |
| 73 | Python `ForceField.to_section` / `from_section` | mrec mapping hung on the ff model | `io.mrec.ForceFieldSection.from_forcefield` |
| 74 | `typifier/mmff/aromaticity.rs` | MMFF aromaticity perception inside the typifier | `perceive::aromaticity::mmff` |
| 75 | capi `molrs_block_col_commit` (no-op), wasm `Block.get`, capi `lib.rs:57` (pub mods + re-exports), `Style.types()` vs `get_types()`, `ff.scale_lj` module = function name | second paths | delete or merge |
| 76 | `ff/params/amber.rs:19,26` | `AMBER_SCEE`/`AMBER_SCNB` | `units::constants` (if they count as engine constants) |
| 77 | Python `molrs.units.AMBER_COULOMB` | only constant exposed | `molrs.core` constants submodule mirroring Rust |

## D. molpy (container-word names, reported by the molpy pass)

| Now | Proposed |
|---|---|
| `molpy.data` | `molpy.resources` |
| `engine/_base.py` | `engine/_engine.py` |
| `wrapper/_base.py` | `wrapper/_wrapper.py` |
| `adapter/_base.py` | `adapter/_adapter.py` |
| `wrapper/_env.py` | `wrapper/_environment.py` |
| `builder/_polymer/ambertools/types.py` | `_cut.py` / `_result.py` |
| `io/_metric.py` | `io/_metric_reader.py` |

## E. Open questions
- **E1.** Final names for the `BondType` clash (C1 #4) and the Python `Box`/`Graph` vs Rust `SimBox`/`MolGraph` pair.
- **E2.** Which abbreviations count as domain standard: `Mic`, `Atd`, `nerf`, `Parmchk`, `MmffBndk`/`Pbci`/`Chg`, `ac`/`prep`, `xcorr`, `ago`.
- **E3.** Whether `io.read_smiles` (from a string) and the format-dispatch doors `io.read_frame`/`write_frame`/`read_frame_bytes` are accepted exceptions under R1, and likewise `io.trajectory.TrajectoryReader`.
- **E4.** cxx renames need a matching change in Atomiverse.
- **E5.** `molrs._lib` → `_native` changes the pickle `__module__`, so old pickles won't load.

## F. User decisions (2026-10-07), binding
1. **Default:** anything not ruled on follows academic/community convention.
2. **Box:** `Box` everywhere; only Rust says `SimBox`, because Rust reserves `Box`.
3. **Acronyms:** PDB and similar are abbreviations, cased as words in CamelCase (`PdbReader`, `Msd`, `Rdf`, `LjCut`, `Mmff`, `OplsAa`, `Bcc`, `Uff`). Domain-standard abbreviations are allowed.
4. **No format dispatch:** delete the dispatching doors (`io.read_frame` / `write_frame`, `FrameFormat`). Every door names its format.
5. **String doors** are `read_<fmt>_str` / `write_<fmt>_str`; examples are `read_smiles_str`, `write_smiles_str` and `read_lammps_log_str`.
6. **Downstream:** rename the downstream repos in the same pass, including Atomiverse.

Applying decision 1:
- Graph is `MolGraph` everywhere.
- The chemical bond enum is `BondOrder`, and `compute::BondOrder` becomes `BondOrientationalOrder`.
- Bytes doors are `read_<fmt>_bytes` / `write_<fmt>_bytes`.
- The generic `TrajectoryReader` is deleted. Each `read_<fmt>_trajectory` returns `io.<fmt>.<Fmt>Reader`.
- `molrs._lib` becomes `_native`.
