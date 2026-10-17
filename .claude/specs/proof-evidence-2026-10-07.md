# Proof evidence, 2026-10-07 (molrs F = 7cde0ea6, before the S9 cosmetic wave)

## IR extensibility
- **molrs-ext-example, 7 tests passing.** A third-party crate adds styles without modifying molrs:
  - a pair `ScalarForm`, priced as LAMMPS lj/smooth/linear;
  - a new category, priced as LAMMPS Urey–Bradley;
  - FENE from its expression alone;
  - a round trip through mrec into a registry that has never seen the crate;
  - typed refusals;
  - the class2 bond-angle term (expression vs native form vs a central difference);
  - deck writing.
- **ff_ir_extension_lammps_check.sh, max relative error, energy/force:**
  - fene: Rust 1.3e-16/6.8e-16, Python 3.4e-16/7.0e-16.
  - lj/smooth/linear: Rust 1.7e-15/1.5e-15, Python 2.5e-15/2.3e-15.
  - urey_bradley: Rust 0/0, Python 0/1.6e-16.
  - bond_angle was skipped (this LAMMPS build lacks CLASS2).
- **molpy test_user_style.py, 6 tests passing.**
  - A 29-line snippet defines `class Fene(StyleDeclaration)` by its expression, uses a BeadSpring typifier in lj units, then compiles and writes `.mrec`.
  - Energy and forces equal the analytic sum (1e-12 / 1e-10).
  - A fresh process that registered nothing prices the record bit for bit.
- **molrs Python IR tests:** 54 (21 + 5 + 18 + 10).

## Protocol completeness and equivalence
- **Completeness matrix:** 53/53 rows pass. Of the 477 cells, 288 are exact, 71 are refused by name and 118 are not applicable.
- **ff_equivalence_check.sh** (worst values; GROMACS is 2025.3 `gmx_d`, plus sander from AmberTools 26.1):
  - native: E 2e-9, F 5e-9.
  - LAMMPS: E 9e-14, F 2e-5 (charmm36 fnorm2; the pinned tolerance).
  - OpenMM: E 8e-8 (chamber cmap), F 5e-11.
  - GROMACS: 5e-9.
- **Other engine checks:**
  - engine codec: worst 2.4e-15.
  - gromacs_engine_check: vs GROMACS 1.6e-9, vs LAMMPS 9.9e-9.
  - OpenMM XML: vs OpenMM 2.96e-12, vs LAMMPS 1.3e-8, written-vs-source XML 4.68e-15.
- **Rust test counts:** ir_invariance 3, builtin_conformance 3, engine_codec_check 10, equivalence_check 6.

## Module structure
- **module_boundaries, 13 tests.** They enforce:
  - ff names io only in tests;
  - perceive and io do not name each other;
  - op is a leaf;
  - the ff DAG ir < potential < style_registry < forcefield < compile < form_conversion;
  - params is a leaf;
  - the typifier does not name charge;
  - no unit-conversion constants or literals, and each factor is defined once.
- **test_public_paths:** molrs-python 119 tests, molpy 144.
- **Built wheel** (36 modules, 445 symbols): 0 duplicate exports, 0 `__module__` mismatches.
- **molrs.md.__all__** = MdDriver, Langevin, MdState, MaxwellBoltzmann, VelocityVerlet. No potentials.
- **molpy identity** (every name is the molrs object): core 178/178, io 174/174, ff 83/83 (8 in style_registry, 2 in compile), perceive 16/16.
- **Rust top-level graph:** 32 edges, acyclic. op and signal are leaves; core depends only on op.

## Full suites
- **molrs:** cargo 3812 + doctests 121, wasm 72, capi 21 + ctest, cxx 15, ffi 16, pytest 1228.
- **molrec:** 357, and ci-conformance 218/218.
- **molpy:** 789 pass with 3 skipped; with AmberTools loaded those 3 pass too.
- **molpack:** Rust 386 + 22, Python 142, parity 14/14 byte-identical.
- **Atomiverse:** cpu-unit 196/196, run at 83a187e6.
