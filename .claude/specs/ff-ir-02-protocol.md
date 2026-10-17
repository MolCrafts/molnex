# ff-ir-02-protocol

## Summary

Make the force-field IR (which adopts the LAMMPS standard) a **protocol**:
anything that has its form — a category, a style, an ordered parameter list
with dimensions, an energy given as an expression or as a kernel — extends
it, from Rust, from Python and from molpy, with no molrs rebuild, and
persists, prices and exports exactly like a built-in. The operator's goal,
verbatim: "证明现在molrs提供了具有可扩展、易泛化的forcefield ir，即便是在molpy这一层也可以让用户轻松扩展。换言之，ir只是protocol，符合这个形式的都能扩展。"

Why now: P2 (UB, CMAP, 1-4 tables) and wave 3 (GROMACS, OpenMM, prmtop)
closed the IR's *content*; its *shape* is still a set of closed match arms
(audit below). P3 (`canonical()`, `to_<form>()`, projection) would add three
more closed tables. Doing the protocol first turns P3's forms into
registrations and makes the projection work for any style.

Builds on `ff-ir-01-class-i-closure` (the closure, the exact-or-refuse
contract) and the molrec `forcefield` chapter (molrec branch
`ir/w3d-compat`, commit 16c4873: array parameters, compound expressions, the
custom-style rules, all landed with WP0).

## Goals

1. **G1 — a new style in an existing category**, from Rust or Python, by an
   expression, a vectorised scalar kernel, or a native constructor; priced at
   both compile doors (`compile`, `compile_typed`) and in MD.
2. **G2 — a new category** (arity 2–5, its own Frame block `<name>s`) priced
   by an expression over the positions of its atoms or a compound kernel;
   `ForceField.def_style` / `def_type` and the typifier `Match` accept it.
3. **G3 — persistence without code**: a custom style with an expression
   written to `.mrec` prices identically in a fresh process that registered
   nothing.
4. **G4 — engines by the parameter form**: LAMMPS coefficient lines derived
   from the parameter order and dimensions; OpenMM `Custom*Force` from the
   expression; every engine that cannot hold a custom style refuses it by
   name.
5. **G5 — the built-ins conform**: every built-in is a registration of the
   same form; those with an expression are checked against it.
6. **G6 — P3 on the protocol**: torsion forms and conversions are form-codec
   registrations; projection with residual works for any style.
7. **G7 — molpy**: a user declares a style in a class of < 30 lines, types a
   system with it, and prices, saves and reloads it.

## Non-goals

- Polarisable categories with their own state (Drude dynamics, AMOEBA
  multipoles), virtual-site *construction* (`virtual_site` stays a
  non-energy category), Class-II cross terms as built-ins (the optional
  `bond_angle` proof style is an extension, not a registry entry).
- One-body expressions (walls, position restraints: they need `x1, y1, z1`),
  per-term exclusion creation (`CategorySpec.excludes` is always `false`),
  and many-body terms beyond five atoms.
- Array parameters inside expressions (Lepton has no arrays); a style with
  an array parameter is priced by a Tier-2 or Tier-3 kernel.
- Persisting parameter dimensions in `.mrec` (an unregistered style read
  from a record has no `Dim`s; engines that need them refuse it until the
  style is registered).
- LAMMPS `*_style lepton` output (the installed LAMMPS has no LEPTON
  package); reading OpenMM `Custom*Force` as expression styles.
- Performance parity of Python kernels with native ones (one Python call per
  style per evaluation is the contract; production kernels are Tier 2 in
  Rust).

## Current state (audit, origin/dev d373c205)

The `(category, style)` kernel dispatch is the only part close to a protocol,
and only in Rust: `register_kernel` / `register_kernel_with` are `pub`
(`molrs/src/ff/potential/registry.rs:518-541`). A custom style in a built-in
category already round-trips through `to_section` / `from_section` as data,
`expression` kept as a string style param (`forcefield/section.rs:110-136`),
but nothing can price it afterwards. molpy keeps no parallel IR (it
re-exports molrs by identity), but its docs say the IR is closed.

Columns: does the item block (a) a new style in an existing category, (b) a
new category, (c) a new P3 form, (d) persistence, (e) engine I/O.

| # | What closes it | file:line | a | b | c | d | e |
|---|---|---|---|---|---|---|---|
| 1 | `StyleDefs` has 7 variants; `empty()` refuses other categories (`UnknownCategory`); `category()` is `&'static str` | `ff/forcefield/mod.rs:201-241` | – | **yes** | – | yes | – |
| 2 | arity table and `DefError::Arity{category:&'static str}` match the enum | `mod.rs:367, 438-459, 653` | – | **yes** | – | – | – |
| 3 | ~115 `StyleDefs::` arms in `type_rows`, `insert_type`, `type_params`, `set_type_*`, `rename_type`, `remove_type` | `mod.rs:535-990` | – | yes | – | – | – |
| 4 | no `register_typed` / `declare_rows`: a third-party pair style has no neighbour-driven form; built-ins silently overridable (a test asserts it) | `registry.rs:188-227, 498-562` | partly | – | – | – | – |
| 5 | no `ParamSource` gate; kernels carry no parameter schema, dimensions or expression | `registry.rs:1-30, 45` | partly | – | – | – | yes |
| 6 | category→block gate hard-coded; `category=="pair"` special cases | `potential/compile.rs:96, 183-186, 241, 257-266, 310` | – | **yes** | – | – | – |
| 7 | `category_arity` = molrec's table; unknown categories accepted with prefix arity (open) | `core/store/forcefield_section.rs:154-163` | – | ok | – | ok | – |
| 8 | only cmap `grid` may be an array column | `forcefield_section.rs:613` | yes | yes | – | **yes** | – |
| 9 | `style_table`: `category_arity(..).expect(..)` panics on other categories | `forcefield/section.rs:293-296` | – | yes | – | **yes** | – |
| 10 | LAMMPS reader: `supported_styles`, `coeff_params`, `molrs_style_name` closed (doctest asserts `bond fene` is refused) | `readers/lammps.rs:514-540, 1101-1240` | **yes** | yes | – | – | **yes** |
| 11 | LAMMPS writer: `coeff_fields`, `lammps_style_name` closed; units converted per arm, not per dimension | `writers/lammps.rs:316-477` | **yes** | yes | – | – | **yes** |
| 12 | GROMACS `bonded_columns`, frcmod, OpenMM XML writer: per-style arms | `writers/gromacs.rs:196-320`, `frcmod.rs:271-330`, w3b `xml.rs:189-301, 439-639` | yes | yes | – | – | **yes** |
| 13 | `TorsionForm` closed enum; `from_params` / `from_series` closed | `forcefield/torsion.rs:849, 904, 1102` | – | – | **yes** | – | – |
| 14 | Python `Category`: 7 handle classes; `def_style` / `get_style` raise for others; `style_handles` silently drops others (`.ok()?`) | `molrs-python/src/ff/handles.rs:20-160`; `ff/mod.rs:1349, 1566-1594` | – | **yes** | – | – | – |
| 15 | Python has no kernel registration; only the whole-system `SubclassPotential` | `molrs-python/src/md.rs:395-490` | **yes** | **yes** | yes | – | – |
| 16 | typifier `Match` has fixed `bonds, angles, dihedrals, impropers` | `ff/typifier/mod.rs:110-118, 216-222`; `molrs-python/src/ff/mod.rs:440-448` | – | **yes** | – | – | – |
| 17 | molpy `potential/__init__.py` imports classes removed in 0.15; `extending-forcefield.md` says "a molrs change … rebuild the wheel"; `engine/lammps._style_lines` loops over 4 kinds | molpy | yes | yes | – | – | yes |
| 18 | molrec dev's registry still ½k; only `ir/w3d-compat` adopts the LAMMPS standard and defines expressions | molrec `docs/spec/forcefield.md` | – | – | – | yes | – |

Already open, kept: `ForceField::def_style` accepts any style name in a
built-in category; the Frame block set is open; `MolGraph::register_kind` is
in Python; Python `Typifier.match` can be subclassed. No expression engine
exists (the only parser is `core/units/parse.rs`).

## The protocol (normative)

The vocabulary lives in `molrs/src/ff/ir/` (`param_dimension.rs`,
`category.rs`, `spec.rs`, `params.rs`, `error.rs`, `form/`, `engine_codec/`,
`expression/`), re-exported from `molrs::ff::ir`; the registry that binds a
style to its kernel (§3, §5: `Registry`, `Kernel`, the conformance checks,
`BuiltinKernels`, the expression bridge `compile_expression`) is
`molrs::ff::style_registry`, and the compiler (`PotentialCompiler`,
`ExplicitTerms`) is `molrs::ff::compile`. `ff`'s submodules are layers, each
naming only those beneath it (checked by `module_boundaries`): `ir` <
`potential` < `style_registry` < `forcefield` < `compile` < `form_conversion`.
`F` is the crate float (`f64`).

### 1. Categories

```rust
pub enum Arity { Exact(u8), SelfOrPair }                       // Exact(0..=5)
pub enum Coordinate { None, Distance, Angle, Dihedral, Improper, Compound }
pub enum EndpointOrder { Reversible, Ordered, Unordered }
pub struct CategorySpec {
    pub name: Cow<'static, str>,      // ^[a-z][a-z0-9_]*$
    pub arity: Arity,
    pub block: Cow<'static, str>,     // the Frame block it prices
    pub coordinate: Coordinate,
    pub order: EndpointOrder,         // how assign_terms matches endpoints
    pub excludes: bool,               // always false (non-goal)
}
impl CategorySpec { pub fn custom(name, arity: u8, coordinate: Coordinate, order: EndpointOrder) -> Self; } // block = name + "s"
```

Built-ins (registered by `Registry::builtin()`, **sealed**):

| name | arity | block | coordinate | order | priced |
|---|---|---|---|---|---|
| `atom` | Exact(0) | `atoms` | None | – | no |
| `bond` | Exact(2) | `bonds` | Distance | Reversible | yes |
| `angle` | Exact(3) | `angles` | Angle | Reversible | yes |
| `dihedral` | Exact(4) | `dihedrals` | Dihedral | Reversible | yes |
| `improper` | Exact(4) | `impropers` | Improper | Ordered | yes |
| `pair` | SelfOrPair | `atoms` (via `type`) | Distance | Unordered | yes, neighbour-driven |
| `cmap` | Exact(5) | `cmaps` | Compound | Reversible | yes |
| `drude` | Exact(2) | `drudes` | Distance | Ordered | spec only (no kernel) |
| `constraint` | Exact(2) | `constraints` | None | Reversible | no |
| `virtual_site` | Exact(0) | `virtual_sites` | None | – | no |

Rules:
- A **custom** category has `Exact(2..=5)`, block exactly `<name>s` (the
  molrec rule, so a record alone locates it), coordinate `Compound` unless
  the user picks one of `Distance` (arity 2), `Angle` (3), `Dihedral` /
  `Improper` (4) to get that geometric variable. `register_category` refuses
  anything else.
- The **only** remaining special case keyed on a category is `SelfOrPair`
  (neighbour-driven pricing); `compile.rs` gates by `CategorySpec.block`,
  never by name.
- An **unregistered** category met in a record or `ForceField` behaves as
  `CategorySpec::custom(name, arity_from_endpoint_columns, Compound,
  Reversible)`: preserved always; priced when its block has rows and its
  style has an expression; `IrError::NoKernel` otherwise. Arity < 2: never
  priced (molrec: no energy).

### 2. Styles and parameters

```rust
pub struct ParamDimension { pub energy: i8, pub length: i8, pub angle: i8, pub charge: i8, pub mass: i8 }
pub enum ParamKind { Scalar, Array { rank: u8 }, Text { choices: Option<Vec<Cow<'static, str>>> } }
pub enum ParamValue { Num(F), Text(Cow<'static, str>) }
pub enum ParamCombination { None, Arithmetic, Geometric, LjEpsilon { sigma: Cow<'static, str> }, LjSigma { epsilon: Cow<'static, str> } }
pub struct ParamSpec {
    pub name: Cow<'static, str>, pub dim: ParamDimension, pub kind: ParamKind,
    pub default: Option<ParamValue>, pub mix: ParamCombination, pub indexed: bool,
}
pub enum LammpsForm { None, Positional { name: Option<Cow<'static, str>> }, Custom(Arc<dyn EngineCodec>) }
pub struct StyleSpec {
    pub category: Cow<'static, str>, pub name: Cow<'static, str>,
    pub params: Vec<ParamSpec>,            // ORDERED: = LAMMPS *_coeff order when a LAMMPS style exists
    pub style_params: Vec<ParamSpec>,      // cutoff, mixing, special keep their reserved meanings
    pub source: ParamSource, pub special: Option<SpecialClass>,   // special: pair styles only, default lj
    pub expression: Option<String>,        // byte-exact Lepton statement of the energy
    pub force_is_gradient: bool,           // false only for coul/charmm (LAMMPS's switched force)
    pub unindexed_one_term: bool,          // dihedral periodic / improper trefoil: k, periodicity, phase = term 1
    pub lammps: LammpsForm,
    pub samples: Vec<ConformanceSample>,              // optional registration samples (§5)
}
```

- **ParamDimension string syntax** (`ParamDimension: FromStr + Display`):
  ```text
  dim    := num ("/" factor)*
  num    := "1" | factor ("*" factor)*
  factor := sym ["^" pos]            pos := [1-9][0-9]*
  sym    := "E" | "L" | "A" | "Q" | "M"     energy, length, angle, charge, mass
  ```
  Examples: `1`, `E`, `E/L^2`, `1/L`, `E*L^6`, `L^3`, `E*L/Q^2`, `A`,
  `E/A^2`, `M`, `Q`. Each symbol at most once; `^0` and negative exponents
  are spelled with `/`. `Display` prints numerator symbols in the order
  E L A Q M joined by `*` (or `1`), then each denominator as `/sym[^n]` in
  the same order; parsing the printed form gives the same `ParamDimension`.
- **Angle semantics of `A`**: `dim == A` (exactly `angle = 1`, nothing else)
  is an **angle value**, stored in `units.angle` (the degree in every
  preset). A **negative** angle exponent is per **radian** and never
  converted. Any other positive angle exponent is refused (`IrError::Dimension`).
  Unit conversion of a parameter multiplies by
  `f_E^e · f_L^l · f_Q^q · f_M^m` (the preset ratios); `A` converts only
  between `units.angle` values (identity in every preset).
- **Names**: a parameter name matches `^[A-Za-z_][A-Za-z0-9_]*$` (a Lepton
  identifier). Reserved, refused as parameter names: `name`, `type`,
  `style`, `itom`…`mtom`, `atomi`…`atomm`, the annotation columns (`class`,
  `element`, `smarts`, `smirks`, `overrides`, `desc`, `doi`), the
  geometric variables `r theta phi chi`, `p1`…`p5`, and
  in pair styles `q`, `q1`, `q2` and any `x1`/`x2` where `x` is another
  parameter of the style. Style-param names reserved with fixed meaning:
  `cutoff` (`L`, Scalar), `mixing` (Text: `arithmetic|geometric|sixthpower`,
  absent = `arithmetic`, molrs's undeclared default), `special` (Text:
  `lj|coul`). Function names are not reserved: a call always carries its
  `(`, so `delta` (`pair coul/cut`'s style param) never collides with
  `delta(x)`.
- **Indexed** parameters: `indexed: true` on `k` means columns `k1, k2, …,
  kM`, contiguous from 1, one `M` per table shared by all indexed parameters
  of the style (a gap is refused, as for `nharmonic`). `unindexed_one_term`
  additionally accepts the bare spelling as term 1 (never both).
- **Array** parameters: `f64[T, S…]` columns of uniform shape (molrec);
  `rank` fixes `len(S…)`; the kernel validates the extents (cmap: square,
  N ≥ 2). Array and Text parameters are not expression variables.
- **Mixing** (pair styles only): a bare-name pair value is the cross row's
  non-null cell, else `ParamCombination` of the two self rows: `Arithmetic` ½(a+b),
  `Geometric` √(ab), `LjEpsilon`/`LjSigma` the joint (ε, σ) rule the style's
  `mixing` names (sixthpower needs both); `ParamCombination::None` makes an unlike pair
  without a cross row an error at compile (`IrError::NoMixing`). For an
  **unregistered** pair style, `epsilon`/`sigma` (and
  `epsilon14`/`sigma14`) mix as `LjEpsilon`/`LjSigma`, every other parameter
  is `ParamCombination::None`.

### 3. Kernels: three tiers

```rust
pub trait ScalarForm: Send + Sync + 'static {   // Tier 2: one call per style per evaluation
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de_dq: &mut [F]);
}
pub trait CompoundForm: Send + Sync + 'static {
    fn eval(&self, x: &[[F; 3]], arity: usize, p: &ParamColumns<'_>, e: &mut [F], grad: &mut [[F; 3]]);
}
pub enum Kernel {
    Expression(Arc<expression::Compiled>),                       // Tier 1
    Scalar(Arc<dyn ScalarForm>), Compound(Arc<dyn CompoundForm>),   // Tier 2
    Ctor { compiled: KernelConstructor,                    // Tier 3 = today's constructors
           typed: Option<(KernelConstructor, SpecialClass)>, rows: RowSource },
}
```

Calling conventions (every tier):
- `q` per term: `Distance` r in `units.length`; `Angle` θ ∈ [0, π] rad;
  `Dihedral` φ ∈ (−π, π] rad, IUPAC sign (molrs `compute_dihedral`), atoms
  in row order; `Improper` the same φ of the listed order (a kernel wanting
  χ takes |φ|). `x` is `n_terms × arity` positions, minimum-imaged
  relative to the term's first atom.
- **Parameters arrive exactly as stored** (IR units; angle values in
  degrees) — no tier converts them. `ParamColumns` gives each per-type scalar
  as an `n_terms` column (`get(name) -> Option<&[F]>`), arrays as
  `array(name) -> Option<ArrayViewD>` with leading axis `n_terms`, text as
  `text(name) -> Option<&[&str]>`, numeric style params broadcast to
  `n_terms` columns, text style params via `style_text(name)`. Pair:
  bare names are the resolved pair values (§2 mixing), plus `q1`, `q2` when
  the frame has `atoms.charge`; Tier 2 does not see `x1`/`x2` (those are an
  expression feature).
- Outputs: `e[i]` the **unweighted** energy of term i, `de_dq[i]` its
  derivative (or `grad` = ∂E/∂x, not the force). The form kernels apply
  the pair special weight / pair-override scale and `r < cutoff`, then
  chain-rule to forces through `flat_coords.rs`, and add the virial
  (`−Σ (dE/dr) r⊗r / r` for pairs).
- Form kernels in `ff/potential/form_kernel/`: `ScalarBonded`
  (`IndexedTerms` over `CategorySpec.block`), `ScalarPair` (`PairDriven`;
  both compile doors, per-param mixing and cross rows, `cutoff`, special
  weight, the pair-override path), `CompoundTerms` (any block). They replace
  the boilerplate a third party would copy from `pair/morse.rs:308-412`.
- Python callables (wrapped like `SubclassPotential`: `Python::attach` + an
  `ErrSlot`, the exception re-raised from `calc_energy`/`calc_forces`):
  scalar `f(q: ndarray[n], **p) -> (e[n], de_dq[n])`; compound
  `f(x: ndarray[n, arity, 3], **p) -> (e[n], grad[n, arity, 3])`; numeric
  params as `ndarray[n]` (arrays `[n, S…]`), text style params as `str`,
  text per-type params as `list[str]`. A wrong dtype/shape/length is
  `IrError::KernelShape`.
- The built-in constructors are a crate-private table (`BuiltinKernels`)
  seeding the one registry as `Kernel::Constructor`s.

### 4. Expressions

Grammar, functions, definitions and variable binding are **exactly** molrec
`docs/spec/forcefield.md` § Expressions (`ir/w3d-compat` 16c4873); the
summary the implementers need:

- Lepton subset: `+ - * / ^`, parentheses, numbers `1`, `1.5`, `.5`,
  `1e-3`; precedence `^` (right-assoc) > unary `-` > `* /` > `+ -`
  (`-a^2 = -(a^2)`, `a^-2 = a^(-2)`); functions `exp log sqrt sin cos tan
  asin acos atan abs` (1 arg), `min max` (2), `step` (x ≥ 0 → 1), `delta`
  (x = 0 → 1), `select(x,y,z)` (x ≠ 0 → y); no named constants.
- Definitions `; name=formula` after the energy; a definition may use only
  names defined to its right (Lepton's order); redefinition, self/left
  reference, or shadowing a variable is an error.
- Compound functions `distance(pa,pb)` (L), `angle(pa,pb,pc)` (rad, [0, π]),
  `dihedral(pa,pb,pc,pd)` (rad, (−π, π], IUPAC) over points `p1`…`pA`, A the
  category's arity; available in every category with points (not `pair`).
- Variables per category: bond/drude `r`; angle `theta`; dihedral `phi`;
  improper `phi` and `chi = abs(phi)`; cmap and custom categories points
  only; pair `r`, `q1`, `q2`, bare `x` (pair value), `x1`/`x2` (self rows);
  every numeric per-type and style parameter by name, **as stored**.
- **Angle-valued parameters are seen in degrees**; the expression converts
  (`theta0*0.017453292519943295`). Decision recorded in §Decisions D3.
- The expression is the unweighted, untruncated per-term energy (pairs:
  weight and `r < cutoff` applied outside). A pair expression must be
  symmetric under 1 ↔ 2.

Engine (`ff/ir/expression/`): `lexer`, `parser` → AST, `check` (binding,
arity, definition order), `dual` (forward-mode `Dual<const N>`), `eval`,
`print` (only for engine rewriting; the stored string is never
re-printed).
- `expression::compile(src, &Binding) -> Result<Compiled, IrError>`; `Binding`
  = category coordinate + arity + parameter names/kinds (+ pair flag).
- Scalar evaluation: N = 1 dual on `q`. Compound evaluation (any use of a
  point function, or a `Compound` category): N = 3·arity duals on `x`, the
  geometric variable defined as its compound function. So dE/dq and ∂E/∂x
  are exact; nobody hand-writes a derivative.
- Derivatives of the non-smooth functions: `step`, `delta` 0; `abs` sign(x)
  (0 at 0); `min`/`max` the chosen argument's (ties: first); `select` the
  chosen branch's; `angle`/`dihedral` at a degenerate geometry (collinear
  points) value as computed, gradient 0.
- **Compile fallback**: a `(category, style)` with no registered kernel but
  an `expression` (on the style instance, from the record) compiles to an
  expression kernel automatically. Priority: registered kernel > style's
  `expression` > `IrError::NoKernel { category, style }` whose message is
  "no kernel for {category} `{style}`: register it
  (molrs.ff.style_registry.register_style) or give it an expression".

### 5. Registration and conformance

```rust
impl Registry {
    pub fn builtin() -> Self;                         // every built-in, sealed
    pub fn register_category(&mut self, c: CategorySpec) -> Result<(), IrError>;
    pub fn register_style(&mut self, s: StyleSpec, k: Option<Kernel>) -> Result<(), IrError>; // None: expression only
    pub fn unregister_style(&mut self, category: &str, name: &str) -> Result<(), IrError>;   // custom only
    pub fn register_form(&mut self, category: &str, style: &str, f: FormCodec) -> Result<(), IrError>;
    pub fn register_engine_form(&mut self, e: Engine, category: &str, style: &str, c: Arc<dyn EngineCodec>) -> Result<(), IrError>;
    pub fn category(&self, name: &str) -> Option<&CategorySpec>;
    pub fn style(&self, category: &str, name: &str) -> Option<(&StyleSpec, Option<&Kernel>)>;
}
pub fn register_category(..); pub fn register_style(..); /* … */   // the same on the global RwLock<Registry>
// all of the above: molrs::ff::style_registry
impl PotentialCompiler<'_> { pub fn with_registry(ff: &ForceField, r: &Registry) -> Self; } // molrs::ff::compile; default: global
```

`register_style` with an expression and no kernel registers a Tier-1 style;
with both, the expression must agree with the kernel. Each refusal is one
`IrError` variant (Python: a class of the same name, all subclasses of
`molrs.ff.ir.IrError(ValueError)`), naming the offending item:

| Variant | When |
|---|---|
| `UnknownCategory { category }` | style in a category neither built-in nor registered |
| `BadName { what, name }` | category/param name outside its pattern |
| `Arity { category, arity }` | custom arity outside 2..=5, or `def_type` with the wrong number of endpoints |
| `BlockName { category, block }` | custom category whose block is not `<name>s` |
| `ReservedParam { style, param }` / `DuplicateParam { style, param }` | §2 names |
| `Dimension { param, dim, reason }` | unparsable or forbidden `ParamDimension` |
| `Parse { expression, at, reason }` | grammar error, definition order |
| `UnboundVariable { style, name }` | a free name that is neither a variable of the category nor a declared numeric param (e.g. `theta` in a bond) |
| `UnknownFunction { name }` / `FunctionArity { name, given, expected }` | |
| `Point { style, point, arity }` | `pk` beyond the arity, or any point in a pair style |
| `CoordinateMismatch { category, kernel }` | `Scalar` kernel on a `Compound`/`None` category, `Compound` kernel declared for a pair |
| `Derivative { style, at, rel }` | a Tier-2 kernel's `de_dq`/`grad` vs central difference, rel > 1e-6 |
| `Disagree { style, at, rel }` | expression vs kernel energy (and force when `force_is_gradient`), rel > 1e-10 |
| `Asymmetric { style }` | pair expression/kernel not symmetric under 1 ↔ 2 (rel > 1e-12) |
| `Sealed { category, style }` | re-registering or unregistering a built-in |
| `Conflict { category, style }` | a different re-registration (identical: no-op; Python `replace=True` overrides a custom one) |
| `NoKernel { category, style }` | compile, §4 |
| `NoMixing { style, param, pair }` | compile, §2 |
| `MissingParam { style, type_, param }` | compile: a row null where the kernel needs it, no default |
| `KernelShape { style, reason }` | Tier-2 output of the wrong shape/dtype, or a Python kernel raising (chained) |
| `NoEngineForm { engine, category, style, reason }` | §8 |
| `FormConflict { family, reason }` / `NoForm { category, style }` | §9 |

**Checks at registration and first use.** Derivative and agreement checks
run at registration on `StyleSpec.samples` (`ConformanceSample { params: Vec<(name,
ParamValue)>, q: (lo, hi) }`, 16 seeded points each, seed `0x1F1F_0002`); a
style registered without samples is checked once per process at its first
compile on up to 8 real terms of the frame. Central difference with
`h = 1e-5·max(1, |q|)`, points within `h` of a domain edge skipped.
Relative error is measured against `max(|value|, RMS of |values| over the
sample set)`. The checks never run per evaluation.

**Built-in gate (G5).** `ff::ir::builtin_conformance` registers every
built-in from its `StyleSpec` and, for each with an expression (Appendix A),
checks agreement (energy rel 1e-10; force too unless `force_is_gradient =
false`) on 64 seeded configurations × parameter sets per style, including
both compile doors for pair styles; it also checks every built-in's
`ParamSource` declaration against what its constructor reads (the missing
ParamSource gate). This proves the built-ins are registrations of the same
form.

### 6. `ForceField` and compile

- `StyleDefs::Relation { category: Arc<str>, arity: u8, types:
  Vec<RelationType { name, endpoints: SmallVec<[String; 5]>, params }> }`
  beside the 7 built-in variants; `Style::category() -> &str`; `DefError`
  carries `String` categories.
- `def_style(category, style, params)` takes the arity from the registry,
  else (an unregistered category from `from_section`) from the endpoint
  columns. `def_type` checks arity (`IrError::Arity`).
- `compile` / `compile_typed` iterate styles, resolve `(CategorySpec,
  kernel)` through the registry (`with_registry` or global), gate on
  `CategorySpec.block`; no `category == "pair"` string tests remain
  (`SelfOrPair` and the `special_bonds`/override machinery are keyed on the
  arity and `SpecialClass`).
- Typifier: `TypeAssignment(links=...)` accepts any registered relation kind, by
  `MolGraph` kind name or class; the category is the one whose `block`
  equals the kind. `assign_terms` matches endpoints by `EndpointOrder` and
  molrec's wildcard rule (fewest wildcards wins, then table order).

### 7. Persistence

- A custom style persists as its molrec style entry: style `params`, the
  `expression` (from the instance, else the registry's — `to_section` writes
  the registry expression of a registered custom style so a fresh process
  can price it; a differing instance expression is checked for agreement at
  first compile), and its table (array columns allowed, molrec 16c4873).
  Built-in styles write no expression.
- `check_style_table` (`forcefield_section.rs:613`) allows any `f64[T, S…]`
  parameter column with molrec's rules; `style_table` (`section.rs:296`)
  takes arity from the registry or the endpoint columns — never panics.
- Reading a style that is unregistered and has no expression succeeds;
  compiling it is `NoKernel`. Reading never evaluates an expression.
- WP7 bumps molrec's pin in `partners.env` to the merged WP0 commit.

### 8. Engine codecs and refusals

```rust
pub enum Engine { Lammps, OpenmmXml, Gromacs, AmberPrmtop, AmberFrcmod }
pub trait EngineCodec: Send + Sync + 'static { /* write/read one style's lines; WP8 defines the item types */ }
```

- **LAMMPS `Positional`** (derived from the spec): `<cat>_style <name>`
  (`pair_style <name> [cutoff]`), `<cat>_coeff <type> v₁ … vₙ` in `params`
  order, each value converted by its `ParamDimension` from the field's units to the
  target `units` (angle values unchanged, per-radian constants unchanged),
  `mixing` → `pair_modify mix`. Refused (`NoEngineForm`): a Text or Array
  parameter, an indexed parameter, a null cell without a default, a style
  param other than `cutoff`/`mixing`. Reading is the inverse: exactly `n`
  numbers, else an error naming the line. `fourier` (indexed),
  `nharmonic`, `lj/charmm` (`one_four`, two cutoffs), `cmap` keep `Custom`
  codecs; every other built-in uses `Positional`. An expression-only style
  is refused ("LAMMPS has no expression engine here: no LEPTON package").
- **OpenMM XML**: an expression style (registered or unregistered) is
  written as the `Custom*Force` of its category — `CustomBondForce` (`r`),
  `CustomAngleForce` (`theta`), `CustomTorsionForce` (`theta` = our `phi`;
  `chi` → `abs(theta)`), `CustomCompoundBondForce` (any point function, any
  `Compound` category; same function names), `CustomNonbondedForce` (pair:
  `x1`/`x2` per-particle; bare `epsilon`/`sigma` rewritten as the mixing
  formula of `epsilon1, epsilon2, …`; a cross row or another bare name →
  refused; `q1*q2` from a per-particle `q`). Parameter values stay in IR
  units as `PerBondParameter`/`GlobalParameter`; the printer rewrites the
  expression so the conversion is exact:
  `4.184*(E[r→r_ir]); r_ir=10*r` (and `10*distance(...)` for point
  functions), with the actual factors from the field's units. Reading
  `Custom*Force` stays refused (molrec).
- **GROMACS, AMBER prmtop, frcmod**: refuse every non-built-in style.
- Every refusal is `IrError::NoEngineForm { engine, category, style, reason }`.
- molpy's `engine/lammps._style_lines` and `io/emit/lammps` call the molrs
  LAMMPS codec (no own loop over 4 kinds).

### 9. P3 on the protocol: form codecs

```rust
pub struct FormCodec {
    pub family: Cow<'static, str>,          // e.g. "torsion"
    pub canonical: bool,                     // exactly one per family
    pub embed:   fn(&TypeParams) -> Result<TypeParams, FormRefusal>,  // this style → the canonical style's params (exact)
    pub project: fn(&TypeParams) -> Result<TypeParams, FormRefusal>,  // canonical → this style, exact on its image, else FormRefusal naming the condition
}
```

- `project` is the **exact** inverse with refusal (ff-ir-01's `to_<form>`);
  the least-squares projection with residual is a separate operation,
  `ForceField::fit_form(category, style, metric) -> (ForceField, FitResidual)`,
  generic over any style because it needs only `E(q)` (expression, Python
  and native styles alike).
- Torsions: family `torsion`, canonical `dihedral periodic` (via
  `FourierSeries::canonical`); the 9 `TorsionForm` arms become 9
  registrations (`improper harmonic` is no Fourier series and gets no
  codec).
- `ForceField::canonical()` maps every style of a family **in the canonical
  style's category** onto it (type-name collisions refused by name); styles
  of the family in another category (impropers) stay. `to_form(category,
  style)` converts within one category through the canonical parameters.
  A family without a canonical, or two canonicals, is `FormConflict`; a
  style without a codec is `NoForm`.

### 10. Python: `molrs.ff.ir` and `molrs.ff.style_registry`

The vocabulary (`ParamSpec`, `StyleSpec`, `CategorySpec`, the `IrError`
classes) is `molrs.ff.ir`; registering and introspecting is
`molrs.ff.style_registry` (`molrs-python/src/ff/ir/`,
`molrs-python/src/ff/style_registry.rs`).

```python
from molrs.ff import ir, style_registry

ir.ParamSpec(name, dim="1", *, kind="scalar"|"array"|"text", rank=None, choices=None,
         default=None, mix=None, indexed=False)
style_registry.register_category(name, arity, *, coordinate="compound", order="reversible")   # block is f"{name}s"
style_registry.register_style(category, name, *, params, style_params=(), expression=None, kernel=None,
                              compound=False, special=None, lammps=None|"positional", samples=None, replace=False)
class Fene(style_registry.StyleDeclaration):                       # __init_subclass__ registers
    category = "bond"; name = "fene"
    params = [ir.ParamSpec("k", "E/L^2"), ir.ParamSpec("r0", "L"), ir.ParamSpec("epsilon", "E"), ir.ParamSpec("sigma", "L")]
    expression = "-0.5*k*r0^2*log(1-(r/r0)^2) + step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)"
    lammps = "positional"
    # optional: def kernel(self, q, **p) -> (e, de_dq)  (or x for compound)
style_registry.styles(category=None) -> list[ir.StyleSpec]; style_registry.categories() -> list[ir.CategorySpec]; style_registry.unregister_style(category, name)  # custom only
style_registry.evaluate(category, name, q=None, *, x=None, **params) -> (e, de_dq | grad)
ir.IrError (ValueError) and one subclass per Rust variant, named <Variant>Error (NoEngineFormError, NoKernelError, SealedError, DimensionError, …)
```

`params` also accepts a `{name: dim}` dict (all Scalar). `kernel` is a
scalar callable unless `compound=True` or the category's coordinate is
`Compound`. `ForceField.def_style` on a custom category returns a generic
`RelationStyle` whose `def_type(name, *endpoints, **params)` is variadic;
`style_handles` returns every category (no silent drop). The `.pyi` lists
all of it.

### 11. molpy

No parallel IR. `molpy.potential` is an identity re-export of `Potential`
and `molrs.ff.ir.{ParamSpec, StyleDeclaration, register_style, register_category,
styles}` (fixing the stale 0.15 imports); `_style_lines` comes from the molrs
LAMMPS codec; `docs/developer/extending-forcefield.md` is rewritten as "no
rebuild".

## Appendix A: built-in expressions

`D` below is `(pi/180)`, parenthesised. The grammar has no named constants
(§4; Lepton has none), so an expression that writes `D` defines `pi` as its
last sub-definition, `; pi=3.141592653589793` (the shortest decimal of
`f64::consts::PI`), as Lepton users do: `angle harmonic` is
`k*(theta-theta0*(pi/180))^2; pi=3.141592653589793`. Parenthesised,
`(pi/180)` evaluates to `0.017453292519943295`, the factor of
`f64::to_radians`, so `theta0*(pi/180)` is `theta0.to_radians()` to the bit
(`theta0*pi/180` rounds differently). The rows below omit the trailing
`; pi=…`. The gate checks every row with an expression; "native only" rows
say why.

| Style | Expression |
|---|---|
| `bond harmonic` | `k*(r-r0)^2` |
| `bond morse` | `d0*(1-exp(-alpha*(r-r0)))^2` |
| `bond class2` | `k2*d^2+k3*d^3+k4*d^4; d=r-r0` |
| `angle harmonic` | `k*(theta-theta0*D)^2` |
| `angle charmm` | `k*(theta-theta0*D)^2+k_ub*(distance(p1,p3)-r_ub)^2` |
| `angle class2` | `k2*d^2+k3*d^3+k4*d^4; d=theta-theta0*D` |
| `dihedral periodic` | generated per table: `Σₘ k<m>*(1+cos(periodicity<m>*phi-phase<m>*D))` (one-term spelling: `k`, `periodicity`, `phase`) |
| `dihedral charmm` | `k*(1+cos(periodicity*phi-phase*D))` (the `w` 1-4 pair is outside the term; gate at `w = 0`) |
| `dihedral opls` | `0.5*(k1*(1+cos(phi))+k2*(1-cos(2*phi))+k3*(1+cos(3*phi))+k4*(1-cos(4*phi)))` |
| `dihedral harmonic` | `k*(1+sign*cos(periodicity*phi))` |
| `dihedral multi/harmonic` | `a1+a2*c+a3*c^2+a4*c^3+a5*c^4; c=cos(phi)` |
| `dihedral nharmonic` | generated per table: `Σᵢ a<i>*c^(i-1); c=cos(phi)` |
| `dihedral class2` | `k1*(1-cos(phi-phi1*D))+k2*(1-cos(2*phi-phi2*D))+k3*(1-cos(3*phi-phi3*D))` |
| `dihedral rb` (molrec registry; molrs registers it **by expression only**, Tier 1 — the first built-in that is pure protocol) | `c0+c1*c+c2*c^2+c3*c^3+c4*c^4+c5*c^5; c=-cos(phi)` |
| `improper harmonic` | `k*(chi-chi0*D)^2` |
| `improper cvff` | `k*(1+sign*cos(periodicity*phi))` |
| `improper periodic` | `k*(1+cos(periodicity*phi-phase*D))` |
| `improper trefoil` | generated: ⅓ Σ over the three orderings the molrs kernel uses, each as `dihedral(…)` of the points |
| `pair lj/cut` | `C*epsilon*((sigma/r)^n-(sigma/r)^m-select(shift,(sigma/cutoff)^n-(sigma/cutoff)^m,0)); C=n/(n-m)*(n/m)^(m/(n-m))` (style params `n`=12, `m`=6, `shift`=0 by default) |
| `pair lj/charmm` | `4*epsilon*((sigma/r)^12-(sigma/r)^6)*S; S=select(step(inner-r),1,(cutoff^2-r^2)^2*(cutoff^2+2*r^2-3*inner^2)/(cutoff^2-inner^2)^3)` |
| `pair coul/charmm` | `coulomb*q1*q2/(dielectric*r)*S; S=…` as above (`force_is_gradient = false`: energy only) |
| `pair coul/cut` | `coulomb*q1*q2/(dielectric*(r+delta))` |
| `pair lj/class2` | `epsilon*(2*(sigma/r)^9-3*(sigma/r)^6)` |
| `pair buck` | `a*exp(-r/rho)-c/r^6` (`ParamCombination::None`) |
| `pair morse` | `d0*((1-exp(-alpha*(r-r0)))^2-1)` (`ParamCombination::None`) |
| `drude harmonic` | `k*r^2` (spec only; no kernel) |
| native only | `cmap charmm` (array grid), `pair coul/long/pme` (Ewald), `pair thole` and `pair coul/tt` (the Coulomb constant is the units', not a parameter), every `mmff_*` / `uff_*` (molrs's own per-instance forms) |
| no energy | `atom full`, `constraint fixed`, `virtual_site *` |

`ParamSource`/`ParamDimension` of every built-in: the molrec registry table (its
dimension column) and the molrs IR guide; WP1 transcribes them.

## The proof (acceptance criteria)

### P-Rust: `molrs-ext-example/` (standalone workspace root, depends on molrs `features=["full","filesystem","stream"]`, `pub` API only)

`src/lib.rs`: `pair lj/smooth/linear` as a `ScalarForm` (~25 lines); custom
category `urey_bradley` (arity 3, block `urey_bradleys`) with a native
`CompoundForm` `harmonic(k_ub, r_ub)`; `bond fene` by expression.
`tests/proof.rs`:

| Test | Criterion |
|---|---|
| `pair_style_matches_lammps` | E and every force component vs pinned LAMMPS `lj/smooth/linear` numbers, rel ≤ 1e-10; `compile_typed` = `compile` rel ≤ 1e-12 |
| `new_category_matches_lammps` | vs LAMMPS `angle_style charmm` with K = 0 (pure 1-3 UB), rel ≤ 1e-10 |
| `fene_matches_lammps` | vs LAMMPS `bond_style fene`, rel ≤ 1e-10 (geometries with (r/R0)² < 0.9) |
| `mrec_round_trip` | energy bit for bit after write/read; with `Registry::builtin()` only, the expression styles still price (bit for bit vs before) and the native-only style is `NoKernel` naming it |
| `nonconforming_refused` | every `IrError` variant of §5 reachable from registration/compile, each asserted by variant and named item |

New gate `ext` in `scripts/check.sh` and `.pre-commit-config.yaml` (hooks
run it on compute nodes).

### P-LAMMPS

`scripts/ff_ir_extension_lammps_check.sh` (as `lammps_cmap_check.sh`): with
`MOLRS_FFEXT_DIR` set the test writes the deck through molrs's LAMMPS writer
(Positional codecs), `lmp run 0`, `thermo_modify format float %.17g`, a
forces dump; the test pins those numbers (energy and forces rel ≤ 1e-10).
The installed lmp has `fene`, `lj/smooth/linear`, `angle charmm`, no
CLASS2: the optional `bond_angle` (class2 `ba`) cross term is checked by
hand value + finite differences + expression = numpy = Rust (rel ≤ 1e-12),
and against LAMMPS only when `MOLRS_LMP_CLASS2` names a CLASS2 build.

### P-Python: `molrs-python/tests/test_ff_ir_extension.py`

| Test | Criterion |
|---|---|
| fene by expression = analytic formula | rel ≤ 1e-12 |
| numpy callable = expression | E rel ≤ 1e-12, F rel ≤ 1e-10 |
| both = pinned LAMMPS `bond_style fene` (deck by `write_lammps_forcefield`: `bond_style fene`, `bond_coeff B-B 30 1.5 1 1`) | rel ≤ 1e-10 |
| custom category `urey_bradley` from Python (`k_ub*(distance(p1,p3)-r_ub)^2`) = pinned LAMMPS | rel ≤ 1e-10 |
| `.mrec` round trip: expression byte for byte; a **subprocess** with no registration reads it and gets the same energy | bit for bit |
| callable-only style in a fresh process | `NoKernel` (a `ValueError`) naming the style and `molrs.ff.style_registry.register_style` |
| refusals: unknown function, unbound variable, sealed `bond harmonic`, wrong `def_type` arity, kernel of the wrong shape, kernel raising, `write_gromacs_top_ff` → `NoEngineForm`, missing param at compile | each raises its `IrError` subclass naming the item |
| array param: `dihedral table/linear` (`table: f64[N]`) Tier-2 numpy kernel, round-trips, = hand linear interpolation | rel ≤ 1e-12 |

### P-molpy: `tests/test_potential/test_user_style.py` (and a < 30-line doc snippet)

`class Fene(mp.potential.StyleSpec)`; a `BeadSpring(mp.typifier.Typifier)`
with `source_forcefield()` (units `lj`) whose `assign` returns `TypeAssignment(nodes,
links={Bond: rows}, styles=[...])`; typify a chain → `to_frame` →
`mp.PotentialCompiler(ff).compile(frame)` → energy and forces = the analytic
sum (rel ≤ 1e-12) → `mp.io.write_mrec` / `read_mrec_forcefield` preserves the
style and its expression.

### Built-ins (G5)

`ff::ir::builtin_conformance` green: every Appendix A expression agrees with
its kernel (rel ≤ 1e-10) on 64 seeded configurations per style; every
`Positional` LAMMPS codec writes the same coefficient lines as today's
writer for the LAMMPS-read hand molecule and the P4 cases (byte-identical).

### Documentation

`molrs-python/docs/guides/extending-forcefield-ir.md`, "Extending the
force-field IR", after "Force-field IR" in `zensical.toml`: the form
(category and style specs, `ParamDimension`), the three tiers and their calling
conventions, categories, expressions (link to molrec), persistence, engines
and refusals, conversions (P3), the conformance errors, and "How this is
checked" with the numbers above.

## Work packages

Runs after wave 3 merges (w3 rewrites all readers/writers, `section.rs`,
`potential/mod.rs`, `pair/exceptions.rs`, `io/zarr/forcefield_io.rs`, Python
`ff/mod.rs`).

| WP | Scope | Size | Depends on | Hotspots |
|---|---|---|---|---|
| WP0 | This spec; molrec: array params, compound functions, custom-style rules, `one_four`, OpenMM/GROMACS/prmtop mapping rows — **done**, molrec `ir/w3d-compat` 16c4873 | S | – | molrec spec |
| WP1 | `ff/ir/{dim,category,spec,registry,conformance,error}.rs`; built-in specs; `compile.rs` generic gates; `with_registry`; sealing — **done**, molrs `ir/p-registry` f61ee0f3 | M | WP0 | `compile.rs`, `registry.rs` |
| WP3a | generic kernels `ff/potential/generic/*` over `ScalarForm` / `CompoundForm` — **done**, f61ee0f3 | M | WP0 | new files |
| WP3b | `ff/ir/expr/*`: parser, duals, compound functions, printer; compile fallback; built-in gate — **done**, `ir/p-expr` 3ba558e2, merged + glued in `ir/p-registry` ef537939 | L | WP3a | new files |
| WP2 | `StyleDefs::Relation`, `category() -> &str`, `section.rs` arity, Python `RelationStyle`/`RelationType`, `style_handles` | M | WP1 | `forcefield/mod.rs`, `section.rs`, `handles.rs` |
| WP5 | Python `molrs.ff.ir` in `molrs-python/src/ff/ir.rs` (2-line hook in `mod.rs`); `.pyi` | M | WP1, WP3b | `_lib.pyi` |
| WP9 | P3 on the protocol: `FormCodec`, `canonical()`, `to_form`, `fit_form` | M | WP1, WP3a | `torsion.rs`, `ir/form.rs` |
| WP6 | `TypeAssignment` with any relation kind; `assign_terms` endpoint matcher | M | WP2 | `typifier/mod.rs`, py `ff/mod.rs` |
| WP7 | persistence: core validate relaxation, partners pin, round-trip tests | S | WP0 merged, WP2 | `forcefield_section.rs` |
| WP8 | engine codecs: LAMMPS / OpenMM-XML (`Custom*Force`) / GROMACS+AMBER refusals (three parallel sub-WPs) | L | WP1, P4 | readers/writers |
| WP10 | proof crate, Python tests, LAMMPS script, doc page, `check.sh` gate | M | WP2–WP8 | new files, `check.sh` |
| WP11 | molpy: re-exports, stale imports, extending doc, test, pin molrs 0.16 | S | WP5, WP6 | molpy `potential/`, `engine/lammps.py` |

Order: wave 4 (parallel) WP0, WP1, WP3a+b, P4, P5 → wave 5 WP2, WP5, WP9 →
wave 6 WP6, WP7, WP8 → wave 7 WP10, WP11. P4 (cross-engine equivalence)
precedes WP8 as its regression net; P5 collides with WP5/WP6 only in
`molrs-python/src/ff/mod.rs`, hence WP5's separate `ir.rs`.

Constraints: all work local, nothing pushed; hooks always run, never
`--no-verify` (the `ext` gate joins the hooks); compiling and LAMMPS checks
on compute nodes only (`cargo mrs-*`, `--manifest-path` for the new crate);
targeted tests during development, `scripts/check.sh all` only before a push.

## Decisions (WP0, binding on the implementers)

- **D1 ParamDimension** has five exponents (E, L, A, Q, **M**); the design's four lacked
  mass, which `atom full` needs. String grammar and printing as §2.
- **D2 `A`**: `dim == "A"` is an angle value in `units.angle` (degrees);
  negative `A` exponents are per radian and never converted; any other
  positive `A` is refused.
- **D3 Parameters are seen as stored by every tier** (expression, Tier 2,
  Python): angle-valued parameters in degrees, the expression converts
  (`theta0*0.017453292519943295`); coordinates in radians. Same as molrec's
  existing rule; no tier converts.
- **D4 Improper binds `phi`** (signed dihedral of the listed order) **and
  `chi = abs(phi)`** (molrec's earlier `chi`-only rule widened); Tier 2's
  `q` is `phi`.
- **D5 Compound functions are available in every category with points**
  (bond … cmap, custom), not only outside the registry, so `angle charmm`
  and `improper trefoil` have expressions; never in `pair`. The geometric
  variable is shorthand for its compound function.
- **D6 Pair binding**: bare `x` = pair value (cross row's non-null cell,
  else `ParamCombination`; for unregistered styles only ε/σ(/14) mix), `x1`/`x2` = self
  rows of the first/second atom's type (first = `atomi` / lower index),
  `q1`/`q2` = `atoms.charge`. Names `q`, `q1`, `q2` and `x`+`x1` collisions
  are refused; pair expressions must be symmetric. The expression is the
  unweighted, untruncated energy.
- **D7 Definitions** follow Lepton's order (a definition uses only names to
  its right); no named constants; `^` binds tighter than unary minus.
- **D8 Custom categories**: arity 2..=5, block exactly `<name>s` (molrec
  now states it), so an unregistered category in a record is priceable from
  its expression alone. Python `register_category` has no `block` argument.
- **D9 Array params** are allowed anywhere (`f64[T, S…]`, uniform, axes ≥ 1,
  finite non-null rows, no precision) but are not expression variables.
- **D10 ParamCombination** gains `LjEpsilon`/`LjSigma` (joint rule of the style's
  `mixing`) beside `None`/`Arithmetic`/`Geometric`; absent `mixing` =
  `arithmetic` (molrs's `Mixing::UNDECLARED`).
- **D11 Defaults** are `Option<ParamValue>` (`Num` or `Text`), not `Option<f64>`,
  so `one_four = "regular"` is a default; Text params may list `choices`.
- **D12 Indexed parameters** (`indexed: true`, contiguous `x1..xM`) cover
  `k<m>`, `a<i>`; `unindexed_one_term` covers `dihedral periodic`'s one-term
  spelling.
- **D13 `register_style(spec, None)`** registers an expression-only style;
  re-registration that differs is `Conflict` unless Python `replace=True`
  (custom only); identical is a no-op; built-ins (styles and categories)
  are sealed in every registry made by `Registry::builtin()`.
- **D14 Conformance checks** run at registration on `samples`, otherwise
  once per process at first compile on ≤ 8 real terms; tolerances 1e-6
  (derivative), 1e-10 (agreement), 1e-12 (symmetry), relative to
  max(|value|, sample RMS).
- **D15 `force_is_gradient = false`** (only `coul/charmm`): agreement on
  energy only; such a style is never priced by its expression fallback.
- **D16 Persistence**: `to_section` writes the registry expression of a
  registered *custom* style; built-ins write none; an instance expression
  differing from the registry's is checked for agreement at first compile.
- **D17 Python errors**: `molrs.ff.ir.IrError(ValueError)`, one subclass per
  Rust variant, same names.
- **D18 FormCodec**: `project` is the exact inverse with refusal; the
  least-squares operation is `fit_form`; one canonical per family;
  `canonical()` acts only in the canonical style's category.
- **D19 `dihedral rb`** stays in molrec's registry and is registered in
  molrs as an expression-only built-in (Appendix A).
- **D20 molrec `pair lj/charmm`**: `inner`/`cutoff` are no longer "both
  required" (an OpenMM-read field has neither; the consumer states them),
  `one_four` ∈ {`regular`, `epsilon14`} refused otherwise on read.
- **D21 Reading OpenMM `Custom*Force`** (other than the two harmonic
  improper energies) stays refused; writing expression styles as
  `Custom*Force` is WP8.

## Open

- Persisting `ParamDimension`s (a molrec `dims` map in the style entry) would let an
  unregistered style export to LAMMPS/OpenMM with unit conversion; deferred
  until a user needs it.
- A LAMMPS `lepton` codec once a LEPTON-enabled lmp is installed.
- One-body expressions (`x1, y1, z1`) for walls/restraints.
