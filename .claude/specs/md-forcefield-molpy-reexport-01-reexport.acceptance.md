---
spec: md-forcefield-molpy-reexport-01-reexport
created: 2026-08-16
criteria:
  - id: ac-001
    summary: molix.md.ForceField is identically molpy.ForceField
    type: code
    pass_when: |
      tests/test_molix/test_md/test_forcefield.py::TestForceFieldReexport
      asserts `from molix.md import ForceField` and
      `from molpy import ForceField as MolpyFF` satisfy `ForceField is MolpyFF`.
    status: verified
    last_checked: 2026-08-16
  - id: ac-002
    summary: Delete homemade ForceField types from molix.md
    type: code
    pass_when: |
      CallableForceField, HarmonicForceField, LennardJonesForceField,
      LennardJonesCutForceField, PotentialForceField, and
      PeriodicPotentialForceField are absent from molix.md.__all__ and
      getattr(molix.md, name, None) is None; importing any of them from
      molix.md.forcefield raises ImportError.
    status: verified
    last_checked: 2026-08-16
  - id: ac-003
    summary: MD.set_forcefield accepts only molpy.ForceField
    type: code
    pass_when: |
      TestMD shows set_forcefield accepts molpy.ForceField and
      ClassIForceField/OPLSForceField (subclasses) and returns self;
      object(), a bare nn.Module, and a pos-callable each raise TypeError
      whose message names ForceField.
    status: verified
    last_checked: 2026-08-16
  - id: ac-004
    summary: MD.set_neighbors is fluent and arms rebuild
    type: code
    pass_when: |
      After set_forcefield + set_integrator + set_neighbors(nl),
      md.set_neighbors(nl) is md and md.integrator.rebuild is True;
      the same pair without set_neighbors leaves rebuild False.
    status: verified
    last_checked: 2026-08-16
  - id: ac-005
    summary: MD evaluates pots on TensorDict with no unit hacks
    type: code
    pass_when: |
      MD.run writes POS_KEY then uses pots.calc_energy and
      pots.calc_forces return values (also stored at ENERGY_KEY /
      FORCES_KEY) to build ForceOutput. Source of src/molix/md/driver.py
      and src/molix/md/forcefield.py contains no energy_scale. Conversion
      uses molix.units.preset_energy_to_md (UnitSystem.preset), not a
      handwritten metal→eV table.
    status: verified
    last_checked: 2026-08-16
  - id: ac-006
    summary: src/molix/md does not import molpot
    type: code
    pass_when: |
      A unit test reading every .py under src/molix/md/ asserts no
      import statement contains molpot (neither `import molpot` nor
      `from molpot`).
    status: verified
    last_checked: 2026-08-16
  - id: ac-007
    summary: Integrator tests use unnamed PotentialLike, not FF subclasses
    type: code
    pass_when: |
      tests/test_molix/test_md/test_integrators.py binds Integrator via
      unnamed PotentialLike helpers or ForceField.def_pairstyle; no class
      defined in that file subclasses ForceField.
    status: verified
    last_checked: 2026-08-16
  - id: ac-008
    summary: NeighborList.build still writes only edge_index+shifts
    type: code
    pass_when: |
      After NeighborList.build(batch), batch["edges"] keys are exactly
      {edge_index, shifts} with shapes (capacity, 2) and (capacity, 3);
      edge_diff and edge_dist are absent.
    status: verified
    last_checked: 2026-08-16
  - id: ac-009
    summary: Regression script reproduces hard-coded public-API goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-forcefield-molpy-reexport-01-reexport.py`
      exits 0 and reproduces: ForceField is molpy.ForceField; the six
      deleted names are absent from molix.md.__all__; two-atom
      ClassIForceField harmonic bond (k=2, r0=1 Å, r=1.5 Å) energy
      equals the literal 0.25 with atol 1e-10; set_neighbors is fluent.
      No third-party import or subprocess.
    status: verified
    last_checked: 2026-08-16
  - id: ac-010
    summary: New public MD mounts carry Google docstrings with units
    type: docs
    pass_when: |
      MD.set_forcefield, MD.set_neighbors, Integrator.bind_force, and
      the internal PotentialLike protocol have Google-style docstrings;
      set_neighbors names NeighborList and Å; set_forcefield names
      molpy.ForceField and to_potentials().
    status: verified
    last_checked: 2026-08-16
  - id: ac-011
    summary: Bare molpy to_potentials fails loud on TensorDict
    type: code
    pass_when: |
      A molpy.ForceField that is not the molpot override (to_potentials
      returns Frame kernels) causes MD.run / first calc_energy to raise
      TypeError naming calc_energy and TensorDict. src/molix/md still
      has no molpot import.
    status: verified
    last_checked: 2026-08-16
  - id: ac-012
    summary: ForceOutput energy/force conversion lives in molix.units
    type: code
    pass_when: |
      Scaling E/F into amu·Å²/fs² uses molix.units / molpy.UnitSystem.preset
      (pint .to). src/molix/md/driver.py contains no energy_scale and no
      handwritten metal→eV or real→kcal table; any helper is in
      src/molix/units.py.
    status: verified
    last_checked: 2026-08-16
out_of_scope:
  - NeighborList edge_diff write (02-nblist)
  - molpot/molzoo consume geometry (03/04)
  - scripts/docs/benchmarks (05-wire)
  - energy_scale or reading ff.units
---

# Acceptance — md-forcefield-molpy-reexport-01-reexport

`molix.md.ForceField` 与 `molpy.ForceField` 是同一对象；自制力场类型删除；`MD.set_forcefield` / `set_neighbors` 是唯一公开挂载；内部走 `to_potentials()` + TensorDict `calc_*`，不 import molpot。

## AC-001 — molix.md.ForceField is identically molpy.ForceField

`TestForceFieldReexport` 断言两边 import 的 `ForceField` 是同一对象。

## AC-002 — Delete homemade ForceField types from molix.md

六个自制名不在 `__all__`，从 `molix.md.forcefield` import 它们会 `ImportError`。

## AC-003 — MD.set_forcefield accepts only molpy.ForceField

`ClassIForceField` / `OPLSForceField` 合法；`object()` / `nn.Module` / callable 抛点名 `ForceField` 的 `TypeError`。

## AC-004 — MD.set_neighbors is fluent and arms rebuild

`set_neighbors` 返回 `self` 且打开 `integrator.rebuild`；未挂表则为 `False`。

## AC-005 — MD evaluates pots on TensorDict with no unit hacks

`calc_energy` / `calc_forces` 返回值组装 `ForceOutput`；源码无 `energy_scale`、不读 `ff.units`。

## AC-006 — src/molix/md does not import molpot

`src/molix/md/` 下无 `import molpot` / `from molpot`。

## AC-007 — Integrator tests use unnamed PotentialLike

`test_integrators.py` 不定义 `ForceField` 子类。

## AC-008 — NeighborList.build still writes only edge_index+shifts

本规格冻结写表面，`edge_diff` 留给 02。

## AC-009 — Regression script reproduces hard-coded public-API goldens

谐振子键能量字面量 `0.25`，无第三方。

## AC-010 — New public MD mounts carry Google docstrings with units

`set_forcefield` / `set_neighbors` / `bind_force` / 内部协议带 Google 文档与单位。
