---
slug: cache-layers-05-residual
created: 2026-09-11
criteria:
  - id: ac-001
    summary: EnergyResidual is unexported keyword-only BatchTask
    type: code
    pass_when: |
      molix.data.residual.EnergyResidual subclasses BatchTask. __init__ is
      keyword-only (coefficients and support required; no write_e0). It is
      not listed in molix.data.__all__. src/molix/data/task.py still names
      only PadMolecularBatch and TotalCharge as shipped subclasses. No
      make_residual symbol exists under src/molix. src/molix/data/residual.py
      does not import molpot.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Schema adds ENERGY_RESIDUAL_KEY only; reuse E0_KEY
    type: code
    pass_when: |
      molix.schema.ENERGY_RESIDUAL_KEY == ("graphs", "energy_residual").
      This slug does not assign E0_KEY; EnergyResidual.execute writes
      graphs.e0 via molix.schema.E0_KEY from cache-layers-04-dtype.
      tests/test_molix/test_schema.py asserts ENERGY_RESIDUAL_KEY.
      has_energy still probes ENERGY_KEY only.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Always write graphs.e0; never overwrite graphs.energy
    type: code
    pass_when: |
      EnergyResidual.execute returns the caller TensorDict with graphs.energy
      the same tensor object as on entry, writes graphs.energy_residual, and
      always writes graphs.e0 as float64. There is no write_e0 parameter.
      It does not write ENERGY_KEY, FORCES_KEY, or atoms.energy.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Construction validates tables; ghosts masked; real Z raises
    type: code
    pass_when: |
      __init__ raises ValueError unless coefficients and support are 1-D,
      the same length, and support.dtype is torch.bool. After
      PadMolecularBatch(..., pad_atom_type=1), ghost rows with Z=1 contribute
      0 to e0 even when c[1] is nonzero. A real atom with support[Z] False
      or Z out of table range raises ValueError naming that Z; the same Z
      on a masked ghost does not.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Tests prove energy_residual column; losses and scripts untouched
    type: code
    pass_when: |
      TestEnergyResidual asserts batch[ENERGY_RESIDUAL_KEY] is populated and
      graphs.energy is unused for the residual identity. tests/test_molix/test_data/test_residual.py
      does not import energy_loss or MSELoss. src/molix/core/losses/molecular.py
      and src/molix/core/losses/energy.py are unmodified. scripts/mm_param_learning/class_i_fit.py
      is unmodified in this slug.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Unit-test goldens match the hard-coded H2O and CH4 literals
    type: runtime
    evaluator_hint: "path: tests/test_molix/test_data/test_residual.py"
    pass_when: |
      python -m pytest tests/test_molix/test_data/test_residual.py -v exits
      0. TestEnergyResidual asserts the literals e0=-2071.0 eV and
      E_res=-0.1234567890123 eV for the H2O case and e0=-1084.50277249196 eV,
      E_res=0.090426813059 eV for the CH4 case, within 1e-12 eV on the
      float64 subtract, with no third-party energy import and no
      regressions/ script.
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: Masked float64 residual matches the numpy oracle
    type: scientific
    evaluator_hint: path: tests/test_molix/test_data/test_residual.py
    pass_when: |
      On a PadMolecularBatch-padded H2O batch (real Z=[8,1,1], pad_atom_type=1
      ghosts, c[1]=-13.6 eV, c[8]=-2043.8 eV, E_total=-2071.1234567890123 eV)
      EnergyResidual writes graphs.e0 and a float64 subtract matching the
      oracle e0=-2071.0 eV, E_res=-0.1234567890123 eV within 1e-12 eV;
      ghost hydrogen adds 0. Subtract uses float64 even when atoms.pos is
      float32.
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: ENERGY_KEY writes do not clobber residual labels
    type: code
    pass_when: |
      After EnergyResidual.execute, replacing batch[ENERGY_KEY] in place leaves
      batch[ENERGY_RESIDUAL_KEY] identity and values unchanged. EnergyResidual
      itself never writes FORCES_KEY.
    status: verified
    last_checked: 2026-09-12
out_of_scope:
  - public molix.data.__all__ export and task.py shipped-subclass listing (07)
  - class_i_fit.energy_force_terms retarget (07; superseded by 07, not this Files list)
  - molecular.py / MSELoss / energy_loss edits
  - write_e0 flag
  - E0_KEY definition (04)
  - cache-layers-03-refs fit/freeze/checkpoint of c[Z]
  - cache-layers-04-dtype collate / batch_to precision policy
  - AtomicReferenceEnergy module edits; ARE stays model-side ADD to ENERGY_KEY
  - molpot imports from molix
  - AtomicDress, Cache, NeighborList
  - inference restoration adapter
  - ForceOutlierWeight / Huber / 200 kcal inlier masks
  - regressions/ scripts (tier deleted; goldens are unit tests)
---

# Acceptance criteria

Done means an unexported `EnergyResidual` BatchTask residualizes collated totals in float64 against a caller-supplied frozen table, with ghosts killed by `atoms.mask`, labels isolated from `ENERGY_KEY`, `graphs.e0` always written, and tests — not `molix.data.__all__` — as this slug’s caller. After the chain, it is the only in-flight label residualizer; 07 retires `energy_force_terms` and exports the type.

## AC-001 — Unexported keyword-only BatchTask

Construction matches `PadMolecularBatch` (`*` then required tensors). The molix/molpot one-way rule is a file-level import check on `residual.py`. Public export is deferred to 07 so this slice does not ship a façade with only tests behind it.

## AC-002 — Schema constant

One new spelling, `ENERGY_RESIDUAL_KEY`, declared next to `ENERGY_KEY`. `E0_KEY` is 04’s address; 05 must not fork it.

## AC-003 — Raw totals stay put; e0 always written

`graphs.energy` is the dataset total through training. Residual is a new column, not an in-place dress. `graphs.e0` is part of the scientific identity and is not optional in this slug.

## AC-004 — Construction fails closed; mask, not Z, defines real atoms

Invalid tables are illegal at `__init__`, not at first `execute`. Default padding writes hydrogen. Treating `Z==1` as real would add \(c[1]\) per ghost. The padder’s `atoms.mask` is the only real/ghost split.

## AC-005 — Column, not a new loss

`energy_loss` already indexes `batch["graphs", target_key]`. This slug proves the column exists. 07 aims the loss. `MSELoss` stays a flat-mapping generic. `class_i_fit.py` is untouched here so 07 can retire `energy_force_terms` in one place.

## AC-006 — Runtime goldens

The mirrored unit module is the golden owner (`regressions/` is gone). Literals are the Domain-basis H2O numbers and the CH4 large-total case.

## AC-007 — Scientific oracle

Same H2O case plus padding. Tolerance `1e-12` eV is tighter than the tester energy-exact band (`1e-10`) and is the float64 subtract of these magnitudes.

## AC-008 — Prediction writes vs labels

`Derivative` already writes `of=ENERGY_KEY`. The test is EnergyResidual’s column split: overwriting energy must not move the residual tensor.
