---
slug: cache-layers-07-protocol
created: 2026-09-12
criteria:
  - id: ac-001
    summary: Official and historical OMol25 loaders are separately named
    type: code
    pass_when: |
      load_omol25_official and load_omol25_official_tiny still exist in
      scripts/mm_param_learning/class_i_fit.py, still default to kcal
      conversion and val_role val-neutral, and remain the callees of
      prod_day.py / quasi_prod.py / calibration gates. load_omol25_fairchem_4m
      is the eV official-4M loader and is the only loader the official
      composition site calls. ClassIFit constructed with load_omol25_official()
      still uses historical Huber/Adam objectives. energy_force_terms remains
      the historical residualizer and is not called from omol25_fairchem_4m.py
      or fairchem_4m_objective.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: No chemistry-ML imports on the molnex Python surface
    type: code
    pass_when: |
      src/, tests/, and scripts/ contain no imports of ase, e3nn, mace,
      fairchem, lammps, or openmm. Protocol goldens are literals, not live
      FAIR/ASE/MACE calls.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Pinned AllScAIP 4M settings exist as the resolved profile
    type: scientific
    pass_when: |
      OMOL25_FAIRCHEM_4M records FAIR commit
      41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9, AdamW lr 8e-4, wd 1e-3,
      cosine 80 epochs, 5-epoch warmup factor 0.2, min factor 0.01, clip 100,
      bf16 off, tf32 on, refs float64, normalizer mean 0 rms 1.433569 on
      energy and forces, PerAtomMAE coeff 10, per-structure L2 coeff 10,
      cutoff 6 Å, max_neighbors 30, pad 50, batch_max_atoms 400 as batch
      capacity, shuffle seed 0, eval/ckpt every 2000 steps, keep 5, and
      reference SHA-256
      e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640.
      PROTOCOL_ADAPTATIONS is a frozen tuple[str, ...] that names
      conservative Derivative, the zero-error denominator, and kcal→eV
      conversion at ForceAwareModel.energy. The profile does not claim
      eSEN, MACE, or UMA reproduction.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Zero-error structures remain in the force denominator
    type: scientific
    pass_when: |
      TestPerStructureL2Force includes a structure with identical predicted
      and reference forces and still divides by the full eligible structure
      count. A batch without atoms.mask treats every atom as real. The
      two-graph PerAtomMAE fixture matches
      0.5 * (1/(1.433569*2) + 1/(1.433569*4)) within 1e-10, uses
      graphs.num_atoms as N_g, is implemented via energy_loss(target_key
      energy_residual, per_atom=True, elementwise L1) / s_E, and does not
      match the atom-weighted 2/(1.433569*6). per_structure_l2_force is not
      force_loss. No E_tilde unscale-before-Derivative assertion exists.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: require_protocol_resolved fails closed on unresolved metadata
    type: code
    pass_when: |
      protocol_unresolved(OMOL25_FAIRCHEM_4M) includes world_size / global
      batch, AdamW betas/eps/decay groups, monitoring IDs/seed, and dataset
      revision/shard digests while those values are None.
      require_protocol_resolved is defined in
      scripts/mm_param_learning/configs.py and raises on a non-empty list.
      official_class_i_fit and omol25_fairchem_4m.run call
      require_protocol_resolved before constructing ClassIFit. No default
      64-rank or guessed monitoring seed is stored as if resolved.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: ClassIFit.model forwards energy_unit; dressing is optional
    type: scientific
    pass_when: |
      ForceAwareModel.energy multiplies container kcal mol⁻¹ energy by a
      Python float (1.0 * units.kilocalorie_per_mole).to(energy_unit).magnitude
      computed once via molix.units, not a buffer. ClassIFit.model() returns
      ForceAwareModel(imm, dressing=<clone or None>,
      energy_unit=self.energy_unit) — not ForceAwareModel(imm, dressing).
      Official ClassIFit(energy_unit=eV) with dressing None does not install
      a zeros table, does not add dressing in forward, and converts at
      energy(). Historical ForceAwareModel(imm, dressing) still constructs,
      still adds dressing in forward, and still uses energy_unit kcal
      (scale 1). There is no reference= kwarg. physics(batch) still binds
      on the host; forward still runs Derivative on self.energy.
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: Extra-role evaluate is isolated and one-home
    type: code
    pass_when: |
      load_omol25_fairchem_4m returns (train, roles) where roles is a mapping
      with keys general, metal, electrolytes, biomolecules, spice, and
      neutral_organics. Views are not concatenated. Official labels are eV.
      Missing omol-elem-refs fails closed. 400 is not a chemistry keep-filter.
      eval_extra_roles copies general scalars before the first extra
      evaluate, calls Trainer.evaluate per remaining key, and after every
      role state["eval"] contains ONLY role-name keys (no leftover top-level
      loss / loss_energy). A real CheckpointHook(save_best=True,
      save_last=True) seeded with _best_value does not replace best.pt or
      last.pt and does not update _best_value when a stub evaluate writes a
      better extra-role loss_energy and fires on_eval_step_complete on live
      hooks. Extra-role loss_energy is not the fit's best metric. ClassIFit
      has no val_roles=. ClassIFit.run still returns the historical metrics
      dict. Official run builds the Trainer via ClassIFit.trainer(model).
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: Official site passes eval_loss=; run installs cosine
    type: code
    pass_when: |
      official_class_i_fit passes loss=fairchem_4m_objective and
      eval_loss=fairchem_4m_objective (same named protocol objective) and
      leaves lr_scheduler_factory None. ClassIFit.trainer(stub) raises while
      loss is fairchem_4m_objective and the factory is None. After assigning
      a Trainer-arity (optimizer)->scheduler factory, eval_step on a stub
      batch equals that objective. official_cosine_factory(steps_per_epoch)
      has Trainer arity. Historical train_objective/Huber still matches the
      kcal stub-batch golden. Official run does not call ClassIFit.run and
      does not pass protocol=.
    status: verified
    last_checked: 2026-09-12
  - id: ac-009
    summary: One published table is a checkpoint submodule and EnergyResidual
    type: code
    pass_when: |
      molix.data.EnergyResidual is listed in molix.data.__all__ and named
      in molix.data.task as a shipped BatchTask subclass. load_omol_elem_refs
      is the only ARE constructor for this protocol and fail-closes on a
      missing artifact. run() is the single load. The official composition
      site imports EnergyResidual from molix.data, clones
      coefficients/support from that instance, and registers the same
      instance as submodule omol_elem_refs via attach_omol_elem_refs
      (model.omol_elem_refs is table). Historical ClassIFit.datamodule does
      not mount EnergyResidual. ForceAwareModel has no reference= flag.
    status: verified
    last_checked: 2026-09-12
  - id: ac-010
    summary: Hard-coded protocol goldens run in mirrored unit tests
    type: runtime
    pass_when: |
      python -m pytest tests/test_scripts/test_mm_param_learning/test_configs.py
      tests/test_scripts/test_mm_param_learning/test_class_i_fit.py
      tests/test_scripts/test_mm_param_learning/test_omol25_fairchem_4m.py -v
      reproduces the literals in Testing strategy: RMS 1.433569, energy/force
      coefficients 10/10, SHA-256
      e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640,
      two-graph L_E = 0.5*(1/(1.433569*2)+1/(1.433569*4)), the zero-error
      structure still in the force denominator, the ForceAwareModel
      energy-seam scale from molix.units, eval_loss= actually used,
      LazyOMol25 subset keeping 1.0 eV as 1.0, extra-role evaluate not
      replacing best.pt, and per-role state["eval"][role] writes with no
      leftover top-level loss. No third-party subprocess. No e2e train.
    status: verified
    last_checked: 2026-09-12
  - id: ac-011
    summary: This spec grants no production GO
    type: code
    pass_when: |
      OMOL25_FAIRCHEM_4M and omol25_fairchem_4m.py do not mark production GO.
      Comments/docs state that
      .claude/notes/production-training-checklist.md still applies and that
      this profile needs a fresh molexp calibration. Historical Class-I
      checklist ticks are not reused as sign-off.
    status: verified
    last_checked: 2026-09-12
  - id: ac-012
    summary: LazyOMol25 instance scale survives subset reconstruction
    type: code
    pass_when: |
      LazyOMol25 stores a pint energy_unit and multiplies
      _sample_from_record / _to_energy_unit / forces by the instance scale,
      never module-level _EV_TO_KCAL. subset() forwards energy_unit and
      dressing. An official eV view and its subset keep a raw 1.0 eV label
      as 1.0. load_omol25_fairchem_4m_tiny constructs every view with
      energy_unit=units.eV. Historical default still converts eV to kcal
      via molix.units. _EV_TO_KCAL remains only on the historical
      _fit_e0_subsample path.
    status: verified
    last_checked: 2026-09-12
  - id: ac-013
    summary: Official val DataModule reuses the train task list
    type: code
    pass_when: |
      official_val_datamodule(ds, train_dm=dm).batch_tasks is
      dm.batch_tasks, which is neighbor_list then TotalCharge then official
      extras (EnergyResidual), not a residual-only list. It also copies
      batch_size, num_workers, pin_memory, prefetch_factor,
      multiprocessing_context, and seed from train_dm (not DataModule
      defaults). Official run closes make_val_datamodule over the
      DataModule fit.datamodule() built.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means `omol25-fairchem-4m` is an inspectable pin in `configs.py`, historical Class-I loaders and `ClassIFit.run` still run the kcal Huber path, `ClassIFit.model()` forwards `energy_unit` into the kcal→eV energy seam, extra-role `Trainer.evaluate` cannot steal `best.pt` or the eval home, `LazyOMol25` conversion is the instance scale, and one official composition site wires eV labels, `eval_loss=`, `EnergyResidual`, and `add_module("omol_elem_refs")` without importing FAIR stacks.

## AC-001 — Forked loaders

`load_omol25_official` is not retargeted. Grep historical launchers for that name; grep the official script for `load_omol25_fairchem_4m`. Official code does not call `energy_force_terms`.

## AC-002 — No chemistry-ML imports

Same gate as the repo TID251 lammps/openmm ban, extended here to ase/e3nn/mace/fairchem on the surfaces this spec touches.

## AC-003 — Pinned settings

`TestOmol25Fairchem4m` is the ledger for every locked hyperparameter. Unresolved fields stay `None` (see AC-005). `PROTOCOL_ADAPTATIONS` is documentation stored as a frozen tuple.

## AC-004 — Zero-error denominator

Independent two- and three-graph tensors; no DataLoader. Missing `atoms.mask` counts every atom. \(L_E\) goes through `energy_loss(..., per_atom=True)` with \(N_g =\) `graphs.num_atoms`. The FAIR `DDPMTLoss.per_structure` nonzero filter is the behavior we refuse. No unscale-`Derivative` choreography.

## AC-005 — Fail closed on unresolved metadata

`require_protocol_resolved` is the analogue of `production_calibration_drift`, and both `official_class_i_fit` and the official launcher actually call it. Inventing 64 ranks or a monitoring seed to empty that list is a failure of this criterion.

## AC-006 — Unit seam is on the composed object

`ClassIFit.model()` is `ForceAwareModel(imm, dressing=…, energy_unit=self.energy_unit)`. Official `dressing=None` does not install zeros. One optional table; no `reference=`. Scale is a Python float. Autograd then yields eV Å⁻¹ forces.

## AC-007 — Named eval roles, isolated evaluate, one home

Loader return contract plus `eval_extra_roles` writing only `state["eval"][role]` after copying general scalars, with CheckpointHook persistence disabled for the extra-role loop. Concatenating views, dropping extra roles, replacing `best.pt` from a domain view, or leaving `state["eval"]["loss"]` as the last extra role is a failure. `ClassIFit.run` keeps its metrics-dict return.

## AC-008 — eval_loss= is passed; cosine is run()-only

Official train and eval share `fairchem_4m_objective`. `official_class_i_fit` is not train-ready (factory `None`); `trainer()` refuses that pair; `run()` installs `official_cosine_factory`. Historical Huber is asserted on a stub batch, not as omitted-kwarg identity.

## AC-009 — One table, two uses; EnergyResidual exported

`load_omol_elem_refs` is the single ARE load (`run()` calls it once). cache-layers-05 left the type unexported; this caller publishes `molix.data.EnergyResidual`, clones tensors from that instance, and registers the same instance as `omol_elem_refs`.

## AC-010 — Runtime goldens

Mirrored unit tests stand in for the deleted `regressions/` tier (CLAUDE.md, 2026-09-07). No e2e train.

## AC-011 — No production GO

Checklist evidence for this exact profile does not exist yet; the spec must not pretend otherwise.

## AC-012 — Live LazyOMol25 scale

`subset()` / tiny official prefixes keep eV. Module-level `_EV_TO_KCAL` is not the official conversion path.

## AC-013 — Val collate matches train collate

Official extra-role DataModules reuse `train_dm.batch_tasks` (neighbor_list + `TotalCharge` + `EnergyResidual`).
