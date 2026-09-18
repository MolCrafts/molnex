---
title: Named protocol omol25-fairchem-4m is composed beside historical Class-I
status: code-complete
grilled: true
created: 2026-09-12
chain: cache-layers
depends:
  - cache-layers-01-sources
  - cache-layers-02-cache
  - cache-layers-03-refs
  - cache-layers-04-dtype
  - cache-layers-05-residual
supersedes: cache-layers-07-protocol
---

# Named protocol omol25-fairchem-4m is composed beside historical Class-I

## Summary

OMol25 training under the named protocol `omol25-fairchem-4m` follows FAIR’s public OMol-only AllScAIP 4M recipe at commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`: official electron-volt labels, the official 4M split plus a named mapping of official validation and domain views, the published combined elemental reference table, RMS scale `1.433569` on energy and forces, Per-atom MAE and per-structure force L2 both with coefficient 10, AdamW with a cosine 80-epoch schedule, and conservative autograd forces recorded as a named adaptation against the upstream direct-force head. Historical Class-I launchers keep calling `load_omol25_official` / `load_omol25_official_tiny` and `ClassIFit.run` with their kcal / val-neutral / Huber / Adam pins. The official path is a separately named loader plus one composition site that reads `OMOL25_FAIRCHEM_4M` field-by-field, forwards `energy_unit=eV` through `ClassIFit.model()` into `ForceAwareModel.energy` (the one kcal→eV owner), mounts `EnergyResidual` cloned from one `load_omol_elem_refs()` instance, injects the same named protocol objective as `loss=` and `eval_loss=`, and scores remaining eval roles with `Trainer.evaluate` into a one-home `state["eval"][role]` map after isolating checkpoint and eval-journal hooks. Unresolved world-size, AdamW beta/eps/decay-group, monitoring-ID, and dataset-revision fields block comparison and production launch rather than being invented; this spec does not grant a production GO.

## Domain basis

Units on this protocol are source units: energy in eV, forces in eV Å⁻¹, positions in Å. Reporting uses meV, meV/atom, and component force MAE in meV Å⁻¹. Do not convert labels to kcal mol⁻¹ on the official path. Convert only through `molix.units.units`.

Published combined elemental reference, applied **once** to raw DFT totals (float64), with \(A_g\) the real atoms of graph \(g\) (never padded slots):

```text
e0[g]     = sum(c[Z[i]] for i in A_g)
E_res[g]  = E_DFT[g] - e0[g]
E_total[g]= E_model_res[g] + e0[g]
```

`c[Z]` is the OMol vector from the pinned combined coefficient file (SHA-256 of the complete upstream YAML bytes, checked 2026-09-11):

```text
e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
```

Do not subtract isolated-atom or heat-of-formation terms again, treat this vector as a small correction on top of another table, or fit a second offset. Historical fit-sample IDs inside that YAML are not established as 4M-only; this protocol uses the table as an external published preprocessing artifact, as the selected recipe does. A train-only QR/SVD fit is a **different named protocol** (generic contract in cache-layers-03-refs), not a fallback when the published table is missing.

Label normalization (same numeric RMS for energy and forces; \(\mu_E = 0\)):

```text
s_E = s_F = 1.433569    # eV and eV Å⁻¹
E_tilde[g] = (E_res[g] - 0) / s_E
F_tilde[i] = F[i] / s_F
```

Intended training loss on valid graphs \(G\) and eligible real atoms \(A_g\), with \(N_g = |A_g|\) (real atoms, never pad capacity). Packed Class-I collate writes that count at `graphs.num_atoms`; \(L_E\) reads that column (this spec does not mount `PadMolecularBatch`):

```text
L_E = mean_g( abs(E_model_res[g] - E_res[g]) / (s_E * N_g) )
L_F = mean_g( mean_i_in_A_g( ||F_model[i] - F_DFT[i]||_2 / s_F ) )
L   = 10 * L_E + 10 * L_F
```

\(L_E\) is PerAtomMAE implemented as `energy_loss(target_key="energy_residual", per_atom=True, elementwise=L1) / s_E` with \(N_g =\) `graphs.num_atoms`. \(L_F\) is Euclidean vector error reduced per structure, then a mean over structures — not component MSE, not `force_loss`, and not an atom-weighted global mean. Energy and force denominators are independent. **Every eligible structure stays in the force denominator**, including structures whose force error is exactly zero. Do not copy `DDPMTLoss.per_structure` dropping zero-error structures from the count. Empty-valid ranks and padding are handled by masks, not by shrinking \(|G|\). Missing `atoms.mask` means every atom is real (same as `force_loss` when `weight_key` is absent from the batch). Ghost masking applies only when a later caller has padded. \(s_E\) and \(s_F\) live in these two loss functions; this protocol does not compose a `GlobalRescale` or unscale inside `Derivative`.

Conservative forces (named adaptation versus the upstream direct-force head):

```text
F_model = - d E_model_res / d pos
```

Elemental \(e0\) does not depend on positions, so it does not contribute to \(F\).

Class-I container energy \(E_{\mathrm{MM}}\) and forces are OPLS-AA / LAMMPS-real **kcal mol⁻¹** and **kcal mol⁻¹ Å⁻¹**. Official labels are eV. One conversion, at the traced energy seam, before `Derivative` sees the scalar:

```text
E_model_res[g] = E_MM[g] * (1 kcal mol⁻¹ → energy_unit via molix.units)
F_model        = - d E_model_res / d pos
```

Always multiply by the pint-derived **Python float** stored once at construction (historical `energy_unit = kilocalorie_per_mole` ⇒ scale 1). Do not copy a conversion float into source, and do not register the scale as a buffer. This protocol is OMol25 eV labels plus conservative Class-I energy converted to eV at `ForceAwareModel.energy`. It is not a second named protocol and not a silent mix of kcal Huber with eV DFT.

Pinned recipe settings (AllScAIP OMol-only 4M at the FAIR commit):

| Setting | `omol25-fairchem-4m` |
|---|---|
| Training duration | 80 epochs |
| Optimizer | AdamW; lr `8e-4`; weight decay `1e-3` |
| LR schedule | Cosine; 5-epoch warmup; start factor `0.2`; min factor `0.01` |
| Gradient-norm clip | `100` |
| Backbone compute | float32; BF16 off; TF32 on |
| Reference arithmetic | float64, independent of backbone |
| Normalizer | mean `0`; RMS `1.433569` on energy **and** forces |
| Energy objective | PerAtomMAE; coefficient `10` |
| Force objective | per-structure L2; coefficient `10` |
| Cutoff graph (upstream) | nonperiodic; `6` Å; up to `30` neighbors |
| Upstream neighbor pad | `50` (capacity, not 50 physical neighbors) |
| Sampler atom budget | `400` — **batch capacity, not a chemistry filter** |
| Shuffle seed | `0` |
| Eval / checkpoint | every `2000` optimizer steps |
| Checkpoint retention | latest `5` |

Unresolved on the public configs inspected here — **do not invent**; record as required metadata that must be resolved before comparison or production launch: world size / global batch / accumulation / tail policy; AdamW \(\beta_1,\beta_2\), \(\epsilon\), and parameter decay groups; monitoring subsample IDs and selection seed; dataset revision and shard digests. The recipe comment “64 ranks” versus `ranks_per_node=1` in the bundled cluster config is not a verified global batch.

Differences versus other published profiles (do **not** label a run as eSEN, MACE, or UMA reproduction):

| Item | OMol25 paper eSEN-sm | This protocol (AllScAIP 4M) | UMA mix | Historical Class-I script |
|---|---|---|---|---|
| Epochs | All schedule 12 | 80 | 30 / 30 mix | 1-day / unbounded `PROD_DAY` |
| Energy / force coeffs | 10 / 5 | 10 / 10 | (UMA recipe) | Huber 1 / 1 |
| RMS | paper/evaluator | `1.433569` | `1.423` | none (kcal Huber) |
| Units | eV | eV (labels + converted \(E_{\mathrm{MM}}\)) | eV | kcal mol⁻¹ via `molix.units` |
| Train split | paper splits | official 4M | multi-dataset mix | `4m` + kcal conversion |
| Val | official | named official val + domain views | mix | `val-neutral` / 8:2 |
| References | HOF + linear | combined `uma_v1_hof_lin_refs` | UMA mix | 8192-row `lstsq` e0 npy |
| Forces | direct head in FAIR | **conservative `Derivative`** | direct | conservative + Huber inliers |
| Inlier cut | none in this recipe | none | — | `200` kcal mol⁻¹ Å⁻¹ |

References:

- Levine, D. S. et al. “The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models.” 2025. https://arxiv.org/abs/2505.08762
- Wood, B. M. et al. “UMA: A Family of Universal Models for Atoms.” 2025. Recipe at the pin: [uma-recipe][uma-recipe]
- Tran, R. et al. “The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis.” *ACS Catalysis* 2023, 13, 3066. https://arxiv.org/abs/2206.08917 https://doi.org/10.1021/acscatal.2c05426
- Dataset docs (raw totals in eV, forces in eV Å⁻¹, charge/spin metadata): https://fair-chem.github.io/omol25/
- FAIR commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`: [recipe][recipe], [tasks][tasks], [dataset][dataset], [coefficients][coefficients], [reference][reference], [loss][loss], [unit][unit], [sampler][sampler], [all-recipe][all-recipe]

[recipe]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/omol_4m.yml
[tasks]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/tasks/omol_direct.yaml
[dataset]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/dataset/omol_4M.yaml
[coefficients]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/element_refs/uma_v1_hof_lin_refs.yaml
[reference]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/src/fairchem/core/modules/normalization/element_references.py
[loss]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/src/fairchem/core/modules/loss.py
[unit]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/src/fairchem/core/units/mlip_unit/mlip_unit.py
[sampler]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/src/fairchem/core/datasets/samplers/max_atom_distributed_sampler.py
[all-recipe]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/omol_all.yml
[uma-recipe]: https://github.com/facebookresearch/fairchem/blob/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/uma/training_release/uma_sm_direct_pretrain.yaml

## Design

The protocol is experiment/training configuration in `scripts/mm_param_learning/configs.py`, the same pin style as `PROD_DAY`. It is not a molix type, not a `Cache` default, and not a new façade module. `scripts/` may import molhub and molnex; `src/` still must not import molhub.

### Law findings (this rewrite)

- 🚨 `eval_extra_roles` / `Trainer.evaluate` — `Trainer.evaluate` is the full eval phase (`_run_eval_phase` → `on_eval_step_complete`). `ClassIFit.trainer()` still mounts `CheckpointHook(save_best=True, save_last=True)` (`class_i_fit.py` trainer hooks; `checkpoint.py` writes `best.pt` / `last.pt` from `("eval", "loss_energy")`). Extra-role evaluate on that hook set can replace `best.pt` when a domain view’s `loss_energy` beats general, rewrite `last.pt`, and journal those scalars as the fit’s eval. Stub tests that only check `evaluate`’s return cannot catch this. **This draft:** keep `Trainer.evaluate` as the scorer. `eval_extra_roles` wraps the extra-role loop in `_without_eval_persistence(trainer)`: remove every live hook that implements `on_eval_step_complete` from `trainer.hooks` for the loop, restore them in `finally`. Do not grow `CheckpointHook`. Extra-role `loss_energy` is not the fit’s best metric; `save_best` / `save_last` / `_best_value` / on-disk `best.pt` and `last.pt` stay the general-eval values. Tests use a real `CheckpointHook` plus a stub `evaluate` that writes a *better* extra-role `loss_energy` and fires `on_eval_step_complete` on whatever hooks are still attached.
- 🔴 `state["eval"]` two homes — snapshotting under `state["eval"]["general"]` then calling `evaluate` leaves live top-level `loss` / `loss_energy` as the last extra role; the `if not isinstance(v, dict)` filter was a workaround for that mix; after `run()`, `state["eval"]["loss"]` would be spice/neutral_organics, not general. **This draft:** copy general scalars into a **local** dict **before** the first extra `evaluate` (`evaluate` mutates the same namespace). After every role, replace the namespace with **only** role-name keys whose values are metric mappings: `trainer.state["eval"] = {role: mapping, ...}` (TrainState allows replacing a namespace with a dict; no slash-key writes). Delete leftover top-level scalars. Nested write only: `state["eval"][role] = …` is the per-role assignment; the stored contract after each role (and after `run()`) is `state["eval"]["general"]`, `state["eval"]["metal"]`, … with **no** `state["eval"]["loss"]`. The isinstance filter may parse `evaluate`’s mixed return; it is not the stored home.
- 🔴 `LazyOMol25.energy_unit` dead if read/rebuild still use `_EV_TO_KCAL` — `subset()` today rebuilds `LazyOMol25(..., dressing=self.dressing)` and would default official prefixes back to kcal; `load_omol25_fairchem_4m_tiny` is a bounded prefix of official roles (that *is* `subset()`). **This draft:** store the pint unit and the scale `float((1.0 * units.eV).to(energy_unit).magnitude)` on the instance. `_sample_from_record`, renamed `_to_energy_unit` (was `_to_kcal`), and force conversion multiply by **that** scale, never module-level `_EV_TO_KCAL`. `subset()` (and any other reconstruction) forwards `energy_unit` and `dressing`. Tiny official views construct with `energy_unit=units.eV`. `_EV_TO_KCAL` remains only for historical `_fit_e0_subsample`.

### Law findings from the previous draft (still closed)

- 🚨 `ClassIFit.model` — `model()` is `ForceAwareModel(imm, dressing=<clone or None>, energy_unit=self.energy_unit)`. Official: `energy_unit=units.eV`, `dressing=None` (no zeros table).
- 🔴 `dressing=` vs `reference=` — one optional table, `dressing=`. No `reference=`. `energy()` always converts container kcal mol⁻¹ → `energy_unit`. `forward` adds the table **iff** it is present.
- 🔴 Official `eval_loss` — official site passes `eval_loss=fairchem_4m_objective` (same named objective as train).
- 🔴 Extra eval roles dropped on `ClassIFit.val_ds` — official site owns extra eval via `Trainer.evaluate`; no `ClassIFit.val_roles=`; `ClassIFit.run` still returns the historical metrics dict.

### Folded non-blockers (must not remain silent)

- **One residualizer per named protocol.** cache-layers-05’s “one residualizer after 07” is one per protocol, not one in the file. Official labels: `EnergyResidual` only; `fairchem_4m_objective` / official `run` / official loaders **must not** call `energy_force_terms`. Historical `train_objective` keeps `energy_force_terms`. Do not add a third subtract (not in official `ForceAwareModel.forward`, not in the official loss).
- **\(L_E\)** is `energy_loss(target_key="energy_residual", pred_key="energy", per_atom=True, elementwise=torch.nn.L1Loss(reduction="none")) / s_E`. \(N_g =\) `graphs.num_atoms`. **`per_structure_l2_force` stays a new function** — `force_loss` is not per-structure vector L2.
- **`omol_elem_refs` has one owner.** `load_omol_elem_refs() -> AtomicReferenceEnergy` (official module) is the only constructor of that table (`scheme="combined_hof_linear"`, `units="eV"`, explicit `supported=`). Missing artifact fail-closes. Official loaders fail-close on absence and do **not** construct the module. `run()` is the single `load_omol_elem_refs()` call. `EnergyResidual` clones tensors from that instance; `attach_omol_elem_refs(model, table)` registers **the same instance** (`model.omol_elem_refs is table`).
- **`lr_scheduler_factory`** keeps Trainer arity `(optimizer) -> scheduler`. `run()` is the only complete official composition: `official_class_i_fit` does **not** set the cosine (factory stays `None` — not train-ready). `ClassIFit.trainer()` raises `ValueError` naming `omol25_fairchem_4m.run` when `self.loss is fairchem_4m_objective` and `lr_scheduler_factory is None`. `run()` assigns `official_cosine_factory(len(dm.train_dataloader()), profile)` then calls `trainer()`.
- **`make_val_datamodule`.** Official `run` calls `fit.datamodule()` once and closes `official_val_datamodule(ds, *, train_dm)` over **that** object: `batch_tasks is train_dm.batch_tasks` **and** `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`, `multiprocessing_context`, and `seed` copied from `train_dm` (not DataModule defaults 32/4/`pin_memory=True`). Not a residual-only list.
- **`attach_omol_elem_refs`** stays a public function because `TestOmol25Fairchem4mScript` targets it; it owns the checkpoint submodule name `omol_elem_refs`.
- **`ForceAwareModel` scale** is a Python `float` computed once via `molix.units`, not a buffer.
- **RMS one home.** `per_atom_mae_energy` / `per_structure_l2_force` close over `OMOL25_FAIRCHEM_4M` for `s_E` / `s_F`; tests still assert the pin literal `1.433569`. Do not hard-code a second copy of the number in the loss bodies.
- **Unresolved AdamW/world-size fields are ledger-only.** `require_protocol_resolved` still fails closed while they are `None`. Filling them in a test-local copy does **not** change `official_class_i_fit`: the factory stays `AdamW(lr, weight_decay)` only, and this spec does not pass `accumulate_grad_batches` / β / ε / decay groups into `Trainer`. Those keys become live in a later spec that wires them; until then they must not be treated as used.
- **`seed=` from the pin.** Official `ClassIFit(..., seed=profile["shuffle_seed"])` with shuffle seed `0`. Historical default `2666` stays when the knob is omitted.
- **Eval-persistence isolation is by callback, not a closed type list.** `_without_eval_persistence` removes every live hook that implements `on_eval_step_complete` (today `CheckpointHook`; also `MolRecMetricsHook` / `JournalHook` / `TensorBoardHook` if present), restores in `finally`. Do not list `Log` unless it actually fires on eval.
- **Cosine `T_max`.** `official_cosine_factory`: linear warmup for `warmup_epochs * steps_per_epoch` steps (start factor `0.2`), then cosine over `(epochs - warmup_epochs) * steps_per_epoch` steps to min factor `0.01` (`SequentialLR` with those lengths). `T_max` is **75 epochs of steps**, not 80. Tests assert the boundary step.
- **Official fits must not call `ClassIFit.run`.** `ClassIFit.run` raises `ValueError` naming `omol25_fairchem_4m.run` when `self.loss is fairchem_4m_objective` (missing `attach_omol_elem_refs` and extra-role map). Historical `train_objective` still runs.
- **Keep-5 prune.** Script-local in `omol25_fairchem_4m.run`: retain `last.pt` / `best.pt` plus the last 5 `step_*.pt`; delete older `step_*.pt`. Do not grow molix `CheckpointHook`.
- **Historical ARE units.** Constructor repair on `load_omol25_official` / `fit_atomic_reference` passes `supported=` and `units=` as the actual historical label unit (`kilocalorie_per_mole` name); provenance only, no conversion. Default ARE `units="eV"` is wrong for lstsq kcal values.

### Pin and unresolved metadata

**`OMOL25_FAIRCHEM_4M`** is a mapping with `name: "omol25-fairchem-4m"` and the locked scientific keys from Domain basis (units eV, official 4M, official val/domain view names from the pinned dataset definition, reference digest, RMS `1.433569`, coeffs `10`/`10`, AdamW lr `8e-4` / wd `1e-3`, cosine 80 / warmup 5 / factors `0.2`/`0.01`, clip `100`, bf16 off, tf32 on, refs float64, cutoff `6` Å, `max_neighbors` 30, pad 50, `batch_max_atoms` 400, shuffle seed `0`, eval/ckpt every 2000 steps, keep 5 checkpoints). Keys that are not established on the public configs (`world_size`, `global_batch`, `accumulate_grad_batches`, `adamw_betas`, `adamw_eps`, `adamw_decay_groups`, `monitoring_ids`, `monitoring_seed`, `dataset_revision`, `shard_digests`) exist and are `None`. **`protocol_unresolved(profile) -> list[str]`** (same shape as `production_calibration_drift`) returns those names while they are `None`. **`require_protocol_resolved(profile) -> None`** lives beside it in `configs.py` and raises if that list is non-empty. The official composition site calls `require_protocol_resolved` before constructing a fit. No library default fills the unresolved keys.

**`PROTOCOL_ADAPTATIONS`** is a frozen `tuple[str, ...]` next to the pin — a human ledger, not config the launcher switches on. It names at least: conservative `molix.autograd.Derivative` versus the upstream direct-force head; zero-error structures remaining in the force denominator; Class-I \(E_{\mathrm{MM}}\)/\(F_{\mathrm{MM}}\) converted kcal mol⁻¹ → eV at `ForceAwareModel.energy` via `molix.units`; upstream nonfinite sanitization replaced by fail-on-invalid-real-labels; Class-I pair list (field radius, F21) versus protocol 6 Å / 30 / pad 50; `batch_max_atoms=400` as batch capacity not a chemistry filter.

### Loader fork

`load_omol25_official` and `load_omol25_official_tiny` stay the kcal / `4m`+`val-neutral` / fitted-ε path that `PROD_DAY` / `QUASI_PROD` / `CALIBRATION` / `prod_day.py` / `quasi_prod.py` / `overfit_gate.py` / `align.py` / `scale_audit.py` already call. `AtomCountCache`, `_fit_e0_subsample`, `fit_atomic_reference`, and historical kcal conversion remain that path’s implementation. Do not delete them.

cache-layers-03-refs made dense `AtomicReferenceEnergy(atomic_energies=…)` fail-closed without `supported=` or `atomic_numbers`. Historical constructors in this file that currently pass only `atomic_energies=` (`load_omol25_official`, `load_omol25_official_tiny`, `fit_atomic_reference`, the labeled zeros fallback) must pass an explicit `supported` mask (lstsq 95-slot table: `supported=torch.ones(95, dtype=torch.bool)`, `scheme="empirical_fit"`). That is a constructor repair so the historical path still constructs, not a protocol change.

**`load_omol25_fairchem_4m`** is the official eV 4M loader. It consumes `molhub.cache.Cache` for optional `n_atoms` indexes when present, **fail-closes if the `omol-elem-refs` artifact is absent** (no QR/SVD, no `AtomicDress.fit`, no ARE construction), and does not write `graphs.energy_residual` or `graphs.e0` to disk. Return type is `tuple[LazyOMol25, Mapping[str, LazyOMol25]]`: train plus a **named** eval mapping with keys `general`, `metal`, `electrolytes`, `biomolecules`, `spice`, `neutral_organics` (official val plus the recipe domain views from cache-layers-01-sources: `val` tarball → `general`; `val_metal_complexes` → `metal`; `val_electrolytes` → `electrolytes`; `val_biomolecules` → `biomolecules`; `val_spice` → `spice`; `val_neutralorganics` → `neutral_organics`). Metrics stay per view. Never concatenate overlapping views into one `val` dataset. **`load_omol25_fairchem_4m_tiny`** is a bounded prefix of those same official roles (`subset()` of the official views, or equivalent reconstruction that forwards `energy_unit` and `dressing`); every `LazyOMol25` it builds is constructed with `energy_unit=units.eV` and `dressing=None`. It still fail-closes on missing `omol-elem-refs` and does not fit ε. `batch_max_atoms` is not a keep-predicate on either official loader.

**`LazyOMol25`** is generalized with `energy_unit` (pint unit, default `units.kilocalorie_per_mole`). At `__init__` store `self.energy_unit` and `self._energy_scale = float((1.0 * units.eV).to(energy_unit).magnitude)`. `_sample_from_record` multiplies energy and forces by `self._energy_scale`. Rename `_to_kcal` → `_to_energy_unit` and multiply by `self._energy_scale` (it is no longer kcal-only). **`subset()` forwards both `dressing=self.dressing` and `energy_unit=self.energy_unit`.** Official loaders construct with `energy_unit=units.eV` (scale 1) and `dressing=None`. No `units="kcal"|"eV"` method switch. Module-level `_EV_TO_KCAL` / `_EV_PER_A_TO_KCAL_PER_A` stay for historical `_fit_e0_subsample` only.

### References and residuals

`load_omol_elem_refs()` in `scripts/mm_param_learning/omol25_fairchem_4m.py` is the one owner that returns `AtomicReferenceEnergy` (float64, `scheme="combined_hof_linear"`, cache-layers-03-refs / 04-dtype). Missing artifact raises. `run()` is the single call. `official_class_i_fit` builds `EnergyResidual(coefficients=…, support=…)` by cloning that instance’s `atomic_energies` / `supported` — one table, two uses, not two loads. `EnergyResidual` writes in-flight `graphs.energy_residual` from raw `graphs.energy` minus the frozen table, real atoms only, and is the owner that keeps derivative outputs from overwriting labels. Imported checkpoints keep the `AtomicReferenceEnergy` they already carry; do not splice this table into an unrelated MACE/UMA checkpoint. cache-layers-05 left `EnergyResidual` unexported; this spec is the current caller and exports it from `molix.data.__all__` (and names it in `molix.data.task` as a shipped `BatchTask` subclass). Official code imports `from molix.data import EnergyResidual`. Historical `energy_force_terms` stays the historical kcal residualizer and is not imported or called from `omol25_fairchem_4m.py`.

### ForceAwareModel — one optional table, energy() is the unit owner

Keeps the Class-I compile split: host `physics(batch)` bind versus one `Derivative(self.energy, wrt=self.wrt)` on the traced half. Signature:

```python
ForceAwareModel(imm, dressing=None, *, energy_unit=units.kilocalorie_per_mole, wrt=("atoms.pos",), compiler=None)
```

Historical `ForceAwareModel(imm, dressing)` stays valid. There is no `reference=` kwarg.

- In `__init__`, compute a Python float once: `self._kcal_to_unit = float((1.0 * units.kilocalorie_per_mole).to(energy_unit).magnitude)`. Store `self.energy_unit`. Pint is not called on the hot path. The scale is **not** a buffer / `state_dict` entry. If `dressing is not None` and that scale is not 1, raise `ValueError` (kcal ε must not be added to a non-kcal energy).
- `energy()` **always** multiplies container kcal mol⁻¹ energy by `_kcal_to_unit`, then returns. `Derivative(self.energy)` therefore emits forces in `energy_unit` / Å.
- `forward` adds `dressing` **iff** `self.dressing is not None` (historical `energy_force_terms` path). Official: `dressing=None`, so no add and no third subtract.
- `_RESULT_KEYS` stash stays (historical `train_objective` still reads `graphs.energy`; official labels live on `graphs.energy_residual`).

### ClassIFit stays historical; knobs are live

`run` does not grow a `protocol=` switch, does not bake AllScAIP numbers, and still returns the metrics `dict[str, float]` that `prod_day.py` writes. It gains the same class of caller knobs `prod_day.py` already uses for cadence/clip, plus the ones the official site must inject and that **`model()` / `trainer()` actually read**:

| Knob | Historical default | Official site passes |
|---|---|---|
| `loss=` | `train_objective` | `fairchem_4m_objective` |
| `eval_loss=` | `val_omol_mae` (existing) | `fairchem_4m_objective` (same named objective as train) |
| `optimizer_factory=` | `lambda p: torch.optim.Adam(p, lr=self.lr)` | AdamW `lr=8e-4`, `weight_decay=1e-3` (no invented β/ε/groups in the pin) |
| `lr_scheduler_factory=` | `None` | **not set by `official_class_i_fit`**; `run()` assigns the cosine closure |
| `batch_tasks=` | `()` | `(EnergyResidual(…),)` |
| `energy_unit=` | `units.kilocalorie_per_mole` | `units.eV` |
| `seed=` | `2666` (existing) | pin shuffle seed `0` |

**`__init__` dressing.** Labeled sources: `self.dressing = getattr(train_records, "dressing", None)` — **leave `None` as `None`**. Do not install `AtomicReferenceEnergy(atomic_energies=torch.zeros(95))`. Unlabeled sources still call `fit_atomic_reference` (with `supported=`).

**`model()` is the unit-seam owner on the composed object.** After building `imm`:

```python
cloned = None
if self.dressing is not None:
    cloned = AtomicReferenceEnergy(
        atomic_energies=self.dressing.atomic_energies.detach().cpu().clone(),
        supported=self.dressing.supported.detach().cpu().clone(),
        scheme=self.dressing.scheme,
        units=self.dressing.units,
    )
return ForceAwareModel(imm, dressing=cloned, energy_unit=self.energy_unit).to(self.device)
```

Official: `self.dressing is None` and `self.energy_unit is units.eV` ⇒ `ForceAwareModel(imm, dressing=None, energy_unit=units.eV)` — conversion on, no dressing add, no zeros table.

**`datamodule`** still mounts `self.forcefield.neighbor_list()` then `TotalCharge()`, then `self.batch_tasks` in order, and stores that tuple on the `DataModule` (`DataModule.batch_tasks`). Historical extras stay empty — no `EnergyResidual`. Official extras are `(EnergyResidual(…),)` so the residual runs after the field list. This spec does not add `PadMolecularBatch` to Class-I (packed batches, no ghosts today). If a later caller includes pad in extras, pad must precede `EnergyResidual` so `atoms.mask` exists (cache-layers-05). `energy_huber` / `force_huber` remain the historical kcal functions.

**`trainer(model) -> Trainer`** is the primitive historical `run()` already inlines. It builds `TaskSpec(loss=self.loss)`, `MultiTaskEvalStep` with `self.eval_loss`, `optimizer_factory=self.optimizer_factory`, `lr_scheduler_factory=self.lr_scheduler_factory` (Trainer contract: `factory(optimizer)`), clip/checkpoint hooks from the existing knobs (`CheckpointHook(save_best=True, save_last=True, best_metric_name=("eval", "loss_energy"))`). If `self.loss is fairchem_4m_objective` and `self.lr_scheduler_factory is None`, raise `ValueError` naming `omol25_fairchem_4m.run` as the complete composition. Historical `ClassIFit.run()` is `datamodule` + `model` + `trainer` + `trainer.train` + the metrics dict (factory `None` is legal for `train_objective`). **`ClassIFit.run` raises the same `ValueError` when `self.loss is fairchem_4m_objective`** — official composition must not go through it (missing `attach_omol_elem_refs` and extra-role map). Official composition calls `trainer(model)` itself after installing the cosine so it can `add_module` first and keep the `Trainer` for extra-role `evaluate`.

### Official composition site

`scripts/mm_param_learning/omol25_fairchem_4m.py` is the `prod_day.py` analogue — not `ClassIFit.run`. Public functions (unit-testable; no e2e train inside them):

- **`load_omol_elem_refs() -> AtomicReferenceEnergy`**. The one owner. Fail-closes on a missing `omol-elem-refs` artifact. `scheme="combined_hof_linear"`, `units="eV"`, explicit `supported=`.
- **`official_class_i_fit(profile, train, roles, table) -> ClassIFit`**. Calls `require_protocol_resolved(profile)`. Builds `EnergyResidual` by cloning `table.atomic_energies` / `table.supported`. Returns `ClassIFit(train, roles["general"], loss=fairchem_4m_objective, eval_loss=fairchem_4m_objective, optimizer_factory=AdamW from pin lr/wd only (β/ε/decay-groups remain ledger-only), batch_tasks=(residual,), energy_unit=units.eV, seed=profile shuffle seed 0, clip_max_norm=100, eval_every_n_steps=2000, epochs=80, max_atoms=batch_max_atoms, …)` reading those fields from the pin. Does **not** set `lr_scheduler_factory`. Does not call `model()`, `trainer()`, or `train()`.
- **`official_cosine_factory(steps_per_epoch, profile=OMOL25_FAIRCHEM_4M) -> Callable`**. Trainer-arity `(optimizer) -> scheduler`: linear warmup for `warmup_epochs * steps_per_epoch` (start factor `0.2`), then cosine over `(epochs - warmup_epochs) * steps_per_epoch` to min factor `0.01` (`SequentialLR`). `T_max` is the cosine segment (75 epochs of steps), not 80. Numbers from the pin. Not a second factory signature.
- **`attach_omol_elem_refs(model, table) -> None`**. `model.add_module("omol_elem_refs", table)`. Official checkpoint submodule name is `omol_elem_refs`. Same instance. Not a second `ForceAwareModel` constructor flag.
- **`official_val_datamodule(val_records, *, train_dm) -> DataModule`**. `DataModule(train_dm.train_dataset, val_records, batch_tasks=train_dm.batch_tasks, batch_size=train_dm.batch_size, num_workers=train_dm.num_workers, pin_memory=train_dm.pin_memory, prefetch_factor=train_dm.prefetch_factor, multiprocessing_context=train_dm.multiprocessing_context, seed=train_dm.seed)` — same packing as general eval, not DataModule defaults.
- **`eval_extra_roles(trainer, roles, *, make_val_datamodule) -> None`**. Copy current top-level `state["eval"]` scalars into a local `general` dict **before** the first extra `evaluate`. Under `_without_eval_persistence(trainer)` (every live hook that implements `on_eval_step_complete` pulled off `trainer.hooks`, restored in `finally`), for every key in `roles` except `"general"` call `trainer.evaluate(make_val_datamodule(ds))`, take the non-dict scalars from that return, and after **every** role set `trainer.state["eval"]` to a dict that contains **only** role-name keys (`general` plus every role scored so far). Nested write only; no slash-key writes; no `ClassIFit.val_roles=`; no concatenation. Extra-role `loss_energy` is not compared for `best.pt`. `make_val_datamodule` is the caller’s primitive (official `run` closes over `official_val_datamodule(..., train_dm=dm)`; tests pass a stub).
- **`run()`** (launcher): `require_protocol_resolved(OMOL25_FAIRCHEM_4M)`; `load_omol25_fairchem_4m()`; `table = load_omol_elem_refs()` (the single load); `fit = official_class_i_fit(...)`; `model = fit.model()`; `attach_omol_elem_refs(model, table)`; `dm = fit.datamodule()`; `fit.lr_scheduler_factory = official_cosine_factory(len(dm.train_dataloader()), OMOL25_FAIRCHEM_4M)`; `trainer = fit.trainer(model)`; `trainer.train(dm, …)`; `eval_extra_roles(trainer, roles, make_val_datamodule=lambda ds: official_val_datamodule(ds, train_dm=dm))`. TF32 on / BF16 off is set here from the pin. After `fit.trainer(model)`, set the live `CheckpointHook.save_every_n_steps` from the pin (`2000`) so `step_*.pt` exist to prune; do **not** change `ClassIFit.trainer()`’s historical `save_every_n_steps=0` (that would unbounded-write on `PROD_DAY`). Then `train()`, then retain `last.pt` / `best.pt` plus the last 5 `step_*.pt`; delete older `step_*.pt`. Do not grow molix `CheckpointHook`. Does not call `ClassIFit.run`.

### Losses

`per_atom_mae_energy` and `per_structure_l2_force` are separate functions in `class_i_fit.py`. `per_atom_mae_energy` is the `energy_loss(..., per_atom=True, elementwise=L1) / s_E` wrapper with `s_E` read from `OMOL25_FAIRCHEM_4M` (not a second energy kernel and not a duplicated `1.433569` literal). `per_structure_l2_force` is new and reads `s_F` from the same pin. `fairchem_4m_objective` is `L = 10 L_E + 10 L_F` on `graphs.energy_residual` and `atoms.forces` (eV after the energy-seam conversion). Not `train_objective(method=…)`, not `energy_force_terms`, not `WeightedLoss` (that type requires `nn.Module` terms).

`per_structure_l2_force`: if `atoms.mask` is present, ghosts contribute 0 through that column (multiply, never a boolean gather). If `atoms.mask` is absent, every atom is real — the same contract as `force_loss` when `weight_key` is missing from the batch. Packed Class-I batches have no ghosts today. Every eligible structure, including zero-error, stays in the graph mean.

`QUASI_PROD` / `PROD_DAY` / `CALIBRATION` remain separately named historical pins. They are not aliases of `omol25-fairchem-4m`. This profile needs a fresh production-training-checklist calibration; historical Class-I GO does not transfer.

### Reuse decision

- reuse `molpot.heads.energy.AtomicReferenceEnergy` — `load_omol_elem_refs` is the one official constructor; freeze via `attach_omol_elem_refs` / `add_module("omol_elem_refs", table)`; never fit on this protocol; historical clones pass `supported=`.
- reuse `molix.data.residual.EnergyResidual` (cache-layers-05-residual) — in-flight residual column and label protection; this spec exports it from `molix.data`.
- reuse `molhub.cache.Cache` (cache-layers-02-cache) — optional source `n_atoms` indexes on the official loader.
- reuse `omol_elem_refs` (cache-layers-03-refs / MolHub resource) — published combined vector and digest.
- reuse `molix.units.units` — kcal→eV at `ForceAwareModel.energy`; official labels stay eV; LazyOMol25 instance scale; no copied conversion floats.
- reuse `molix.autograd.Derivative` — conservative forces; named adaptation vs direct head.
- reuse `molix.core.losses.energy_loss` — official \(L_E\) is `energy_loss(target_key="energy_residual", per_atom=True, elementwise=L1) / s_E`; \(N_g =\) `graphs.num_atoms`.
- reuse `production_calibration_drift` pattern — `protocol_unresolved` and `require_protocol_resolved` next to it in `configs.py`.
- reuse `PROD_DAY` / `prod_day.py` pattern — mapping plus a field-by-field launcher; no new protocol class.
- reuse `PadMolecularBatch` — compose `atoms.mask` when a caller pads; this spec does not mount a new padder.
- reuse `TotalCharge` — still mounted on both paths.
- reuse `ENERGY_KEY` / `ENERGY_RESIDUAL_KEY` / `energy_loss(target_key=)` contract — official losses read `graphs.energy_residual`.
- reuse `load_omol25_official` / `load_omol25_official_tiny` / historical `ClassIFit.run` defaults — unchanged callers; `run` still returns the metrics dict.
- reuse `Trainer.evaluate` — extra eval roles; official site is the caller and isolates persistence around it.
- reuse `Trainer.lr_scheduler_factory(optimizer)` arity — do not invent a second factory signature.
- reuse `force_loss` missing-`weight_key` behaviour — missing `atoms.mask` ⇒ every atom is real (mask contract only; not the reduction).
- reuse `CheckpointHook` — do not grow it; official extra-role eval removes it from the live hook list then restores it.
- do not reuse `AtomCountCache`, `_fit_e0_subsample`, `_e0_cache_path`, `fit_atomic_reference`, `molhub.dataset.transforms.AtomicDress` on the official path — historical still owns those symbols; do not delete them in this spec.
- do not reuse `GlobalRescale` — RMS `1.433569` divides inside `per_atom_mae_energy` / `per_structure_l2_force` only.
- do not reuse `force_loss` for \(L_F\) — it is not per-structure vector L2; **new** `per_structure_l2_force`.
- do not reuse `energy_force_terms` on the official path — historical kcal residualizer only.
- do not reuse `WeightedLoss` — it requires `nn.Module` terms; `fairchem_4m_objective` is `10 L_E + 10 L_F`.
- generalize `LazyOMol25` in `scripts/mm_param_learning/class_i_fit.py` with `energy_unit` (default kcal), instance scale, `_to_energy_unit`, and `subset()` forwarding `energy_unit` + `dressing`.
- generalize `ForceAwareModel` in `scripts/mm_param_learning/class_i_fit.py` to keep the host-bind vs `Derivative` split, convert kcal→`energy_unit` in `energy()` with a Python float, and take **one** optional `dressing=` (no `reference=`).
- generalize `ClassIFit` in `scripts/mm_param_learning/class_i_fit.py` with `loss=`, `optimizer_factory=`, `lr_scheduler_factory=`, `batch_tasks=`, `eval_loss=`, `energy_unit=` actually forwarded by `model()` / `trainer()`; add `trainer(model)` as the Trainer primitive (refuses official objective without a cosine factory); historical defaults unchanged.
- new — `OMOL25_FAIRCHEM_4M` mapping in existing `configs.py` because no current pin is the FAIR 4M recipe.
- new — `load_omol25_fairchem_4m` / `load_omol25_fairchem_4m_tiny` because retargeting `load_omol25_official` would silently mix eV 4M into historical launchers.
- new — `per_atom_mae_energy` (wrapper over `energy_loss`) / `per_structure_l2_force` / `fairchem_4m_objective` because `energy_huber` / `force_huber` are a different named objective.
- new — `scripts/mm_param_learning/omol25_fairchem_4m.py` as the one official composition site (`load_omol_elem_refs`, `official_class_i_fit`, `official_cosine_factory`, `attach_omol_elem_refs`, `official_val_datamodule`, `eval_extra_roles`, `run`).

## Files to create or modify

- `scripts/mm_param_learning/configs.py`
- `scripts/mm_param_learning/class_i_fit.py` (`load_omol25_official`, `load_omol25_official_tiny` kept historical; add `load_omol25_fairchem_4m`, `load_omol25_fairchem_4m_tiny`)
- `scripts/mm_param_learning/omol25_fairchem_4m.py` (new)
- `src/molix/data/__init__.py`
- `src/molix/data/task.py`
- `tests/test_scripts/test_mm_param_learning/test_configs.py` (new)
- `tests/test_scripts/test_mm_param_learning/test_class_i_fit.py`
- `tests/test_scripts/test_mm_param_learning/test_omol25_fairchem_4m.py` (new)

## Tasks

- [x] Write failing unit tests for OMOL25_FAIRCHEM_4M (tests/test_scripts/test_mm_param_learning/test_configs.py → TestOmol25Fairchem4m)
- [x] Implement OMOL25_FAIRCHEM_4M, PROTOCOL_ADAPTATIONS, protocol_unresolved, and require_protocol_resolved in scripts/mm_param_learning/configs.py
- [x] Write failing unit tests for load_omol25_fairchem_4m, LazyOMol25 instance scale and subset forwarding, ForceAwareModel energy-unit seam, protocol losses, and ClassIFit.model/trainer knobs (tests/test_scripts/test_mm_param_learning/test_class_i_fit.py → TestLoadOmol25Fairchem4m, TestLazyOMol25, TestForceAwareModel, TestPerAtomMaeEnergy, TestPerStructureL2Force, TestClassIFit)
- [x] Implement load_omol25_fairchem_4m and load_omol25_fairchem_4m_tiny in scripts/mm_param_learning/class_i_fit.py; keep load_omol25_official and load_omol25_official_tiny historical; generalize LazyOMol25 with instance energy_unit scale, _to_energy_unit, and subset() forwarding energy_unit and dressing; pass supported= on historical AtomicReferenceEnergy constructors
- [x] Generalize ForceAwareModel in scripts/mm_param_learning/class_i_fit.py to keep physics() vs Derivative(self.energy), convert kcal mol⁻¹ to energy_unit in energy() via a Python float from molix.units, and take one optional dressing= (no reference=)
- [x] Generalize ClassIFit in scripts/mm_param_learning/class_i_fit.py so model() is ForceAwareModel(imm, dressing=clone-or-None, energy_unit=self.energy_unit), dressing None stays None, trainer(model) builds the Trainer and refuses fairchem_4m_objective without a cosine factory, loss=/optimizer_factory=/lr_scheduler_factory=/batch_tasks=/eval_loss=/energy_unit= are the knobs that trainer/model read, and per_atom_mae_energy / per_structure_l2_force / fairchem_4m_objective live beside train_objective without calling energy_force_terms
- [x] Export EnergyResidual from src/molix/data/__init__.py and name it in src/molix/data/task.py
- [x] Write failing unit tests for official_class_i_fit, load_omol_elem_refs, attach_omol_elem_refs, eval_extra_roles persistence isolation and one-home eval, official_val_datamodule, and require_protocol_resolved (tests/test_scripts/test_mm_param_learning/test_omol25_fairchem_4m.py → TestOmol25Fairchem4mScript)
- [x] Implement official composition site scripts/mm_param_learning/omol25_fairchem_4m.py (official_class_i_fit passes eval_loss=fairchem_4m_objective and energy_unit=eV and does not set the cosine; run is the complete composition; eval_extra_roles isolates CheckpointHook/journal hooks and writes only role-name keys; attach_omol_elem_refs registers the load_omol_elem_refs instance)
- [x] Run full check + test suite

## Testing strategy

Unit tests only, mirrored under `tests/test_scripts/test_mm_param_learning/`. One function per test. Green with `python -m pytest {path} -v` on that file only. No e2e under `tests/`. No FAIR/ASE/MACE/e3nn/lammps/openmm imports in `src/`, `tests/`, or `scripts/`. Goldens are literals; third-party oracles are offline-only. No `hasattr` / signature-default / omitted-kwarg-identity tests. This repo has no `regressions/` (deleted 2026-09-07); the public-API golden is the mirrored unit modules below.

`TestOmol25Fairchem4m` (`test_configs.py`):

- Happy: `OMOL25_FAIRCHEM_4M["name"] == "omol25-fairchem-4m"`; lr `8e-4`; wd `1e-3`; epochs `80`; warmup epochs `5`; start factor `0.2`; min factor `0.01`; clip `100`; bf16 false; tf32 true; refs dtype float64; RMS `1.433569` on energy and forces; energy/force coeffs `10`/`10`; cutoff `6.0` Å; `max_neighbors` 30; neighbor pad `50`; `batch_max_atoms` 400; shuffle seed `0`; eval/ckpt `2000`; keep `5`; reference SHA-256 `e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640`; FAIR commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`; energy unit `"eV"`.
- Edge: `protocol_unresolved` is non-empty while world size / AdamW betas / monitoring IDs / dataset revision are `None`; `require_protocol_resolved` raises on that list. Filling a fake world size in the test does **not** ship as a default.
- Edge: profile has no keys `val-neutral`, Huber beta, `8192`, `FORCE_OUTLIER_KCAL_A`, or kcal conversion. `PROTOCOL_ADAPTATIONS` is a `tuple[str, …]` that names conservative `Derivative`, the zero-error denominator, and the kcal→eV energy-seam conversion. Launchers do not branch on those strings. Doc/repr does not claim eSEN/MACE/UMA reproduction.
- Domain: `batch_max_atoms` is a distinct key from any chemistry `max_atoms` filter; 400 is not used as a keep-predicate in the profile.

`TestAtomCountCache` and `TestForceHuber` stay — they own historical symbols this spec does not delete. Do not add tests that `AtomCountCache` / `fit_atomic_reference` are absent.

`TestLoadOmol25Fairchem4m`: return value is `(train, roles)` with `roles` a mapping whose keys are exactly `general`, `metal`, `electrolytes`, `biomolecules`, `spice`, `neutral_organics`; the values are distinct datasets (no concatenation, overlapping views remain separate). Missing `omol-elem-refs` fails closed (no ARE constructed by the loader). `load_omol25_official` still returns `tuple[LazyOMol25, LazyOMol25]` and still uses `val_role="val-neutral"`. Official datasets have `dressing is None`. Tiny official views keep eV labels (`1.0` eV stays `1.0`).

`TestLazyOMol25`: official construction with `energy_unit=units.eV` keeps a raw `1.0` eV as `1.0`; historical default still converts through `molix.units` to kcal mol⁻¹. `subset()` of an eV view still keeps `1.0` eV as `1.0` (reconstruction forwards `energy_unit`; it does not fall back to `_EV_TO_KCAL`). Counts on the official path come from Cache/metadata, not a `root:len` npy path.

`TestForceAwareModel`: `ForceAwareModel(imm, dressing)` still constructs and `forward` still adds dressing (historical kcal objective on a stub batch). `physics` then `Derivative` split remains (`energy` seam still goes through `self.energy` after `physics`). With `energy_unit=units.eV` and `dressing=None`, a stub that writes `1.0` kcal mol⁻¹ makes `energy()` write `1.0 * (1.0 * units.kilocalorie_per_mole).to(units.eV).magnitude` (compute the expected scale in the test via `molix.units`, do not paste a float); `forward` does not add a table. `ForceAwareModel(imm, dressing=table, energy_unit=units.eV)` raises. There is no `reference=` argument to pass.

`TestClassIFit`: on a stub labeled batch, historical `train_objective` / Huber still produces the kcal residual+inlier value (owned objective, not “when knobs omitted”). `ClassIFit(..., energy_unit=units.eV)` with `train_records.dressing is None` leaves `fit.dressing is None`; `fit.model()` is a `ForceAwareModel` whose `energy_unit` is eV and whose `dressing` is `None` (no zeros table). `ClassIFit(..., loss=fairchem_4m_objective)` with `lr_scheduler_factory is None` makes `trainer(stub)` raise. After assigning a Trainer-arity dummy factory, `fit.trainer(stub_module)` uses the injected `loss=` / `eval_loss=` / `optimizer_factory=` on a stub batch. `datamodule` with `batch_tasks=(EnergyResidual(…),)` includes that task after `TotalCharge`. Import is `from molix.data import EnergyResidual`.

`TestPerAtomMaeEnergy` / `TestPerStructureL2Force` (hard-coded, physical units eV / eV Å⁻¹):

- Two graphs, hand-set `c[1] = -10.0` eV (not a claim about `omol_elem_refs` numerics). Graph A: \(N=2\), \(E_{\mathrm{DFT}} = -20.0 + 1.0\), so \(E_{\mathrm{res}}=1.0\). Graph B: \(N=4\), \(E_{\mathrm{DFT}} = -40.0 + 1.0\), \(E_{\mathrm{res}}=1.0\). Model residual `0` on both. \(L_E = \tfrac12(1/(s\cdot 2) + 1/(s\cdot 4))\) with \(s=1.433569\). Assert the literal `0.5 * (1.0 / (1.433569 * 2) + 1.0 / (1.433569 * 4))` within `1e-10`. This is **not** the atom-weighted value `2.0 / (1.433569 * 6)`. \(N_g\) is `graphs.num_atoms`.
- Forces: three structures, the third with `F_pred == F_ref` (zero error). \(L_F\) divides by `3`, not `2`. A batch **without** `atoms.mask` treats every atom as real. A batch **with** `atoms.mask` zeros ghost rows. Do not require a padder on the official path.
- `fairchem_4m_objective` does not call `energy_force_terms` (it reads `graphs.energy_residual`).
- Do **not** add an \(E_{\mathrm{tilde}}=E_{\mathrm{res}}/s\) unscale-`Derivative` golden; no scaler is composed.

`TestOmol25Fairchem4mScript` (`test_omol25_fairchem_4m.py`) — no `trainer.train`, no DataLoader epoch:

- `official_class_i_fit(OMOL25_FAIRCHEM_4M, …)` raises via `require_protocol_resolved`. A test-local filled copy (not shipped) constructs a `ClassIFit` whose `eval_loss` and `loss` are `fairchem_4m_objective`, whose `energy_unit` is eV, and whose `lr_scheduler_factory is None`. `trainer(stub)` raises until a Trainer-arity factory is assigned; then `eval_step` on a stub batch equals `fairchem_4m_objective(preds, batch)`.
- `load_omol_elem_refs` fail-closes on a missing artifact. `attach_omol_elem_refs(model, table)` registers submodule name `omol_elem_refs` and `model.omol_elem_refs is table`. `EnergyResidual` used by `official_class_i_fit` is cloned from that table’s `atomic_energies` / `supported` (one table, two uses). Import is `from molix.data import EnergyResidual`.
- `eval_extra_roles` one-home: start from `state["eval"]["loss"]` / `loss_energy` as general scalars; after extra roles, `state["eval"]` keys are **only** role names (`general` plus every non-`general` key passed in); leftover top-level `loss` is gone; `state["eval"]["general"]` still holds the pre-extra scalars.
- `eval_extra_roles` persistence isolation: real `CheckpointHook(save_best=True, save_last=True, best_metric_name=("eval", "loss_energy"))` with a seeded `_best_value` and a stub `evaluate` that writes a *better* extra-role `loss_energy` and fires `on_eval_step_complete` on the live `trainer.hooks`. After `eval_extra_roles`, `_best_value` is unchanged and `best.pt` / `last.pt` were not rewritten from the extra role. A stub journal hook’s `on_eval_step_complete` must not run during extra-role evaluate. `make_val_datamodule` may be a lambda here.
- `official_val_datamodule(ds, train_dm=dm).batch_tasks is dm.batch_tasks` (neighbor_list + `TotalCharge` + extras, not a residual-only list).

Repo policy deleted `regressions/` (2026-09-07). The public-API golden for this spec is `tests/test_scripts/test_mm_param_learning/test_configs.py` with the literals above (FAIR commit, RMS, SHA-256, coeffs). Comment provenance: FAIR commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`, files `omol_4m.yml` / `omol_direct.yaml` / `uma_v1_hof_lin_refs.yaml`, date 2026-09-11. No third-party runtime.

Static: `scripts/mm_param_learning/` and `tests/test_scripts/test_mm_param_learning/` contain no `ase`, `e3nn`, `mace`, `fairchem`, `lammps`, or `openmm` imports.

## Out of scope

- PackedCache, FrameStore rename, `molix.block/1` wire changes.
- molnex `src/` importing molhub.
- Deleting `molix.data.NeighborList`; changing derivation-head / `Derivative` semantics.
- Implementing `Cache`, `omol_elem_refs` arithmetic, `EnergyResidual` arithmetic, or float64 column policy (those are cache-layers-02…05). This spec only exports and mounts `EnergyResidual`.
- Deleting `AtomCountCache`, `_fit_e0_subsample`, or `fit_atomic_reference` — historical `load_omol25_official` still uses them.
- Deleting historical `energy_force_terms` — it remains the kcal residualizer; official must not call it.
- Rewriting `prod_day.py` / `quasi_prod.py` / `overfit_gate.py` / `align.py` / `profile_class_i.py` onto the official loader. (`profile_class_i.py` still constructs a dense zeros table; not this slug’s caller.)
- Changing `ClassIFit.run`’s return type from the metrics dict (historical launchers write that dict).
- Adding `ClassIFit.val_roles=` or concatenating official domain views.
- Adding `ForceAwareModel(reference=)` or any second table kwarg.
- Growing `CheckpointHook` with a pause API or keep-5 API (official site isolates by removing the hook from the live list; script-local retention of 5 snapshots only).
- Changing `Trainer.evaluate` into a scoring primitive that skips hooks.
- Generic QR/SVD train-only reference fit (different named protocol; not a fallback).
- Rewriting `QUASI_PROD` / `PROD_DAY` / `CALIBRATION` into this protocol, or inheriting historical Class-I production GO.
- Direct-force head, DDPMTLoss zero-error drop, eSEN/MACE/UMA reproduction claims, `GlobalRescale` on the official path.
- Mounting `PadMolecularBatch` on Class-I (packed path has no ghosts; pad-before-residual is a later compile/static-shape concern).
- Inventing world size / AdamW β / monitoring seeds as if resolved.
- `benchmarks/bench_class_i_omol.py` (not on `ruff check scripts/`).
- Production job submission. `.claude/notes/production-training-checklist.md` still applies; this profile needs a new molexp calibration experiment.
