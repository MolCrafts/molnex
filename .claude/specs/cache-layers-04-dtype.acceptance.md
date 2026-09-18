---
slug: cache-layers-04-dtype
created: 2026-09-11
criteria:
  - id: ac-001
    summary: Frame collate keeps energy/e0 float64 while pos follows dtype
    type: runtime
    evaluator_hint: "path: tests/test_molix/test_data/test_collate.py"
    pass_when: |
      python -m pytest tests/test_molix/test_data/test_collate.py -v is green.
      collate_molecules(frames, dtype=torch.float32) and MoleculeCollator(dtype=torch.float32)
      emit atoms.pos as float32 and graphs.energy as float64. When meta e0 is present,
      graphs.e0 is float64. An atoms extra named energy follows dtype. Energy value
      -74800.123456789 eV matches the float64 literal exactly; hard-coded
      ulp32(74800.0)=0.0078125 eV; abs(float32(E)-E)>0.001 eV. TensorFrame.from_tensordict
      shares data_ptr on both leaves. No regressions/ script.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: batch_to named_apply skips ENERGY_KEY and E0_KEY only
    type: runtime
    evaluator_hint: "path: tests/test_molix/test_data/test_dtype_propagation.py"
    pass_when: |
      python -m pytest tests/test_molix/test_data/test_dtype_propagation.py -v is green.
      batch_to on a TensorDict uses named_apply(..., nested_keys=True) and leaves
      ENERGY_KEY and E0_KEY as float64 while casting atoms.pos and ATOMIC_ENERGY_KEY.
      Frame-route _CollateFn under fp32 and mixed precision modes emits pos at ftype
      and graphs.energy at float64. torch.autocast does not change that energy dtype.
      A FrameStore energy leaf that arrives float32 is not upcast via .double().
      src/molix/core/steps/__init__.py batch_to and src/molix/data/datamodule.py
      contain no .double() repair of energy. Plain-dict recurse is not NestedKey-aware.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: DataMixture atom_counts and pair_counts are planning snapshots
    type: runtime
    evaluator_hint: "path: tests/test_molix/test_data/test_mixture.py"
    pass_when: |
      python -m pytest tests/test_molix/test_data/test_mixture.py -v is green.
      DataMixture.atom_counts and DataMixture.pair_counts are detached cloned long
      tensors whose data_ptr differs from the source; mutating the source tensors
      does not change the mixture. A source without atom_counts or without
      pair_counts raises ValueError and never calls __getitem__.
      src/molix/data/mixture.py _source_atom_counts and _source_edge_counts do not
      build a full-length Python list via range(len(source)) and do not zeros-fill
      a missing vector. TokenBudgetBatchSampler and PadMolecularBatch source files
      are unmodified.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: E0_KEY and FLOAT64_LABEL_KEYS live in schema; no residual key
    type: code
    pass_when: |
      molix.schema.E0_KEY == ("graphs", "e0") and molix.schema.FLOAT64_LABEL_KEYS
      == frozenset({ENERGY_KEY, E0_KEY}). collate_molecules and batch_to import
      FLOAT64_LABEL_KEYS from molix.schema. molix.schema does not define
      ENERGY_RESIDUAL_KEY. TensorFrame / as_tensor_frame gain no dtype/cast methods.
      No PrecisionPolicy type and no batch_to(..., preserve=) knob exist under src/.
      molix.units is not given a kcal conversion at this layer; collate and batch_to
      docstrings say binary64 / Frame units, not eV.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Kr-scale float64 total survives Frame collate at ftype float32
    type: scientific
    evaluator_hint: "path: tests/test_molix/test_data/test_collate.py"
    pass_when: |
      Frame collate of E_raw = -74800.123456789 eV at dtype=torch.float32 writes
      graphs.energy whose bits equal the float64 literal; ulp32(74800.0) is asserted
      as 0.0078125 eV; abs(float32(E_raw) - E_raw) > 0.001 eV. Provenance comment
      names IEEE-754, Levine arXiv:2505.08762, FAIR commit 41d36dc6. No fairchem
      import at test time.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Mixed-dtype energy MSE is halted; Residual is not implemented here
    type: code
    pass_when: |
      src/molix/core/losses/molecular.py is unmodified by this spec. No Residual
      type, no ENERGY_RESIDUAL_KEY, and no mixed-dtype cast inside energy_loss
      exist under src/molix as part of this slug. Out of scope in the spec body
      states energy MSE on mixed-dtype batches is unusable until cache-layers-05.
    status: verified
    last_checked: 2026-09-12
out_of_scope:
  - Residual / ENERGY_RESIDUAL_KEY / energy_loss mixed-dtype fix (cache-layers-05)
  - AtomicReferenceEnergy, Cache, NeighborList, FrameStore wire dtypes, molpot, scripts
  - PrecisionPolicy, preserve= knob, regressions/ scripts
---

# Acceptance criteria

Done means Frame collate emits binary64 graph totals while compute leaves follow `ftype`, store collate neither narrows nor upcasts those totals, `batch_to` skips `FLOAT64_LABEL_KEYS` by NestedKey on TensorDicts, and `DataMixture` owns frozen count snapshots. Residualization, schema residual keys, and mixed-dtype energy MSE stay later specs — named as a halt, not a silent break.

## AC-001 — Frame collate keeps energy/e0 float64 while pos follows dtype

`tests/test_molix/test_data/test_collate.py` is the public-API regression example (no `regressions/` tree). Kr-scale total `-74800.123456789` eV is the golden; `0.0078125` eV is the hard-coded binary32 ulp. NestedKey skip: atoms extra `energy` is not protected.

## AC-002 — batch_to named_apply skips ENERGY_KEY and E0_KEY only

`apply(fn)` cannot see nested paths and a leaf-name skip would cast `ATOMIC_ENERGY_KEY`. Frame route must show mixed dtypes on one batch. Store route must not pretend a float32 mmap column is float64. Autocast is a dtype assertion, not a new type. Dict recurse is documented TensorDict-only.

## AC-003 — DataMixture snapshot immutability

Planning boundary is `DataMixture.__init__`. Live mmap is not a second authority. `pair_counts` clones the same way as `atom_counts`. `TokenBudgetBatchSampler` keeps reading `dataset.atom_counts` / `dataset.pair_counts`.

## AC-004 — schema addresses; no residual key

`E0_KEY` is required (architect 🔴). `ENERGY_RESIDUAL_KEY` is forbidden in this slug. `FLOAT64_LABEL_KEYS` is the one skip set.

## AC-005 — Kr-scale scientific golden

Same literals as AC-001, scored as a scientific comparison against IEEE-754 / Levine / FAIR DoubleTensor.

## AC-006 — mixed-dtype MSE halt

Do not “fix” `energy_loss` here and do not land Residual. Training on raw totals under mixed dtypes is explicitly unusable until cache-layers-05.
