---
slug: cache-layers-03-refs
created: 2026-09-11
criteria:
  - id: ac-001
    summary: Unsupported Z raises; dense zeros are not support
    type: code
    pass_when: |
      TestAtomicReferenceEnergy in
      tests/test_molpot/test_heads/test_energy.py: a table supported only
      on Z={1,6,8,26,36} raises ValueError listing 7 from
      AtomicReferenceEnergy.validate and from eager pooled on Z containing
      nitrogen; a supported slot with c[Z]=0.0 does not raise; Z=999 is
      not clipped; dense __init__ without atomic_numbers and without a
      boolean supported mask raises; masked-out unsupported Z does not
      raise.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: c[Z] is float64 independent of ftype and .float()
    type: code
    pass_when: |
      AtomicReferenceEnergy built under default config ftype float32 has
      atomic_energies.dtype torch.float64, an empty parameter list, and
      persistent buffers atomic_energies and supported in state_dict.
      module.float() leaves atomic_energies as float64. load_state_dict
      restores those buffers plus scheme and units extra_state.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Float64 cancellation recovers a meV-scale residual
    type: scientific
    evaluator_hint: path: tests/test_molpot/test_heads/test_energy.py
    pass_when: |
      Using hard-coded omol_elem_refs c[Kr]=-74932.45491967947 eV, a 350-Kr
      graph has e0=350*c[Kr] in float64; E_DFT=e0+0.35; float64 E_DFT-e0
      equals 0.35 eV within 1e-8 eV (a few binary64 ulps of |e0|~2.6e7).
      The same subtraction after casting
      e0 through float32 disagrees by more than 0.1 eV. Water 2*c[H]+c[O]
      equals -2079.7006341793223 eV in float64.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Unit-test goldens embed omol_elem_refs literals
    type: runtime
    pass_when: |
      python -m pytest tests/test_molpot/test_heads/test_energy.py -v
      is green. TestAtomicReferenceEnergy lookup asserts the literals
      H=-16.001316695716987, C=-1037.0951083942327, O=-2047.6980007878883,
      Fe=-34381.61625929281, Kr=-74932.45491967947 eV and comments SHA-256
      e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640.
      The file does not import molhub or fairchem and does not live under
      regressions/.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: pooled sums every row unless a structural mask is passed
    type: code
    pass_when: |
      TestAtomicReferenceEnergy: Z=[8,1,1,1] with the last row a ghost
      hydrogen, pooled without mask equals water e0 plus c[H]; pooled with
      mask=[True,True,True,False] equals water e0 only. pooled still
      delegates the reduction to EnergyAggregation(pooling="sum").
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Frozen mapping only; encoder and Checkpoint untouched
    type: code
    pass_when: |
      src/molpot/heads/energy.py contains no QR/SVD/lstsq fitter.
      AtomicReferenceEnergy still constructs as
      AtomicReferenceEnergy(atomic_energies=..., atomic_numbers=...).
      src/molzoo/mace/encoder.py, src/molix/core/checkpoint/, and
      src/molpot/heads/rescale.py are unmodified. src/molpot and
      tests/test_molpot do not import molhub.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means `AtomicReferenceEnergy` is a float64 frozen `Z → c[Z]` map with
an explicit support mask, fail-closed unsupported elements, and optional
structural masking on `pooled`, verified by the mirrored unit tests with
hard-coded `omol_elem_refs` literals.

## AC-001 — Unsupported Z raises; dense zeros are not support

The old lookup treated every dense slot as legal and returned `0.0` for
missing elements. Support is now a boolean buffer filled only from
`atomic_numbers` or an explicit `supported` argument. `validate` and eager
`pooled` raise `ValueError` listing the unique bad Z, matching
`MACEEncoder.validate_elements`.

## AC-002 — c[Z] is float64 independent of ftype and `.float()`

`config["ftype"]` stays the compute dtype for MLP heads. The reference
table does not follow it, and `_apply` snaps it back after a module-wide
cast so Trainer `.float()` cannot destroy the cancellation contract.
`Checkpoint` is reused because these are ordinary persistent buffers.

## AC-003 — Float64 cancellation recovers a meV-scale residual

Textbook case: 350 krypton atoms, residual `0.35` eV. Binary32 ulp of the
sum is ~2 eV, larger than the residual. Citations: Levine 2025,
Tran OC22 (158)–(159), FAIR DoubleTensor comment, YAML SHA-256 above.

## AC-004 — Unit-test goldens embed omol_elem_refs literals

Project rule: no `regressions/` directory. The public-API scenario is the
mirrored `TestAtomicReferenceEnergy` class. Literals are copied offline
from `uma_v1_hof_lin_refs.yaml` key `omol_elem_refs`; the test never talks
to GitHub or MolHub.

## AC-005 — pooled sums every row unless a structural mask is passed

`pad_atom_type=1` is hydrogen, a supported element. Without a mask those
ghost rows add `c[1]`. The multiply-before-sum is the caller's structural
mask, not the support mask. Residual (later slug) is expected to pass
`atoms.mask`.

## AC-006 — Frozen mapping only; encoder and Checkpoint untouched

This slug generalizes one module. It does not grow a fitter, a cache, or
a molzoo edit. `GlobalRescale` remains the orthogonal frozen affine.
