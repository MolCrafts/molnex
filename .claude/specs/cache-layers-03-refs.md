---
title: Generalize AtomicReferenceEnergy to a float64 frozen Z→c[Z] mapping
status: code-complete
grilled: true
created: 2026-09-11
chain: cache-layers
depends: cache-layers-01-sources
supersedes: cache-layers-01-cache (reference-state slice only)
---

# Generalize AtomicReferenceEnergy to a float64 frozen Z→c[Z] mapping

## Summary

`molpot.heads.energy.AtomicReferenceEnergy` becomes the generic frozen
element-to-coefficient map: `c[Z]` is stored and applied in float64
regardless of `config["ftype"]`, an explicit boolean support mask decides
which atomic numbers are legal (dense zeros do not), and unsupported `Z`
raises instead of returning a silent zero or clipping into range. `pooled`
still sums every row it is given; ghost atoms whose placeholder `Z` is a
real element (default `pad_atom_type=1` hydrogen) contribute unless the
caller passes a structural mask. The module is not an OMol25 fitter and
does not own residual labels. `MACEEncoder` construction is unchanged;
`GlobalRescale` stays the frozen affine sibling; `Checkpoint` keeps saving
whatever `state_dict` already contains.

## Domain basis

Isolated-atom, published combined heat-of-formation plus linear, and
empirical fitted tables are three different scientific objects. This type
is only the frozen map `Z ↦ c[Z]` (eV). It does not fit `c`.

Levine et al., "OMol25: the Open Molecules 2025 dataset"
https://arxiv.org/html/2505.08762v1

Tran et al., "The Open Catalyst 2022 (OC22) dataset"
https://arxiv.org/html/2206.08917 eqs (158)–(159)

FAIR `omol_direct.yaml` (`fairchem@41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`,
`configs/allscaip/tasks/omol_direct.yaml`) pins the table as
`torch.DoubleTensor` with the comment
"do to the scale of the numbers this needs to be a double".

SHA-256 of the complete upstream YAML bytes
(`configs/allscaip/element_refs/uma_v1_hof_lin_refs.yaml`), checked
2026-09-11:

```text
e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
```

Per graph `g`, with `c[Z]` in eV and the sum over **real** atoms only:

```text
e0[g]     = sum_i c[Z_i]          # eV, float64
E_res     = E_DFT - e0            # eV, float64
E_total   = E_model_res + e0      # eV
```

`e0` is a function of atomic numbers only, so `∂e0/∂r = 0` and adding or
subtracting it leaves position forces and Hessians unchanged.

Precision: binary32 `ulp(|c_Kr| ~ 7.5e4 eV) ~ 8 meV`; a 350-Kr sum has
`ulp ~ 2 eV`. The residual that training sees lives at meV/atom. Casting
the already-rounded sum with a late `.double()` cannot recover those bits.
The OMol25 combined table (`omol_elem_refs`) is the published HOF+linear
vector, not a physical isolated-atom energy, and not a local QR/SVD fit.

Selected `omol_elem_refs[Z]` literals (eV; index = `Z`; offline extract
from that YAML on 2026-09-11, no fairchem/molhub import):

| Z | element | c[Z] / eV |
|---|---------|-----------|
| 1 | H | `-16.001316695716987` |
| 6 | C | `-1037.0951083942327` |
| 8 | O | `-2047.6980007878883` |
| 26 | Fe | `-34381.61625929281` |
| 36 | Kr | `-74932.45491967947` |

Water-like `2 H + O` reference:
`e0 = 2 c[H] + c[O] = -2079.7006341793223 eV`.
350-Kr reference: `e0 = 350 c[Kr] = -26226359.221887815 eV`.

## Design

`AtomicReferenceEnergy` remains the single `nn.Module` in
`src/molpot/heads/energy.py`. No new type, no fitter, no BatchTask, no
TensorFrame columns. `MACEEncoder` already constructs it as
`AtomicReferenceEnergy(atomic_energies=spec.atomic_energies, atomic_numbers=spec.atomic_numbers)`;
that call stays valid. New arguments are keyword-only with defaults.

**Buffers (persistent, not `nn.Parameter`, in `state_dict`).**

- `atomic_energies`: dense Z-indexed `(max_z + 1,)`, **always**
  `torch.float64`, never `config["ftype"]`.
- `supported`: same length, `torch.bool`. `True` only for Z the
  constructor was told about. A stored `0.0` is a legal coefficient when
  `supported[Z]` is true (isolated-atom tables routinely have a zero
  slot). A stored `0.0` with `supported[Z]` false is not a coefficient.

**Construction.**

- `atomic_numbers` given: scatter `atomic_energies` (table order) into the
  dense lookup; `supported` is true exactly at those Z. `max_z` still
  sizes the table. Duplicate Z raises `ValueError`.
- `atomic_numbers` omitted: `atomic_energies` is already dense. `supported`
  (bool, same length) is **required**. Missing both is `ValueError` —
  dense zeros do not imply support.
- `scheme`: `"isolated_atom"` (default, MACE E0), `"combined_hof_linear"`
  (published HOF+linear such as `omol_elem_refs`), or `"empirical_fit"`.
  Any other string raises. This label does not change arithmetic.
- `units`: `"eV"` by default. No conversion floats in this slug.
- `scheme` and `units` ride in `get_extra_state` / `set_extra_state` so
  they survive `state_dict` (and therefore `molix.core.checkpoint.Checkpoint`,
  which already serializes `model.state_dict()`).

Input values may arrive as `config["ftype"]` (MACE spec tensors). `__init__`
casts the lookup to float64 once. There is no QR, SVD, or `lstsq`.

**`_apply`.** `nn.Module.float()` / `.to(dtype=float32)` would otherwise
downcast the lookup. Override `_apply` (same idea as
`GTOHartreeEnergy._apply` / `DensityInteraction._apply`): after
`super()._apply`, recast `atomic_energies` to float64 on whatever device
the buffer now lives. Bool `supported` is not a floating buffer and is
left to PyTorch.

**Support vs structural mask.**

Support answers "does this *element* have a defined `c[Z]`?".
`atoms.mask` answers "is this *row* a real atom?".
`PadMolecularBatch(pad_atom_type=1)` writes hydrogen on ghost rows.
Hydrogen is a supported element on any H-containing table; those rows
must not count in `e0` unless the caller says they are real.

- `validate(Z, mask=None) -> None`: eager, host sync. Among **unmasked**
  rows, out-of-range Z or `supported[Z] is False` raises `ValueError`
  listing the unique offending atomic numbers (same shape as
  `MACEEncoder.validate_elements`). No clip onto `max_z`. Masked-out rows
  are not checked for support. Every Z, masked or not, must still lie in
  `[0, table_size)` so the lookup is a legal index; out-of-range with a
  mask still raises. Not a hot-path method.
- `forward(Z, mask=None) -> (N,) float64`: compile-clean index into
  `atomic_energies`. Optional `mask` is a multiply, never a slice
  (`values * mask.to(dtype=values.dtype)`), matching
  `MacePotential._sum_energy`. **Does not** call `validate` — MACE's
  `energy_core` calls `self.atomic_energies(Z)` on the hot path and
  already has a one-shot eager `validate_elements`. Putting a host sync
  here would tax every eager MACE step.
- `pooled(Z, batch, num_graphs, mask=None) -> (B,) float64`: the dressing
  API. Eager (`not torch.compiler.is_compiling()`) calls `validate` then
  `EnergyAggregation(pooling="sum")` on `forward(Z, mask)`. Under dynamo
  the validate branch folds away (tracing-time constant), same contract
  as `MacePotential.energy`. **Without `mask`, pooled sums every given
  row.** Ghost hydrogen contributes `c[1]`. Masking is the caller's job;
  the later Residual BatchTask passes `atoms.mask`. This slug does not
  invent Residual.

`∂e0/∂r = 0` is structural: neither `forward` nor `pooled` reads
positions.

**Siblings, not this type.** `GlobalRescale` remains the frozen
`y = scale * x + shift` in `molpot.heads.rescale` (still `config["ftype"]`;
OMol RMS `1.433569` is O(1)). `Checkpoint` is not edited; persistent
buffers already round-trip. `MACEEncoder.atomic_energies` is not edited;
dtype promotion of `e0` inside `MacePotential.energy_core` is molzoo, out
of scope.

**Compile.** `forward` stays an index plus an optional multiply: no
`.item()`, no data-dependent shape, no Python branch on tensor values.
`validate` is the named host-sync seam.

### Reuse decision

- `generalize AtomicReferenceEnergy.__init__` — promote dtype to float64
  and add the explicit support mask / scheme / units / `_apply` invariant;
  the type already exists and is the MACE E0 owner.
- `reuse AtomicReferenceEnergy.forward` / `pooled` — same symbols; `pooled`
  keeps `EnergyAggregation`.
- `reuse EnergyAggregation` — the only graph sum; do not reimplement
  `index_add`.
- `reuse GlobalRescale` — frozen normalizer sibling; do not fold scale/shift
  into the reference table.
- `reuse MACEEncoder.atomic_energies` — construction call unchanged; do not
  edit `src/molzoo/mace/encoder.py`.
- `reuse Checkpoint` — `model.state_dict()` already contains persistent
  buffers; do not edit `src/molix/core/checkpoint`.
- no `new` symbol.

## Files to create or modify

- `src/molpot/heads/energy.py`
- `tests/test_molpot/test_heads/test_energy.py`

## Tasks

- [x] Write failing unit tests for AtomicReferenceEnergy dtype, support, mask, scheme, and state_dict (tests/test_molpot/test_heads/test_energy.py → TestAtomicReferenceEnergy) (rework: `test_lookup_buffer_honours_the_fp64_precision` currently pins `config["ftype"]`; `test_pooled_sums_e0_per_graph` currently infers support from a dense zero-filled table)
- [x] Embed hard-coded omol_elem_refs H/C/O/Fe/Kr literals and the 350-Kr float64 cancellation case in tests/test_molpot/test_heads/test_energy.py (public API; SHA-256 in a comment; no `regressions/` directory; no molhub/fairchem runtime)
- [x] Generalize AtomicReferenceEnergy.__init__ in src/molpot/heads/energy.py to float64 lookup, explicit supported buffer, scheme/units extra_state, and _apply that refuses to downcast c[Z]
- [x] Implement AtomicReferenceEnergy.validate, forward(mask=), and pooled(mask=) in src/molpot/heads/energy.py (EnergyAggregation sum; eager raise only on pooled/validate)
- [x] Add Google-style docstring on AtomicReferenceEnergy in src/molpot/heads/energy.py with units eV, the three schemes, support vs structural mask, compile/eager split, and the Domain-basis references
- [x] Verify against the 350-Kr cancellation oracle and the unsupported-Z raise
- [x] Run full check + test suite

## Testing strategy

Mirror: `src/molpot/heads/energy.py` →
`tests/test_molpot/test_heads/test_energy.py`, class
`TestAtomicReferenceEnergy`. One method per concern. Fast tier except none
of these tests compile. Unit green:
`python -m pytest tests/test_molpot/test_heads/test_energy.py -v`.
`regressions/` was deleted on 2026-09-07 (CLAUDE.md / notes); goldens are
literals in this class.

Happy path:

- Scatter constructor with `atomic_numbers=[1, 6, 8, 26, 36]` and the
  five `omol_elem_refs` literals, `scheme="combined_hof_linear"`:
  `forward` returns those five values as float64.
- `pooled` on `Z=[8, 1, 1]`, `batch=[0, 0, 0]`, `num_graphs=1` equals
  `-2079.7006341793223` eV.
- `list(module.parameters())` is empty; `state_dict` contains
  `atomic_energies` (float64) and `supported` (bool); `load_state_dict`
  restores scheme/units via extra_state.
- Default `scheme="isolated_atom"` so the MACEEncoder call still
  constructs.

Edge cases:

- Dense constructor without `atomic_numbers` and without `supported`
  raises.
- Supported Z whose `c[Z] == 0.0` does **not** raise.
- Unsupported in-range Z (e.g. N=7 on an H/C/O/Fe/Kr table) raises
  `ValueError` matching `[7]` from `validate` and from eager `pooled`.
- Out-of-range Z (e.g. 999) raises; it is not clipped to `max_z`.
- Ghost row `Z=1` with `mask=[True, True, True, False]` on
  `[8, 1, 1, 1]`: unmasked `pooled` is water + `c[H]`; masked `pooled`
  is water only. Masked unsupported Z does not raise.
- Built under default fp32 `config["ftype"]`, buffer is still float64;
  `module.float()` leaves `atomic_energies` as float64 (`_apply`).
- `forward` does not read positions: `e0.sum().backward()` does not
  populate a `pos.grad`.

Domain / cancellation (hard-coded; comment tool=`fairchem` YAML, commit
`41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`, key `omol_elem_refs`, date
2026-09-11, SHA-256 above):

- 350 atoms of Kr, `E_res = 0.35` eV (`1 meV/atom`).
  `E_DFT = e0_f64 + 0.35`. Float64 `E_DFT - e0_f64` matches `0.35` within
  `1e-12` eV. The same subtraction after a float32 round-trip of `e0`
  disagrees by more than `0.1` eV (binary32 ulp of the ~2.6e7 eV sum is
  ~2 eV).

Do not import molhub, fairchem, ASE, or subprocess an oracle.

## Out of scope

- BatchTask Residual, `graphs.e0` / `graphs.energy_residual` columns,
  collate / FrameStore precision policy, autocast rules.
- `molhub.cache.Cache`, source indexes, atom-count indexes.
- QR/SVD / `lstsq` fitter, training-scope selection, digest broadcast.
- `AtomicDress` / `AtomicDressing`.
- `NeighborList`, `PadMolecularBatch` implementation (only consumed as
  the `pad_atom_type=1` fact).
- Edits to `src/molzoo/mace/encoder.py`, `src/molzoo/mace/potential.py`,
  `src/molix/core/checkpoint/`, `src/molpot/heads/rescale.py`.
- Scripts under `scripts/mm_param_learning/` (they construct dense
  zero tables without a support mask; a later slug updates them).
- Loading official MACE checkpoints that lack the new `supported` key
  (upstream remap already rebuilds E0 at construction).
- Production training, OMol25 protocol hyperparameters (`cache-layers-02`).
