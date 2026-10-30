# Rondo MACE conformer encoder

| Field | Value |
|---|---|
| Opened | 2026-10-28 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`), phase 4 with `rondo-enc-05-swap` |
| Status | implemented 2026-10-29 (unit tier green); the experiment is spec 05 |

## 1. Contract

`MaceConformerEncoder(spec, *, dtype)` in `src/molrep/encoders/mace_conformer/` (the
2026-10-28 layout ruling puts every encoder under `molrep/encoders/<arch>/`; the draft's
`molzoo/rondo/encoder.py` is superseded) returns `EncoderFeatures(atom, bond)` at ByteFF-26's
widths (atom 256, bond 256), so ByteFF-26's heads read it unchanged. `Rondo(ff, *, dtype,
encoder=...)` takes it in place of the covalent-graph encoder; it is registered as `graph`
either way, and the heads are sized from `encoder.width` / `encoder.bond_width`.

Primitives (`forward` composes them):

| method | does |
|---|---|
| `invariants(frame)` | every MACE layer's `0e` block, concatenated `(N, L·F)`, each conformer on its own geometry |
| `reference_atoms(frame)` (static) | each atom's twin in the first conformer of its profile |
| `readout(frame, invariants, pos)` | atom and bond features, class-averaged |
| `forward(frame)` | `readout(frame, invariants(frame)[twin], pos[twin])` |

## 2. Decisions

- **Backbone.** `molrep.encoders.mace.MACEEncoder` by composition (`self.mace`), its
  primitives only — never its `forward`, which rejects `graphs.total_charge` on an
  unconditioned spec and reads positions with a gradient. `MaceConformerEncoderSpec`
  subclasses `MACEEncoderSpec` and states MACE-OFF's body schedule (agnostic first layer,
  residual after, per-element products, no distance transform, no conditioning). Defaults:
  THEMol's ten elements, `r_max 5`, `l_max 2`, `max_hidden_l 1`, 128 features, 2 layers,
  correlation 3, radial MLP `(64, 64, 64)`. Trained from scratch; a MACE-OFF24 warm start is
  not in phase 4.
- **Positions, detached.** `pairs.pair_diff.detach()` — the padder's dead pairs are
  displaced beyond the cutoff there; rebuilt from `atoms.pos` they would be zero-length
  self-loops inside it. Bond lengths come from `atoms.pos.detach()`. No ∂parameter/∂R term
  enters any force: the energy is a Class-I energy at fixed parameters.
- **Pair list.** The batch's own list (full, both directions, `NeighborList`). Phase 2's
  field prices every intramolecular pair (`AMBER_CUTOFF` 50 Å), so the list is every pair
  of the molecule; the polynomial envelope zeroes each one beyond `r_max`. About half the
  edge work is on pairs beyond 5 Å — accepted for a quick experiment, measured in spec 05.
- **Reference conformer.** The first graph of each `graphs.profile` (molix `ProfileIndex`,
  batch-local in `[0, B)`; a THEMol row is one profile and its frames share atom order).
  Computed in the model from `atoms.batch` and `graphs.profile` with scatter-min, static
  shapes, no host sync, so no new batch task. An atom whose reference graph has another
  atom count (the padder's ghosts: the sink shares its profile with empty dead slots) is
  its own reference. The draft's `molix.data.ReferenceConformer` task is not needed.
  Every conformer still runs through MACE (static shapes); only the reference's
  invariants are read.
- **Atom readout.** `MLP[s_i ‖ emb(formal charge)]` (Linear–SiLU–Linear, hidden 256) → 256.
  The formal-charge embedding is an addition to the draft: geometry does not fix a charge
  state, and ByteFF-26's encoder reads it too.
- **Bond readout.** `½(MLP[s_i ‖ s_j ‖ φ(r)] + MLP[s_j ‖ s_i ‖ φ(r)]) + emb(order) +
  emb(in_ring)`, `φ` a 16-Gaussian basis of the reference bond length on 0.8–2.4 Å
  (finite at r = 0, the dead rows' length; a Bessel basis is not). The draft had no length
  term; r0 and k depend on it first.
- **Equivalence.** Atom and bond rows are averaged over `atoms.equi_index` and the bond
  classes `Equivalence.bonds` derives, as ByteFF-26 does: a conformer breaks topological
  symmetry, the contract restores it.

## 3. Open

- `avg_num_neighbors` defaults to 16 (a message normalisation, not a fitted quantity); the
  THEMol mean within 5 Å is measured in spec 05's calibration and replaces it if it is off
  by more than 2×.
- Compile: the joint bodies run under `fullgraph=True` (Trainer.compile); the dynamo
  explain of the encoder is a `slow` test written with spec 05's gate 2.

## 4. Tests (unit tier, `tests/test_molrep/test_encoders/test_mace_conformer/`)

Shapes; rigid-motion invariance; equivalent atoms and bonds equal; every conformer of a
profile reads the first one's geometry; two profiles read their own; a ghost whose
reference graph differs in size is its own reference; no gradient reaches `atoms.pos`;
bond readout symmetric and finite at zero length; formal charge clamped. Rondo:
an injected encoder feeds every head and trains on the energy; a covalent setting beside
an injected encoder is refused.
