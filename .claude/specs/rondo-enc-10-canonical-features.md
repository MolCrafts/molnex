# The canonical feature layer

| Field | Value |
|---|---|
| Opened | 2026-10-30 |
| Chain | rondo-enc (charter `rondo-enc-00-charter.md`, row "CanonicalAdapter") |
| Status | dropped 2026-10-30 (operator: "我们不做这个方向了"); kept as the record of the design and the decisions in §7 |
| Scope | operator 2026-10-30 ("设计 canonical feature 层") |

## 1. Why

Today the interface between encoders and heads is only a **type**: `EncoderFeatures(atom
(N, D), bond (N_bonds, D'))`. What the numbers *mean* is whatever ByteFF-26 member-01's EGT
happens to produce — an arbitrary learned basis, specific to that one member. Every other
encoder reaches the heads by imitating it (phase 4: MACE distilled to 0.078 relative MSE,
~28 % RMS off, and needed a physics fine-tune to land within 10 % of B). So "swap the
encoder" has meant "imitate EGT", and nothing in the tree says what a compliant encoder must
deliver, how to check it, or how a head trained on one encoder's features would fare on
another's.

`MaceConformerEncoder` is the symptom: it bundles a MACE body, a reference-conformer choice
(a data concern) and readouts onto EGT's space (an adapter) in one class with one name.

## 2. What the layer is

A **contract** on the encoder → heads seam, in three parts: structure, invariances,
semantics. An encoder is *compliant* when its adapter's output meets all three.

### 2.1 Structure (static, unit-testable)

`CanonicalFeatures(atom (N, 256), bond (N_bonds, 256))`, at the caller's dtype —
`EncoderFeatures` renamed for what it now promises, with the widths fixed by one
`CanonicalSpec(width=256, bond_width=256)` that Rondo validates at construction (an
encoder whose adapter emits other widths is refused by name). Bonds in `bonds.bond_index`
order, one row per bond for both directions.

### 2.2 Invariances (unit-testable, one contract test per encoder)

1. Permutation equivariance over atoms and bonds.
2. Topological equivalence: rows of atoms (bonds) in one `equi_index` class are equal.
3. **Topology-level**: the features are a function of the molecule, not of its conformer —
   static parameters are computed once per topology (charter, phase 6 `parameterize`). A
   geometric encoder satisfies this by reading one reference conformer per molecule (a
   frame task, §3), never by averaging over whatever conformers a batch happens to hold.
4. No gradient reaches `atoms.pos` (parameters are static; forces come from the classical
   terms only).
5. Normalised: every channel has zero mean and unit variance over the reference set
   (`CanonicalSpec.reference`: THEMol shard-000 train molecules), enforced by the adapter's
   final affine-free normalisation with frozen statistics stored as buffers.

### 2.3 Semantics (what the numbers mean) — the decision

The space must be defined by something other than one trained encoder. Proposal: the
canonical space is defined **functionally, by a frozen canonical decoder set**:

* **parameter decoders** — the published bonded heads and the pair heads Rondo prices with,
  frozen: features are canonical when these heads map them to correct parameters;
* **chemistry probes** — linear readouts, trained once and frozen, from canonical atom /
  bond rows to perception labels the data already carries (element, formal charge, ring
  membership and smallest ring, bond order, `equi_index`-free hybridisation proxy from
  degree × element, GAFF2 σ / ε where typed).

An adapter is trained to minimise the decoders' losses (parameters against the reference
model's on the same molecules + probe accuracy), never to copy another encoder's raw
vectors. The probes pin a coarse, interpretable skeleton of the space; the parameter
decoders pin everything the heads use. Two encoders that satisfy both are interchangeable
by construction, and the claim is measurable (§4).

Alternatives considered:

| anchor | cost | weakness |
|---|---|---|
| A. EGT member-01's raw basis (status quo, distillation) | none | arbitrary, member-specific; every encoder must imitate one network |
| B. frozen decoders + probes (proposed) | train the probes once; adapters train on decoder losses | probes cover chemistry only coarsely; the heads still carry ByteFF's choices |
| C. multi-encoder joint training (heads and adapters trained together, encoder sampled per step) | a full training campaign | needs pretraining-scale compute; the right end state if B's transfer fails |

B keeps every existing head, costs one probe fit, and moves the definition off any single
encoder. C is the fallback the measurement in §4 decides on.

## 3. Components and where they live

| piece | place | notes |
|---|---|---|
| `CanonicalFeatures`, `CanonicalSpec` | `molrep/encoders/features.py` | replaces `EncoderFeatures`; widths, reference-set id |
| `CanonicalNorm` (frozen-statistics normalisation) | `molrep/encoders/features.py` | the last op of every adapter; statistics fitted once on the reference set |
| per-encoder adapter | that encoder's `layers.py` | EGT: identity + `CanonicalNorm`; MACE: `AtomReadout` / `BondReadout` + `CanonicalNorm` (today's readouts, renamed for what they are) |
| reference conformer | `molix.data` task (`ReferenceConformer`, a `BatchTask`) | writes the profile's first frame's geometry beside each conformer; replaces the gather inside `MaceConformerEncoder` |
| `MaceConformerEncoder` | removed | becomes `MACEEncoder` + the adapter, fed by `ReferenceConformer` |
| chemistry probes | `molrep/heads/` (frozen, with their fit record) | used by training (adapter loss) and by the interchangeability report |
| contract tests | `tests/test_molrep/test_encoders/test_<arch>/` | one per encoder: 2.1 + 2.2 only — structure and invariances, never "interchangeable" |
| interchangeability measurement | lab (`rondo-enc/phaseN/`) | §4; an experiment, not a unit test |

Rondo validates `encoder.canonical == rondo.spec.canonical` at construction; the heads are
sized from `CanonicalSpec`, not from whichever encoder was passed.

## 4. How interchangeability is measured (lab, not tests)

On held-out molecules, for encoders E1, E2 behind their adapters:

1. **Decoder agreement** — parameters from the frozen heads on E1 vs E2 features, per table
   (median |Δ| / median |value|), and the phase-2 per-molecule scores for each.
2. **Head transfer** — a head trained *after* the fact on E1's canonical features (phase 2's
   D2 LJ head is the first one) applied unchanged to E2's: its gain over B should carry
   over. This is the operational test the current design has never passed or failed.
3. **Probe accuracy** per encoder, and linear CKA between E1 and E2 canonical rows.

Gate (proposal): head transfer keeps ≥ 90 % of the head's native gain, decoder agreement
within the phase-4 10 % bar. Failing it is the signal to go to anchor C.

## 5. Migration

1. Introduce `CanonicalSpec` / `CanonicalFeatures` / `CanonicalNorm`; fit the norm on EGT
   member-01 over the reference set; Rondo validates widths. Existing heads keep working
   (EGT adapter = identity + norm; the published heads see a whitened input, so they are
   loaded with the inverse affine folded into their first layer — exact, no retraining).
2. `ReferenceConformer` task; MACE adapter out of `MaceConformerEncoder`; class removed;
   M-DP's checkpoint remapped (keys only).
3. Fit the chemistry probes on EGT canonical features (frozen afterwards).
4. Re-fit the MACE adapter on decoder + probe losses (no distillation to raw EGT vectors);
   run §4.
5. Message-passing encoder: adapter + contract test; trained through the same losses.

Each step is its own commit with its gates; 1–2 are refactors with bit-for-bit parity
checks (folded affine), 3–5 are training under the production checklist.

## 6. Open

3. Widths: fixed at 256 / 256 (ByteFF's), or a different canonical width with every adapter
   (including EGT's) projecting into it? — under clarification.

## 7. Operator decisions (2026-10-30)

1. **Anchor B.** Frozen physical parameter heads and linear chemistry probes define the
   operational semantics; each adapter trains against these decoder objectives, never by
   matching EGT's hidden vectors. Requirements:
   - **D2's LJ head stays out of every adapter-training objective.** It is trained on EGT
     canonical features, frozen, and its **zero-shot transfer to MACE is the decisive
     compatibility test** (§4.2).
   - One shared, fixed normalisation reference; the structural and symmetry contracts
     (§2.1, §2.2) enforced.
   - MACE: **parameter consistency across different reference conformers is evaluated**
     (each conformer of a profile taken as the reference in turn, parameters compared);
     detaching coordinates alone does not show topology-level behaviour.
   - Migration keeps **numerical parity within tolerance**; bit-for-bit equality after
     folding the normalisation into linear layers is not required (fp reassociation).
   - Frozen decoders do not identify a unique latent representation, so interchangeability
     is **established empirically** by held-out head transfer, never assumed.
   - If the transfer gate fails, **first look for missing semantic constraints** (probes /
     decoders that do not pin what the transferred head reads) before moving to anchor C.
2. **Topology-level is a hard rule** (§2.2.3): no conformer-dependent canonical features;
   phase 5's geometry-dependent correction gets its own channel outside this layer.
