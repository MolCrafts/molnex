# Rondo architecture review — 2026-10-12

Asked for by the operator after S1 small passed its gate with `hessian_eq` +
`skip_large_nb` ("抽身出来，系统评估网络架构是否有问题"). Two independent
inputs: a read of the model against ByteFF (`ml-expert` agent, one 10 s CPU
probe on the repo fixtures) and CPU diagnostics on real THEMol hessian val data
(`.jobs/arch_diagnostics.py`, jobs 3691055 on the eq best.pt and 3691174 on the
untrained dry-run checkpoint; JSON beside each run's `param_report.json`).

## Verdict

No architectural defect explains the S1 symptoms; each has another cause,
now measured:

| symptom | cause | evidence |
|---|---|---|
| partial-Hessian loss plateaus at 0.435 (untrained 0.49, 8-molecule overfit 0.44) | functional form: 97 % of the loss is the 1-3 blocks, which a harmonic angle (no Urey-Bradley / stretch-bend, torsions zero in S1) cannot represent | 1-2 residual 0.091 → 0.0385 trained; 1-3 residual 0.719 → 0.661; C-C 1-3 = 1.00 |
| "train 24 vs eval 37" | the training value is `log_knee(mean, 10)`: 24 ⇔ raw 40.6; eval is the raw mean 37.4 | `StationaryForceLoss(soft_max=10)` |
| sp3 θ0 walked 109° → 95° | the force at a minimum has no gradient along a common θ0 shift at a symmetric centre; fixed by `hessian_eq` | runs 3680577 / 3685025 / 3685026 / 3689502 |
| eval swings 39 → 59 at lr 1e-6 | a strained state (θ0 14° off) plus a very sharp r0 valley: `dF/draw_r0 ≈ 2K·0.1·r0 ≈ 135 kcal/mol/Å per unit raw`, so ≤ 1e-3 per weight over 600 steps moves `mean F²` by tens | agent probe B; the worst val molecules went 15-38 → 190-260 between best and last, alkyl-rich |

What is fine: reversal / outer-leg symmetry of every context; automorphism
equivariance to 2e-7 (ByteFF's `equi_features` averaging is redundant, A7
holds); phase/sign conventions (periodic, OPLS `K_n = 2k_n`, cvff three cyclic
rows) match ByteFF; hybridization references; C∞ geometry kernels; pre-norm hops
(feature RMS 1.0-1.3); zero-init heads with per-row references; the EEM solve
is well conditioned (η 1.6-5.0 on val atoms, floor 1e-3 never approached);
the 2-hop encoder's features separate the geometric classes the S1 data
distinguishes (bond-length variance explained 0.999, within-group 0.006 Å;
angle 0.991, within 1.15° vs 7.7° total — the same untrained, so this is the
input channels + topology, which is what the question was).

## Real architectural issues, ranked (they bite in S2 / S3, not S1)

**Status 2026-10-12 (evening):** 1 and 4 are fixed (spec A21 / A22 —
`molrep.chem.BondTable` + contexts reading the term's own bonds; `ExpAround`
windows). 2 is measured, not changed: `.jobs/head_grad_probe.py` logs
‖dL/draw‖ per head per objective and the trunk share per stream, at init and
after 30 steps; ByteFF's own gains have the same r0 : θ0 : torsion ratio
(15·std: 2.86 Å / 149° / 17.8 kcal/mol per unit), so a rebalance is not an
alignment and needs that measurement first. **D2 measured** (job 3723477,
small tier, cap 4096, 2 real nested batches, rms over live rows): at init
`hessian_geom` gives r0 1.1e-2 vs θ0 1.5e-3 (7×, as computed), `hessian_eq`
r0 8.5e-4 vs θ0 2.2e-4, `hessian_k` bond K 2.0e-5 ≈ angle K 2.2e-5 ≫ improper
k 1.7e-6, `hessian_eq_improper` 4.4e-5. After 30 Adam steps the trunk share
per objective (dL/d features at the stream norms): `hessian_geom` atom
1.1e-4 / bond 3.0e-4 / angle 6.4e-5, `hessian_eq` 2e-5, `mbis` 2e-6 (atom),
`hessian_k` ~1e-6, `hessian_eq_improper` ~6e-7 — the shared trunk is driven
~100× more by the geometry objective than by the stiffness one. In S1 this
has no visible cost (hessian_k sits at its form floor either way); a
per-head backward scale is the S2/S3 lever if torsion / LJ learning stalls.
3 stays as described (mbis in every stage is load-bearing). Every checkpoint
before the fix is invalid (the context MLPs changed width). **Night**: the
first long run on these fixes (3723816, 26 400 steps) failed only the
feature-RMS gate — the improper stream drifted to 5.9 under the
straight-through window of fix 4 (rows the strain objective wants at the
floor pull forever), so the windows now saturate (spec A24, ByteFF's tanh).
And the large tier's OOM at the tier capacity was the training step, not the
model: an eval loop holding the previous batch, eval graphs left on the
resident validation batches, and five encoder passes per hessian batch —
fixed in `molix.core.steps` (eval release, one `tables` per source), see the
checklist entry of 2026-10-12 night.

1. **Valence contexts never see the bonds of their own term**
   (`src/molrep/chem/context.py:104-130, 152-181, 209-235` read `h_atom` only;
   `encoder.py:284, 298, 312` never pass `h_bond`). ByteFF interleaves the
   term's edge features (`layers.py:486-510`, `OutputWithBond`). Probe: in a
   Kekulé C8H8 ring the torsion about a single bond and the one about a double
   bond get bit-identical proper features at 2, 4 and 8 hops while the two
   bond features differ by 2.6. Exact collisions need vertex-transitive graphs
   (rare); the soft version is universal — every torsion head must decode its
   central bond order from a 2-hop atom summary at width 32, and the central
   bond order is the main determinant of a torsion barrier. **Fix before S2**:
   per-term bond slot columns from the loader (`angles.bond_ij/bond_jk`,
   `propers.bond_ij/jk/kl`, `impropers.bond_ci/cj/ck`; `molix.data.Topology`
   already walks bonds) and contexts `MLP([h_i, h_ij, h_j, h_jk, h_k])` with
   the same reversal symmetry. ~40 lines in `context.py` + one loader task. The
   spec's §5 crosswalk says "valence symmetry matched" and does not record
   that the bond inputs were dropped — add an adaptation row with the risk.
2. **Head gains put the parameter families 100-1000× apart in gradient
   scale through one shared trunk** (`classical.py`: r0 gain 0.1 → 135
   kcal/mol/Å per unit raw at K 450; θ0 20°/unit → ~20; torsion `1/n`
   amplitude → ~0.7; σ 0.1 → ~0.6). Per-objective `trainable` detaches
   *tables*, not the trunk's share: in a stage where one objective trains
   several tables (S3 `relax`: every bonded table + σ/ε) the trunk follows r0,
   and torsion / LJ learn through their own last layer only. ByteFF has the
   same gains in principle but one family per stage. **Fix**: rebalance gains
   so a unit of raw moves the force comparably per family (e.g. r0 0.1 → 0.02,
   θ0 20° → 5°, k 1 → 0.3), or a per-head backward scale; verify by logging
   `‖dL/draw‖` per head per objective (D2). Hours.
3. **Charge model** (`charge/kernels/diagonal.py:74-91`, `chi_eta.py:43-47`):
   diagonal EEM is a per-atom readout `−χ/η` plus a *global* redistribution of
   the residual charge ∝ 1/η over the whole molecule (non-local, no inert
   atoms); every head starts at a reference except this one (q = 0); and
   `∂E_coul/∂q = 0` at q = 0, so an E/F objective can never move charges off
   zero — the curriculum's "mbis in every stage" is load-bearing and nothing
   says so. Measured on the hessian set: per-atom charges moved p50 0.08 e /
   max 0.29 e between best and last of the clip-10 run while the mbis val loss
   stayed 0.004 — the charges on hessian-set chemistry are underdetermined by
   the mbis set alone. **Options**: (a) a ByteFF-style BCC head on `h_bond` +
   formal charge (local, exactly conserving, non-zero at init; ~60 lines);
   (b) keep EEM with per-element χ/η reference tables so raw = 0 is classical
   EEM. Only (a) removes the non-locality. Also: a unit test that
   `grad(E_coul, chi)` is 0 at init, and a static check that a stage naming
   `atoms.charge` under an E/F loss follows an MBIS objective.
4. **Force constants unbounded above** (`ExpAround`, `classical.py:64-79`;
   ByteFF bounds everything with `tanh_output` / `param_range`). One outlier
   Hessian row can send a K to 1e4-1e5; the loss drops the row (`blowup=1e4`)
   but the head is free to go there. **Fix**: straight-through clamp in log
   space (`K ∈ [K_ref/8, 8 K_ref]`, `r0 ∈ [0.7, 1.4] r_ref`). Ten lines.

Capacity, not defects: small tier is 2 hops × 32 wide, no jumping knowledge,
32-wide heads; ByteFF is 4 EGT layers × 256, `jk='cat'`, 256-wide heads. Two
hops suffice for bond/angle stiffness (measured) and are marginal for torsions
(3 bonds from the central bond) and short for MBIS charges / LJ (substituent
effects at 3-4 bonds). Messages are undirected (`blocks.py:107-122`,
recoverable from the residual) and the improper context pools legs before its
MLP (`context.py:233-235`) — minor, both fall out of fix 1.

## Diagnostics that settle what is still open

- D1 (fix 1): on a THEMol shard count torsions whose `(h_i,h_j,h_k,h_l)` tuple
  maps to more than one central `bond_type`; train the proper head alone on the
  torsion-scan loss with vs without term bond features.
- D2 (fix 2): hook each head's last `Linear` and log `‖dL/draw‖_rms` per head
  per objective for 50 real steps.
- D3 (plateau): one free parameter per term on the 8-molecule set under
  `PartialHessianLoss` — the network-free floor by pair class.
- D5 (fix 3): regress `q_pred − q_MBIS` on `1/η_i`; MBIS std across conformers
  of one molecule = the floor of any geometry-free charge model; charge head
  alone with 2 vs 4 hops.
