# Cache boundaries, reference state and training parity

Decision/review: 2026-09-11. Generalizes the OMol25/Class-I investigation;
the old proposal to unify atom counts and reference fits in a derived Store
is superseded. Implementation is specified by the `cache-layers` chain:
[01-sources](../specs/cache-layers-01-sources.md),
[02-cache](../specs/cache-layers-02-cache.md),
[03-refs](../specs/cache-layers-03-refs.md),
[04-dtype](../specs/cache-layers-04-dtype.md),
[05-residual](../specs/cache-layers-05-residual.md).
Geometry-memo and `omol25-fairchem-4m` protocol slices did not clear
architect design-mode after one rewrite and are not persisted.

## The separation

| Concern | Owner | Persistence |
|---|---|---|
| Source data, scientific units, splits and stable record identity | MolHub | Durable dataset |
| Rebuildable source indexes | MolHub `Cache` | Optional, bounded, immutable snapshots |
| Reference energies / normalizers | Training/checkpoint owner | Small required model state |
| Residual labels, force weights, padding | molix BatchTask / TensorFrame | In-flight columns only |
| Geometry binding, neighbor validity, compiled buffers | Runtime owner | Explicitly invalidated RAM/device state |

Name the disposable source-index abstraction `molhub.cache.Cache`. No legacy
API compatibility is required. Existing FrameStore is a different concept:
the canonical materialized-data format, not this cache's new name. Do not
call the injected snapshot object `store=`; the keyword is `cache=`.

Cache field names match the Frame. `n_atoms` is a graph-level quantity, so
the public read is `loaded.cache("graph.natoms")` (not `cache("atoms")`,
not `n_atoms`, not molnex `graphs.num_atoms`). Extraction is
`frame["atoms"].nrows`; do not write `frame.meta["n_atoms"]`.

Only molhub datasets and `molhub.loader.DataModule` call `cache`. molix
knows `DataModuleProtocol` and optional count tensors on the packer; it
does not import molhub or look up snapshots. ClassIFit / molzoo models do
not own a `datamodule()`.

A TensorFrame is a typed view, not evidence that its contents are unchanged.
Fresh training batches miss object-keyed memos; reused/static batches can
change contents without changing identity. Remove identity-based shortcuts,
but retain eager validation on every new binding. Do not misclassify a
cell-dependent inverse or a neighbor skin/displacement tracker as a dataset
cache or delete it without replacing its validity contract.

## Disk is exceptional

Reuse source counts. When absent, at most one count-index family is added,
partitioned by canonical source shard, not one array per experiment/subset.
An int32 count array costs about 16 MB at 4M rows and 400 MB at 100M. A
95-element float64 reference table is only 760 bytes before provenance.
Per-frame e0, residual energy, force weights, masks and neighbors are not
persisted. Full materialization/staging requires its own budget and is never
an AtomicDressing side effect.

Cache misses may increase work, not alter sample eligibility or labels.
Unknown counts are not measured bounds; exact filters need exact values.
Schedule/count snapshots are explicit at planning boundaries; checkpoint the
resume plan independently of optional Cache retention.

Immutable revision and ordered row identity are essential. A path and row
count cannot identify content; sorting subset indices loses row order. Use
existing manifests, not an all-data hash scan on each launch.

No shared per-row writes from training workers. Aggregate discoveries and
publish immutable, validated snapshots through a coordinator. Atomic
publication is not exactly-once computation; rename is not exclusive-winner
arbitration. Bounded duplicate preparation is preferable to a distributed
compute-once service for a tiny reference table.

## Reference energy is a scientific contract

Distinguish fixed physical references, empirical fitted offsets and the
per-graph sum. Load or fit on a declared training scope, validate, freeze,
checkpoint, apply once, and restore at inference. Unsupported elements do not
get silent zero references; padding slots do not imply element support. Use
direct float64 QR/SVD for bounded fits with coverage/rank diagnostics. No
implicit all-OMol fit and no worker-local fit.

Raw total energy and coefficients remain float64 through subtraction. Only
the residual follows model precision. A late `.double()` cannot repair an
earlier float32 cast. Column-aware precision applies to both sample and
FrameStore batch paths and to model-wide dtype/device moves.

Padding and label masks compose with task weights. Protect labels from
in-place prediction writes. Preserve standalone total-energy inference;
the cheap reference lookup may live in an inference adapter. Charge/spin
conditioning remains a model responsibility, not an elemental adjustment.

## OMol25 alignment is explicit, not a mixture of recipes

The adjacent alignment spec pins FAIR's public OMol-only 4M AllScAIP recipe
at a commit, including reference table, normalization, units, splits, losses
and optimizer/scheduler. It distinguishes this recipe from the original
OMol25 paper and UMA multi-dataset training. This is a chosen configuration
reference, not a claim that all published works share settings.

Old Class-I calibration (4m/val-neutral, custom kcal loss/outlier thresholds,
8192-sample fitted offsets) is not evidence for this new profile. Do not
inherit those defaults silently or approve production without new evidence.
Conservative versus direct forces and correctness fixes to upstream edge
behavior are named adaptations, not hidden differences.

## Known implementation debt covered by the specs

- Weak source IDs and row-order collisions; count hints entering exact filters.
- Stale DataMixture count copies and full-row Python lists during bounded fits.
- Reference coefficients/labels downcast before float64 subtraction.
- Missing-element zero fallback, clipped Z and normal-equation fitting.
- Batch-task masks omitting padding; derivative writes overwriting labels.
- Geometry memo deletion without replacement validation/binding duties.
- MolHub owns source/index integration, not a parallel molix persistence
  service. Stale MolHub writer imports must match the actual FrameStore API
  when that materialization path is exercised.

These are implementation obligations, not completed fixes. Unit, fault-
injection, distributed and bounded hardware evidence remain pending.
