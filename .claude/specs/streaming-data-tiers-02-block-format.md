---
title: READY-tier block store format — streaming-appendable column blocks
status: approved
created: 2026-08-14
---

# READY-tier block store format — streaming-appendable column blocks

## Summary

The streaming-tier strategy (`molhub:.claude/specs/streaming-data-tiers-01-strategy.md`)
fixed the tier model and handed `-02` one deliverable: the on-disk format of a
TRAINING-READY block. This spec defines it. A **block** is one self-contained
directory holding 1–5 GB of unit-aligned training data as raw little-endian
column files plus a `manifest.json` and a small `commit.log`. A converter writes
it **incrementally, one sampling unit at a time** — the tail of each column file
is appended, fsynced, and then committed by a fixed-size crc32-checked log
record, so a crash resumes from the last committed unit by truncating
uncommitted tails; no predecessor bytes are ever rewritten. Storage is
**two-level ragged**: molecular topology (atomic numbers, bond/angle/proper/
improper columns, molecule-local atom indices) is stored **once per unit** while
positions / energies / forces are stored per frame, and every pointer vector is
**derived at open** from per-unit counts rather than stored, so a pointer can
never disagree with its payload. Reads are zero-copy `mmap` of whole column
files at offset 0 (`torch.UntypedStorage.from_file`), and the reader serves the
accessor surface the existing loader already consumes: `atom_counts`-style
pointer arithmetic for `TaskMixtureBatchSampler`, and a `frame_payload(frames)`
adapter that produces a `PackedCache`-v1-shaped payload so `PackedView` +
`collate_packed` build batches **unchanged** — no second collate path. `seal()`
finalizes the manifest with per-column sha256 plus a whole-block digest, which
is the only witness of READY (I5) and the artifact-role shape a molhub registry
manifest consumes. `PackedCache` v1 keeps its run-scoped charter untouched; this
spec adds no conveyor, no scheduler wiring, and no v1 migration.

## Domain basis

### Units and dtypes of stored physical quantities

The store performs **no unit conversion and no silent dtype narrowing** — a
dtype mismatch against the frozen schema raises. Units are declared once in
`manifest.provenance.units` and are the producer's responsibility (molhub owns
unit resolution via molpy `UnitSystem`; strategy § 9). MolCrafts-published
blocks use the repo convention:

| Quantity | Column | dtype | Unit |
|---|---|---|---|
| positions | `frame_pos` `(F_atoms, 3)` | `<f4` | Å |
| forces | `frame_atom_target_forces` `(F_atoms, 3)` | `<f4` | kcal mol⁻¹ Å⁻¹ |
| energy | `frame_target_energy` `(F,)` | `<f4` | kcal mol⁻¹ |
| atomic number | `atom_Z` `(A,)` | `<i2` | dimensionless |
| mass (optional) | `atom_mass` `(A,)` | `<f4` | amu |
| internal-coordinate force constants (optional) | `internal_target_hessian_diag` `(T,)` | `<f4` | kcal mol⁻¹ Å⁻² (bonds) / kcal mol⁻¹ rad⁻² (angles, torsions) — the per-row internal-coordinate kind decides, exactly as in `ProjectedInternalHessianLoss` (`src/molix/core/losses/molecular.py:280-282`) |

Big-endian hosts are rejected (`NotImplementedError`) rather than silently
byteswapped; dtype strings in the manifest are explicitly little-endian so a
column's bytes — and therefore its sha256 — are host-independent.

### Two-level row arithmetic (the format's only equations)

Let unit (molecule) `m` have `N_m = mol_n_atoms[m]` atoms and
`n_m = mol_n_frames[m]` frames. Three exclusive-cumsum pointers are **derived at
open**, never stored:

```
mol_atom_ptr[m]       = Σ_{m'<m} N_m'                  # molecule-atom space rows
mol_frame_ptr[m]      = Σ_{m'<m} n_m'                  # frame space rows
mol_frame_atom_ptr[m] = Σ_{m'<m} n_m' · N_m'           # frame-atom space rows
frame_mol             = repeat_interleave(arange(M), mol_n_frames)   # (F,)
```

For frame `f` with `m = frame_mol[f]` and local ordinal `k = f − mol_frame_ptr[m]`:

```
frame-atom rows of f    = mol_frame_atom_ptr[m] + k·N_m + [0, N_m)
topology atom rows of f = mol_atom_ptr[m] + [0, N_m)
term rows of family X   = X_ptr[m] + [0, X_n[m])       # X ∈ {bond, angle, proper, improper, internal}
```

`N_m` is constant within a unit **structurally** (a unit is one topology plus
its frames), which is what makes the `k·N_m` stride valid and removes any need
for a per-frame atom pointer column.

### Byte accounting and sizing (inherited anchors + derived consequences)

Per-frame payload for a 40-atom drug-like molecule, from strategy § Domain basis:
≈ 500 B/frame E-only (40·3·4 B + ~20 B), ≈ 1.0 kB/frame E+F. Consequences:

| Quantity | Arithmetic | Value |
|---|---|---|
| Frames in a 2 GB E-only block | 2·10⁹ / 500 | 4.0 × 10⁶ |
| Frames in a 2 GB E+F block | 2·10⁹ / 1000 | 2.0 × 10⁶ |
| Torsion role, 50–90 GB, 2 GB blocks | 50/2 … 90/2 | 25–45 blocks |
| Relax-diverse role, 100–200 GB | 100/2 … 200/2 | 50–100 blocks |
| Block index for one source | O(100–200) blocks × ~200 B | ≤ 40 kB — trivially hot (strategy § 5 R1) |
| Topology amortization | one 40-atom topology (bonds ≈ 40, angles ≈ 70, propers ≈ 100; 4 int32 cols) ≈ 2.8 kB | amortized over n_m frames; per-frame at n_m = 20 → 140 B, and **0 B** if it were re-stored per frame it would exceed the whole E-only frame budget by 5.6× |

**Why no edge / neighbor-list columns (decision, with arithmetic).** A 40-atom
molecule is smaller than a 5 Å cutoff sphere pairwise, so `G(r_c)` is nearly
complete: 40·39 = 1560 ordered pairs ≈ 1.2 × 10³ directed edges/frame. Storing
only `edge_index` (2 × int32) costs 9.6 kB/frame ≈ **10× the E+F payload**; with
`edge_diff` + `edge_dist` it is 28.8 kB/frame ≈ **29×**. It would also pin `r_c`
and `symmetry` into the artifact. Blocks therefore store geometry + topology +
targets only, and the neighbor graph is built downstream
(`molix.data.tasks.NeighborList` / MD bind path). This is the one place the
strategy's "make `c_prep` collapse to verify+mmap" goal is deliberately not
taken to the limit, and the factor 10–29 is the reason.

**Inode discipline.** Files per block = `manifest.json` + `commit.log` + one
`.bin` per column ≈ **14 (E-only, no valence) to 35 (E+F with four valence
families)**. At 200 blocks/source that is ≈ 2.8 k–7 k inodes, versus the
≈ 1 M files + 260 k directories `TensorDict.memmap_` produces for a 130 k-molecule
dataset (`src/molix/data/cache.py:26-43`), which is why v1 chose packed
single-file layout in the first place. One file per column is the smallest file
count that still permits offset-0 `mmap` per column (see below).

**Write amplification of the commit protocol.** A commit record is
`unit_index (i8) + n_columns (i4) + n_columns × committed_len (i8) + crc32 (u4)`
= 16 + 8·C bytes; at C = 16 columns, 144 B. A unit of 20 E+F frames carries
≈ 20 kB of payload, so commit overhead ≤ 0.8 % (≤ 1.5 % for E-only units of
10 frames). `manifest.json` (O(columns) ≈ 4 kB) is rewritten only at block
creation, when a new task label first appears (≤ 64 times, dictionary is
bounded), and at `seal()` — **not** per unit, which is what keeps this off the
200 k-commit path. Sealing streams the block once for sha256; at an assumed
1 GB/s single-core sha256 a 2 GB block costs ≈ 2 s (assumption, measured by the
benchmark task, not asserted here).

### Convergent-practice basis (survey 2026-08-14)

Concatenated per-field arrays + cumsum/offset index is the convergent layout for
sharded ML data stores, and this format is a deliberate instance of it, not an
invention:

- PyG `InMemoryDataset.collate` — concatenated per-key tensors + a `slices`
  dict; Fey & Lenssen, *Fast Graph Representation Learning with PyTorch
  Geometric*, arXiv:1903.02428 — https://arxiv.org/abs/1903.02428
- Apache Arrow columnar `List` layout — offsets buffer + values buffer, the
  same ragged encoding as `X_ptr` + concatenated columns —
  https://arrow.apache.org/docs/format/Columnar.html
- MosaicML Streaming (MDS) shard layout — u32 sample count + (n+1) u32 offsets +
  raw bytes per shard, O(1) random access, many medium shards —
  https://github.com/mosaicml/streaming
- Zarr v3 `sharding_indexed` — shard = concatenated inner chunks + an
  (offset, nbytes) index, motivated explicitly by inode reduction; ZEP0002 —
  https://zarr.dev/zeps/accepted/ZEP0002.html
- safetensors — u64 header size + JSON header (dtype/shape/data_offsets) +
  contiguous buffer, no holes; append is not supported, which is precisely why
  this format keeps its metadata in a **separate** file instead of a prefix
  header — https://github.com/huggingface/safetensors
- npy-append-array — grows axis 0 by rewriting a padded 64-byte-aligned header
  and recovers interrupted files by inferring shape from file size; the
  precedent for "commit record + truncate-on-resume" —
  https://github.com/xor2k/npy-append-array
- Sharded streaming I/O for large-scale training: Aizman, Maltby, Breuel,
  *High Performance I/O For Large Scale Deep Learning*, arXiv:2001.01858 —
  https://arxiv.org/abs/2001.01858
- Independent-shard concatenated-array stores validated for atomistic ML
  (3–5× over HDF5 on NFS): Atompack, arXiv:2606.29975 —
  https://arxiv.org/abs/2606.29975
- Sampling discipline the format must preserve (molecule-level grouping before
  configuration sampling): Wang, Fass, Chodera, Chem. Sci. **13**, 12016 (2022),
  DOI:10.1039/D2SC02739A — https://doi.org/10.1039/D2SC02739A

Two survey findings are recorded as **motivation for measurement, not as
asserted numbers** (identifiers not pinned at survey time, so they are not
citable evidence): Lustre client `mmap` page-fault readahead was historically
poor and a 2021 `llite` patch reported ≈ 9.75× improvement on sequential mmap
reads — hence Task 8 benchmarks `mmap` vs `pread` on the actual client before
the hot read path is frozen; and LMDB-style B+tree stores were reported
2.8–3.8× slower than packed formats under shuffled access, which is why a
key-value store is not a candidate here.

## Design

### 0. Placement and the migration dependency

Target module: **`src/molix/store/block.py`**, with exports in
`src/molix/store/__init__.py`. This assumes the parallel **molix-data-dissolution**
spec (drafting concurrently) has moved `cache.py` + `dataset.py` into
`src/molix/store/`. If that migration has **not** landed when this spec is
implemented, every path in this spec adjusts **mechanically**:
`src/molix/store/block.py` → `src/molix/data/block.py`,
`tests/test_molix/test_store/test_block.py` → `tests/test_molix/test_data/test_block.py`.
No other design element depends on the migration; the module imports nothing
from `cache.py` or `dataset.py`.

Vocabulary guard: "block" here is a **storage block** of the streaming tier. It
is not `molpy.Block` (a chemical block) and not a molrep `nn` block. No bare
`Block` symbol is introduced, so no import can collide.

### 1. On-disk layout

```
<block-root>/
  manifest.json          # schema, split, provenance, task dictionary, digests, sealed
  commit.log             # fixed-size crc32 commit records (append-only)
  mol_n_atoms.bin        # one raw little-endian column per declared column
  mol_n_frames.bin
  mol_task.bin
  unit_key_len.bin
  unit_key_bytes.bin
  atom_Z.bin
  bond_n.bin  bond_atomi.bin  bond_atomj.bin  bond_type.bin
  angle_n.bin angle_atomi.bin angle_atomj.bin angle_atomk.bin angle_type.bin
  ...
  frame_pos.bin
  frame_target_energy.bin
  frame_atom_target_forces.bin
```

Column files are **raw payload only** — no header, no padding, no holes.
Rows are appended along dim 0; inner shape (e.g. `(3,)` for `frame_pos`) is
declared in the manifest and never varies. A COO `(2, N)` layout is
**forbidden on disk** because growing its last axis is not an append; bonds are
therefore two 1-D columns (`bond_atomi` / `bond_atomj`) and `bond_index` is
stacked only at payload-assembly time. This also matches the repo's existing
valence column convention (`angles.atomi/atomj/atomk`, `src/molix/data/collate.py:88-93`).

**Column spaces** (fixed taxonomy of six; every column declares one):

| space | rows | examples |
|---|---|---|
| `molecule` | units `M` | `mol_n_atoms`, `mol_n_frames`, `mol_task`, `unit_key_len`, `mol_target_*` |
| `molecule_atom` | `A = Σ N_m` | `atom_Z`, `atom_mass`, `atom_type` |
| `molecule_term:<family>` | `Σ X_n[m]` | `bond_*`, `angle_*`, `proper_*`, `improper_*`, `internal_*` |
| `frame` | `F = Σ n_m` | `frame_target_energy`, `frame_target_*` |
| `frame_atom` | `Σ n_m·N_m` | `frame_pos`, `frame_atom_target_forces` |
| `bytes` | `Σ unit_key_len[m]` | `unit_key_bytes` |

**Naming determines the payload route**, table-driven in the module the way
`INDEX_KEYS` (`src/molix/data/collate.py:77`) is declarative rather than
hard-coded per call site: `atom_Z → atoms.Z`, `frame_pos → atoms.pos`,
`frame_target_<n> → graphs.targets.<n>`, `frame_atom_target_<n> → atoms.targets.<n>`,
`mol_target_<n> → graphs.targets.<n>`, `bond_atomi/atomj → bonds.bond_index`,
`bond_type → bonds.bond_types`, `<family>_atom{i,j,k,l}` / `<family>_type` →
that family's bucket columns. The writer **validates at schema-freeze time that
every column name matches a route rule** and raises otherwise — an unroutable
column cannot enter a block, so `frame_payload` never silently drops data.

**Required columns**: `mol_n_atoms`, `mol_n_frames`, `mol_task`, `unit_key_len`,
`unit_key_bytes`, `atom_Z`, `frame_pos`. Everything else is optional and
declared by the first appended unit.

**Derived, never stored**: `mol_atom_ptr`, `mol_frame_ptr`, `mol_frame_atom_ptr`,
every `X_ptr`, `frame_mol`, and `unit_key_ptr` — all computed by cumsum /
`repeat_interleave` at open (O(M + F) ints; ≈ 20 MB for a 4 M-frame block, ms of
work). One source of truth per fact eliminates the pointer/payload-disagreement
class of corruption entirely.

**64-byte alignment is moot here — recorded so nobody adds a header later.**
The npy precedent needs 64-byte header padding because payload follows metadata
*in the same file*. Here every column is its own file mapped at **offset 0**, so
each column's row 0 sits on a page boundary by construction; intra-column rows
(e.g. 12-byte `frame_pos` rows) need no alignment because `mmap` correctness and
zero-copy depend only on the mapping base pointer. No padding, no header, no
alignment field.

**`manifest.json`** fields: `format: "molix.block/1"`, `schema_version: 1`,
`columns: {name: {dtype, inner_shape, space, route}}`, `split` (one value per
block — see § 4), `task_names` (bounded dictionary, ≤ 64), `units`, per-space
`rows`, `provenance` (`{source: {...}, generator: {tool, version, commit},
created_at, units: {...}}` — opaque, caller-supplied, validated only for key
presence so no molhub coordinate is ever hard-coded in molnex; strategy § 9),
`digests: {column: sha256}`, `block_digest`, `sealed: bool`. `block_digest` =
sha256 over the **canonical JSON** (sorted keys, `separators=(",",":")`, UTF-8,
no whitespace) of `{columns, rows, split, units, digests}` — canonicalization is
specified because the regression pins these digests as literals.

### 2. Append protocol and crash recovery

Per `append_unit(unit)`:

1. Append each column's rows with `O_APPEND` writes (no seeks, no rewrites of
   any predecessor byte).
2. `fsync` every touched column file.
3. Append one fixed-size commit record `(unit_index i8, n_columns i4,
   committed_len[i8] × C, crc32 u4)` to `commit.log`; `fsync` the log.

The log record is the **single commit point** — the ordering above means a
record can only exist if the bytes it counts are already durable. Recovery
(`BlockWriter.resume(root)`): read the frozen schema from `manifest.json`, scan
`commit.log` backwards for the last record with a valid crc32 and a length that
is a whole multiple of the record size, `os.truncate` every column to its
recorded length, truncate the log itself past that record, and continue at
`unit_index + 1`. This makes every stage idempotent and the true state
re-derivable from *files present ∧ log* (I5), and it is the npy-append-array
recovery idea with an explicit record instead of an inferred file size. A crash
before the first commit leaves no manifest → start fresh.

`manifest.json` is written atomically (`tmp` → `fsync` → `os.replace`, the same
POSIX-atomic discipline `PackedCache.save` already uses,
`src/molix/data/cache.py:220-229`) at: block creation (schema freeze from the
first unit), first appearance of a new task label, and `seal()`.

`seal()` streams every column in 1 MiB chunks to compute per-column sha256,
writes `digests` + `block_digest` + `sealed: true`, sets column files to `0o444`
(advisory; the binding immutability witness is the manifest flag + digests), and
makes every later `append_unit` raise. **A sealed, digest-verifying manifest is
the only witness of READY** — no `_READY` sentinel file, no half-ready state a
consumer can read (I5), and each column file + its sha256 is exactly one
artifact role for a molhub registry manifest.

### 3. Read path

`BlockReader(root)` **refuses an unsealed block** (`ValueError`) — a consumer
structurally cannot read a partially written block.

Column mapping primitive (one seam, `BlockReader._map_column`):
`torch.UntypedStorage.from_file(path, shared=False, nbytes=...)` (MAP_PRIVATE on
a read-only file, offset 0 — which is why per-column files were chosen over one
packed file with per-column offsets: `torch.from_file` /
`UntypedStorage.from_file` have **no byte-offset parameter**), then
`torch.empty(0, dtype=dt).set_(storage, 0, shape, stride)`. Zero-copy is
*asserted*, not assumed: `tensor.untyped_storage().filename` must equal the
column path. Documented contingency: if the installed torch cannot map a
`0o444` file this way, the RED test in Task 5 fails immediately and the fallback
is `torch.from_numpy(np.memmap(path, mode="r", dtype=..., shape=...))` (numpy
takes `offset=` natively and `torch.from_numpy` on a memmap is zero-copy) behind
the same single seam — a one-line change, not a redesign.

**Layering guard**: `block.py` imports only stdlib + `torch` (+ `numpy` if the
fallback fires). It must **not** import `tensordict` or any collate symbol —
mirroring the guard that keeps `cache.py` IO-only
(`dynamic-batching-packed-collate-02-collate`, ac-005). Batch construction stays
the caller's job.

Accessor surface (mirrors what the loader already consumes, so no parallel
hierarchy is spawned):

| Member | Returns |
|---|---|
| `column(name)` | mmap'd tensor for one declared column |
| `unit_count` / `frame_count` | `int` |
| `unit_frames(unit)` | `(n_m,)` long — the frame rows of one unit |
| `unit_key(i)` / `unit_index(key)` | `str` / `int` (lazy reverse map) |
| `atom_counts` | `(F,)` long — `mol_n_atoms[frame_mol]`, pointer arithmetic only, mirroring `MmapDataset.atom_counts` (`src/molix/data/dataset.py:313-326`) |
| `edge_counts` | raises `ValueError` naming the deliberate absence of edge columns and pointing at `NeighborList` (see § Domain basis) |
| `max_atoms` | `int` |
| `task_codes` / `task_names` | `(F,)` int32 dictionary codes / the bounded label tuple |
| `group_codes` | `(F,)` int32 — **the owning unit row is the group id** (one unit ≙ one group, strategy § 2), so no string column and no per-frame group table exists |
| `verify()` | recompute digests; raise naming the first mismatching column |
| `frame_payload(frames)` | v1-shaped packed payload for exactly those frames, in that order |

`task_codes` / `group_codes` being int32 columns rather than parent-space Python
sequences is the representation strategy § 5 R1 asked for; decoding to the
`list[str]` `TaskMixtureBatchSampler` accepts is a caller one-liner over
`task_names`, and the sampler's own budget path needs only `atom_counts`, so a
plan can be built off a `BlockReader` **without unpacking a single frame**.

**`frame_payload(frames)` is an adapter to an existing contract, not a new blob.**
It assembles the dense `PackedCache`-v1 payload subset that
`collate_packed` consumes (`atoms` / `graphs` / `bonds` / `angles` / `propers` /
`impropers` buckets, `atom_ptr` / `bond_ptr` / `angle_ptr` / `proper_ptr` /
`improper_ptr` recomputed *locally over the selected frames*, `n_samples`),
expanding molecule-level topology to the selected frames only (batch-sized, the
same copy the collate would have made) and keeping topology atom indices
**molecule-local `[0, N_m)`** exactly as v1 samples do, so `collate_packed`'s own
`rebase` performs the global offsetting unchanged. Batch construction is then
the caller's composition — `collate_packed(PackedView(reader.frame_payload(f)), range(len(f)))`
— and its output must equal `collate_molecules` leaf-for-leaf, which is the
established equivalence oracle. Blocks carry no edges, so the payload exercises
`collate_packed`'s empty-edge fallback path (`src/molix/data/collate.py:527-533`).
`PackedCache.unpack_sample` compatibility is explicitly not claimed.

Path-independence (I1): topology indices are unit-local, all pointers are
derived, provenance is embedded, and nothing references another block — so a
sealed block directory can be copied or moved and still reads and verifies.

### 4. Unit alignment and split purity (I2 / I7, structural)

- The **only** ingestion method is `append_unit(unit: SamplingUnit)`. There is no
  `append_frame`, so a unit cannot straddle a block boundary — unit integrity is
  structural, exactly as `TokenBudgetPacker` makes it structural inside a batch
  (`src/molix/data/sampler.py:571-596`).
- A `SamplingUnit` is **one topology plus ≥ 1 frames of it**. `append_unit`
  validates `frame`-space rows == `n_frames`, `frame_atom`-space rows ==
  `n_frames · N`, and every topology atom index in `[0, N)`; otherwise it raises
  naming the offending column.
- Block-full is the **caller's** decision: it reads `writer.committed_bytes`
  against its 1–5 GB target and calls `seal()` before opening the next block. The
  store ships no size policy and no `run_everything` driver.
- **Split purity**: `split` is a constructor argument recorded once in the
  manifest. There is no per-frame or per-unit split field anywhere in the format
  and `append_unit` takes no `split` parameter, so a block cannot mix splits.
  Splits themselves are produced upstream (`class1-universal-splits@1`) and are
  never re-derived here.

### 5. Types (primitive, single-responsibility; construct → configure → one concern → read)

| Type | Responsibility | Members |
|---|---|---|
| `SamplingUnit` | frozen value type: one appendable unit | `key: str`, `task: str`, `topology: Mapping[str, Tensor]`, `frames: Mapping[str, Tensor]`; derived `n_atoms`, `n_frames` |
| `BlockManifest` | the JSON contract | `read(root)` (alternate constructor), `write_atomic(root)`, `column(name)`, `sealed`, `schema_version` |
| `BlockWriter` | append + commit + seal | `__init__(root, *, split, provenance)`, `resume(root)` (alternate constructor: recovery semantics), `append_unit(unit) -> int`, `committed_bytes`, `unit_count`, `seal()` |
| `BlockReader` | read accessors + views | § 3 table |

No `make_*` / `create_*` factory, no ambient context blob, no façade that hides
write→seal→read. Composition lives in `regressions/`, `scripts/`, and (later)
`-03` / `-04`.

### 6. Measured decisions (not assumptions)

Per the Lustre survey caveat, Task 8 measures on the target client before the hot
path is frozen: (a) column read throughput, `mmap` gather vs `os.pread` of the
same byte ranges, sequential and shuffled-frame access; (b) per-unit
append + commit + fsync cost (units/s, MB/s). The numbers, client, date, and the
verdict are recorded in the module docstring and `.claude/notes/notes.md`. If
`pread` wins by more than 1.5×, the docstring records the follow-up (a
`pread`-based column reader behind the same `_map_column` seam) rather than
leaving the gap silent.

### Reuse decision

No `librarian_report` was supplied with this task and the strategy spec already
flagged that a blueprint refresh before `-02` is advisable; the verdicts below
therefore come from direct inspection of every named surface in the current tree.

- `collate_packed` (`src/molix/data/collate.py:422`) — **reuse verbatim.** The
  reader adapts to its payload contract; no block-side collate is written.
- `PackedView` (`src/molix/data/dataset.py:48`) — **reuse verbatim.** It accepts
  any `Mapping` payload, so `PackedView(reader.frame_payload(frames))` needs no
  change.
- `rebase` + `INDEX_KEYS` (`src/molix/data/collate.py:77`, `:96`) — **reuse** for
  index rebasing (via `collate_packed`) and as the **pattern** for the
  declarative column→payload route table.
- `TaskMixtureBatchSampler` (`src/molix/data/sampler.py:267`) and its
  `atom_counts` contract (`_frame_budgets`, `:532-547`) — **reuse.** The reader
  serves those attributes; the sampler is not touched.
- `MmapDataset` / `PackedView` accessor set (`src/molix/data/dataset.py:313-355`)
  — **pattern**, deliberately not `generalize` *here*: `BlockReader` mirrors
  `atom_counts` / `max_atoms` naming and pointer-derived semantics, but wiring a
  block-backed `BaseDataset` / `DataModule` seam is `-04`'s owned surface
  (strategy § 12), so widening the dataset hierarchy now would pre-empt it.
- `PackedCache` v1 (`src/molix/data/cache.py:91`) — **new** artifact, per its own
  charter (`cache.py:1-49`: run-scoped, unversioned, unvalidated, single-file).
  v1 is not widened, not migrated, and its `FORMAT_VERSION` is untouched. What is
  reused from it is the *discipline*: atomic `os.replace` commit
  (`cache.py:220-229`), concatenated columns + cumsum pointers, and the
  inode-budget rationale.
- `PackedCache.save`'s schema inference across samples (`cache.py:298-355`) —
  **pattern**: schema is frozen from the first unit and later units are validated
  against it, with an actionable `ValueError` naming the offending column.
- molpy / molhub boundaries — **reuse** unchanged: units, provenance semantics,
  splits, and remote coordinates stay producer-side; `provenance` is opaque JSON
  here (strategy § 9), and no dataset coordinate enters a molnex API.

## Files to create or modify

- `src/molix/store/block.py` (new)
- `src/molix/store/__init__.py` (new — created by the molix-data-dissolution spec
  if that lands first; this spec then only adds the four exports)
- `tests/test_molix/test_store/test_block.py` (new)
- `tests/test_molix/test_store/__init__.py` (new)
- `scripts/store/bench_block_read.py` (new)
- `regressions/streaming-data-tiers-02-block-format.py` (new)
- `.claude/notes/notes.md`

## Tasks

- [ ] Write failing unit tests for `SamplingUnit` and `BlockManifest` (`tests/test_molix/test_store/test_block.py` + package `tests/test_molix/test_store/__init__.py` → `TestSamplingUnit`, `TestBlockManifest`): unit shape validation (`frame` rows == `n_frames`, `frame_atom` rows == `n_frames·N`, topology atom indices in `[0, N)`), unroutable column name rejection, manifest JSON round-trip, canonical-JSON `block_digest` determinism, sealed/unsealed discrimination, `schema_version` mismatch rejection
- [ ] Implement `SamplingUnit` and `BlockManifest` in `src/molix/store/block.py` (frozen value types, little-endian dtype strings with big-endian-host `NotImplementedError`, declarative column space/route table mirroring `collate.py:77` `INDEX_KEYS`, atomic manifest write via tmp → fsync → `os.replace`)
- [ ] Write failing unit tests for `BlockWriter` (`TestBlockWriter`): schema freeze from the first unit, per-unit commit record + `committed_bytes`, `BlockWriter.resume` truncating an uncommitted tail so the sealed result is byte-identical to an uninterrupted write, torn/crc-invalid commit record ignored, split purity (no `split` parameter on `append_unit`, no per-frame split column), `append_unit` after `seal()` rejected, dtype / missing-required-column `ValueError`
- [ ] Implement `BlockWriter.append_unit` / `resume` / `committed_bytes` / `seal` in `src/molix/store/block.py` (`O_APPEND` column tails → fsync → fixed-size crc32 commit record → fsync; manifest rewritten only at creation / new task label / seal; `seal` streams per-column sha256 + `block_digest`, sets `sealed` and `0o444`)
- [ ] Write failing unit tests for `BlockReader` (`TestBlockReader`): unsealed-block refusal, `verify()` naming the first mismatching column after a single flipped byte, file-backed zero-copy mapping (`untyped_storage().filename` == column path), derived pointers and the two-level `mol_frame_atom_ptr[m] + k·N_m` arithmetic, `unit_frames` / `unit_key` / `unit_index`, `atom_counts`, `task_codes` + `task_names`, `group_codes` == owning unit row, `edge_counts` actionable `ValueError`, and the `collate_packed(PackedView(frame_payload(frames)), …)` parity oracle against a `PackedCache`-v1 payload over identical samples (bonds + angles + propers included)
- [ ] Implement `BlockReader` in `src/molix/store/block.py` (offset-0 `torch.UntypedStorage.from_file` column mapping behind a single `_map_column` seam, pointers derived at open, § 3 accessors, `frame_payload` v1-payload adapter with unit-local topology indices) and export `BlockManifest` / `BlockReader` / `BlockWriter` / `SamplingUnit` from `src/molix/store/__init__.py` with alphabetized `__all__`
- [ ] Add Google-style docstrings with units and tensor shapes for the four public types and every public method, and record in the `block.py` module docstring: the append/commit protocol, the derived-not-stored pointer rule, the 64-byte-alignment-moot rationale, the edge-column exclusion arithmetic, and the `molpy.Block` naming disambiguation
- [ ] Measure block reads (`mmap` column gather vs `os.pread`, sequential and shuffled) and per-unit append + commit + fsync cost on the target Lustre client with `scripts/store/bench_block_read.py`, and record the numbers, client, date and verdict in the `block.py` module docstring and `.claude/notes/notes.md`
- [ ] Add regression example `regressions/streaming-data-tiers-02-block-format.py` (public API only: write two unit-aligned blocks, torn-tail resume, mmap read-back, relocation, collate parity; hard-coded sha256 goldens; no third-party runtime)
- [ ] Run full check + test suite (`$META.build.check`, `python -m pytest tests/ -q`, `python scripts/check_test_mirror.py`)

## Testing strategy

Unit tests live in `tests/test_molix/test_store/test_block.py` (mirroring
`src/molix/store/block.py`; adjust to `tests/test_molix/test_data/test_block.py`
if the dissolution migration has not landed), with test classes mirroring types:
`TestSamplingUnit`, `TestBlockManifest`, `TestBlockWriter`, `TestBlockReader`.
Each test targets a single method. No e2e under `tests/`.

- **Happy path** — write 3 units (2 molecules × multi-frame + 1 five-frame scan,
  with bonds and angles), seal, read back: values per column equal the inputs
  bit-for-bit (`torch.equal`), `unit_frames` returns exactly the appended frames,
  `atom_counts` equals the hand-written per-frame atom counts, `task_codes`
  decode via `task_names` to the appended labels, `group_codes` equals the owning
  unit row.
- **Append / crash-resume** — after a clean unit commit, append junk bytes to one
  column file (a torn tail) and, separately, append a truncated commit record;
  `BlockWriter.resume` must truncate to the last valid record in both cases, and
  the block sealed after resuming must be **byte-identical** (per-column sha256)
  to a block written without interruption.
- **Edge cases** — single-unit block; single-frame unit; a unit with no valence
  columns; a second unit missing a column declared by the first (`ValueError`); a
  dtype mismatch (`ValueError`); an unroutable column name (`ValueError`);
  `append_unit` after `seal()` (`ValueError`); `BlockReader` on an unsealed block
  (`ValueError`); `edge_counts` (`ValueError` naming the deliberate absence);
  `verify()` after flipping one byte.
- **Zero-copy proof** — `reader.column("frame_pos").untyped_storage().filename`
  equals the column file path (deterministic, not an RSS heuristic), plus values
  equal to the written tensor.
- **Equivalence oracle (parity)** — build a `PackedCache` v1 from the same
  per-sample dicts, collate both through `collate_packed`, and assert identical
  key sets, `torch.equal` per leaf, identical dtypes, and identical `batch_size`
  on `atoms` / `graphs` / `bonds` / `angles` / `propers` and top level. This is
  the same oracle discipline as `dynamic-batching-packed-collate-02-collate`.
- **Sampler metadata** — `TaskMixtureBatchSampler` constructed against a
  `BlockReader` (`atom_counts` + decoded `task_names` labels + `group_codes` as
  group ids, `max_atoms` only) builds a plan; a monkeypatched counter proves no
  column beyond the count vectors is read.
- **Domain validation with hard-coded values** — the two-level row arithmetic is
  checked against hand-computed literals for a fixture with `N = (3, 4)`,
  `n_frames = (2, 3)`: `mol_atom_ptr = (0, 3, 7)`, `mol_frame_ptr = (0, 2, 5)`,
  `mol_frame_atom_ptr = (0, 6, 18)`, and frame 3 (unit 1, `k = 1`) occupying
  frame-atom rows `6 + 1·4 = 10 … 14`. Fixture floats are exactly representable
  in float32 (multiples of 0.5) so digests and equality are host-independent and
  **no FP tolerance is involved anywhere in this spec**.
- **Layering guard** — a source assertion that `block.py` contains no
  `tensordict` and no `molix.data.collate` / `collate_packed` import.

**Regression example** (owed to this spec by strategy § Testing strategy):
`regressions/streaming-data-tiers-02-block-format.py` — public API only. It
writes two unit-aligned blocks into a `TemporaryDirectory` (block A: 2 units with
bonds + angles; block B: one 5-frame scan), performs one torn-tail
`BlockWriter.resume` leg, seals both, copies block A to a second path
(relocation), mmap-reads both back, and asserts: unit integrity
(`unit_frames` / `group_codes` partition the frames exactly, every unit whole in
one block), per-column sha256 and `block_digest` equal to **hard-coded literal
goldens**, `verify()` clean on the relocated copy, and `collate_packed` output
equal to the v1 oracle. Provenance comment records tool, version, command, and
date; no third-party oracle is imported or subprocessed at any time.

## Out of scope

- **The conveyor** (`-03`, molhub): prefetch/prepare stages, the block ledger,
  byte-bounded queues, leases, refcounted deletion, digest-at-the-boundary
  fetching (I3/I4/I6/I9/I10). This spec only makes a block's READY witness
  machine-checkable.
- **Scheduler / loader integration** (`-04`): a block-backed `BaseDataset`,
  `DataModule` / `batch_sampler_source` wiring, plan↔block mapping, R1 crossover
  measurement. `BlockReader` exposes the accessors `-04` will consume; it
  registers no dataset type.
- **`PackedCache` v1 changes or migration** — v1's charter, `FORMAT_VERSION`, and
  every existing cache stay untouched; there is no v1→block converter here.
- **Edge / neighbor-list columns** — excluded by the 10–29× arithmetic in
  § Domain basis; a future `schema_version` may add a `frame_edge` space.
- **Compression / codec choice** (zstd level, chunk size) and any on-the-wire
  format — a separate measurable question; blocks are stored uncompressed so
  `mmap` stays zero-copy.
- **Remote coordinates, dataset adapters, unit conversion, split production** —
  producer-side (molhub / molpy); `provenance` is opaque here.
- **Multi-node coordination**, lease files, and stale-lease expiry (`-03`).
- **Docs pages** — `docs/api/molix.md` and the `molix.store` package narrative are
  owned by the molix-data-dissolution spec; this spec documents via module and
  API docstrings only.
- **Big-endian hosts** — rejected with `NotImplementedError` rather than
  supported.
- **Any training run** — the Iron law and
  `.claude/notes/production-training-checklist.md` remain fully binding; nothing
  here is calibration evidence.
