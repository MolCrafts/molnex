---
slug: streaming-data-tiers-02-block-format
criteria:
  - id: ac-001
    summary: Append is incremental and a torn tail resumes byte-identically
    type: code
    pass_when: |
      Tests in tests/test_molix/test_store/test_block.py::TestBlockWriter show
      (a) each append_unit only appends to column tails (predecessor bytes
      unchanged: sha256 of the prefix of every column before and after an append
      is equal), and (b) after appending junk bytes to a column file and,
      separately, after truncating the last commit.log record, BlockWriter.resume
      truncates to the last crc32-valid record and the block sealed afterwards
      has per-column sha256 and block_digest identical to a block written with
      the same units without interruption.
    status: pending
  - id: ac-002
    summary: Unit integrity is structural — no frame-level ingestion path exists
    type: code
    pass_when: |
      Tests assert inspect.signature(BlockWriter) exposes no append_frame or
      per-frame entry point; append_unit raises ValueError naming the offending
      column when frame-space rows != n_frames, frame_atom-space rows !=
      n_frames * N, or any topology atom index falls outside [0, N); and for a
      sealed 3-unit block unit_frames(u) for u in range(unit_count) partitions
      range(frame_count) exactly with group_codes[f] == owning unit row for
      every f.
    status: pending
  - id: ac-003
    summary: Split purity is structural — one split per block, no per-frame field
    type: code
    pass_when: |
      A test asserts "split" is not a parameter of
      inspect.signature(BlockWriter.append_unit), no declared column has a name
      or route mentioning split, and manifest.json["split"] equals the
      BlockWriter constructor argument for a block whose units carry mixed task
      labels.
    status: pending
  - id: ac-004
    summary: Column reads are zero-copy file-backed mmap at offset 0
    type: code
    pass_when: |
      A test on a sealed block asserts
      reader.column("frame_pos").untyped_storage().filename ==
      str(root / "frame_pos.bin") (or, if the documented numpy fallback is in
      use, np.shares_memory with the np.memmap), and that the returned tensor
      equals the written tensor via torch.equal with the manifest-declared dtype
      and inner shape.
    status: pending
  - id: ac-005
    summary: Seal makes the block immutable and digest-verifiable
    type: code
    pass_when: |
      Tests show that after seal(): manifest.json has sealed true, a sha256 per
      declared column and a block_digest reproducible from the documented
      canonical JSON; append_unit raises ValueError; BlockReader.verify() passes;
      flipping one byte in one column makes verify() raise a ValueError naming
      that column; and BlockReader(root) on an unsealed block raises ValueError.
    status: pending
  - id: ac-006
    summary: frame_payload + collate_packed equals the collate_molecules oracle
    type: code
    pass_when: |
      A test builds identical samples into (a) a PackedCache v1 and (b) a sealed
      block, then asserts collate_packed(PackedView(reader.frame_payload(frames)),
      range(len(frames))) equals collate_packed over the v1 PackedView (and
      equals collate_molecules over the same per-sample dicts) leaf-for-leaf:
      identical key sets, torch.equal per leaf, identical dtypes, identical
      batch_size on atoms / edges / graphs / bonds / angles / propers and top
      level — for a fixture carrying bonds, angles and propers.
    status: pending
  - id: ac-007
    summary: Two-level row arithmetic matches hand-computed pointer literals
    type: code
    pass_when: |
      A test with N=(3,4) atoms and n_frames=(2,3) asserts the pointers derived
      at open equal the literals mol_atom_ptr=(0,3,7), mol_frame_ptr=(0,2,5),
      mol_frame_atom_ptr=(0,6,18), frame_mol=(0,0,1,1,1), atom_counts=(3,3,4,4,4),
      and that frame 3 resolves to frame-atom rows 10..14 and topology atom rows
      3..7; no pointer vector is present as a column file on disk.
    status: pending
  - id: ac-008
    summary: Sampler metadata accessors plan without unpacking; edges refuse loudly
    type: code
    pass_when: |
      A test constructs TaskMixtureBatchSampler with dataset=BlockReader(root),
      max_atoms set, task labels decoded from task_codes/task_names and group ids
      from group_codes, and asserts a non-empty deterministic plan is built while
      a counter proves no payload column other than the count vectors was mapped;
      a second test asserts reader.edge_counts raises ValueError whose message
      names the absent edge columns and points at NeighborList.
    status: pending
  - id: ac-009
    summary: block.py stays IO-only — no tensordict, no collate import
    type: code
    pass_when: |
      A test reads src/molix/store/block.py source (or src/molix/data/block.py
      pre-migration) and asserts neither "tensordict" nor "collate" appears in
      any import statement, and that imports are limited to stdlib, torch and
      (optionally) numpy.
    status: pending
  - id: ac-010
    summary: Regression example reproduces hard-coded block digests and parity
    type: runtime
    pass_when: |
      `python regressions/streaming-data-tiers-02-block-format.py` exits 0 on the
      branch: it writes two unit-aligned blocks, performs one torn-tail resume
      leg, seals, relocates block A to a second path, mmap-reads both back, and
      asserts (exactly, no tolerance) every per-column sha256 and both
      block_digest values against the literal goldens embedded in the script,
      unit integrity (unit_frames partition + whole units), verify() clean on the
      relocated copy, and collate_packed parity vs the PackedCache v1 oracle.
      The script imports no third party beyond torch/numpy and starts no
      subprocess.
    status: pending
  - id: ac-011
    summary: mmap-vs-pread and commit cost are measured and recorded, not assumed
    type: performance
    evaluator_hint: "scripts/store/bench_block_read.py"
    pass_when: |
      `python scripts/store/bench_block_read.py <sealed-block>` prints a table of
      MB/s for mmap-gather and os.pread column reads (sequential and
      shuffled-frame) plus per-unit append+commit+fsync cost (units/s, MB/s), and
      the block.py module docstring plus .claude/notes/notes.md record those
      numbers with client, filesystem, date, and an explicit verdict; if pread
      exceeds mmap by more than 1.5x the docstring records the named follow-up
      behind the _map_column seam.
    status: pending
  - id: ac-012
    summary: Public API carries Google docstrings with units, shapes and format rules
    type: docs
    pass_when: |
      SamplingUnit, BlockManifest, BlockWriter, BlockReader and every public
      method carry Google-style docstrings with Args/Returns/Raises and tensor
      shape + unit annotations (e.g. ``(F_atoms, 3)`` Å, ``(F,)`` kcal mol^-1);
      the block.py module docstring states the append/commit protocol, the
      derived-not-stored pointer rule, the 64-byte-alignment-moot rationale, the
      edge-column exclusion arithmetic, and the molpy.Block disambiguation;
      `ruff check` and `ruff format --check` are clean on the new files.
    status: pending
  - id: ac-013
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      On the branch containing this spec, `$META.build.check`,
      `python -m pytest tests/ -q` and `python scripts/check_test_mirror.py` all
      exit 0.
    status: pending
---

# Acceptance criteria

ac-001 through ac-005 are the format's five load-bearing properties, one
criterion each: streaming-appendable with crash resume (I5), structural unit
integrity (I2), structural split purity (I7), zero-copy mmap reads, and
seal-time immutability with embedded digests (the READY witness). ac-006 is the
equivalence oracle that keeps this format inside the existing loader instead of
forking a second collate path; ac-007 pins the two-level pointer arithmetic to
hand-derivable literals so a regression in the ragged layout cannot hide behind
round-trip symmetry. ac-008 proves the metadata surface the eager plan needs is
served without unpacking, and that the deliberate absence of edge columns fails
loudly rather than silently yielding zero. ac-009 is the layering guard mirroring
`dynamic-batching-packed-collate-02-collate` ac-005. ac-010 is the delivery gate
for the regression example this spec inherited from the strategy spec; ac-011
converts the Lustre mmap caveat from an assumption into a recorded measurement;
ac-012 and ac-013 are the docs and full-suite gates.

No `scientific`-typed criterion is emitted: `$META` configures no external bench
repository, the stored quantities carry no model to validate, and every numeric
claim in this spec is either exact (byte digests, integer pointer arithmetic,
`torch.equal`) or explicitly a measurement owned by ac-011.
