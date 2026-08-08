---
slug: mace-subpackage-restructure-07-cleanup
criteria:
  - id: ac-001
    summary: ensure_graphs honours num_graphs and stays backward compatible
    type: code
    pass_when: |
      tests/test_molpot/test_derivation/test_protocol.py::TestEnsureGraphs passes:
      ensure_graphs(TensorDict(batch_size=[]), num_graphs=3)["graphs"].batch_size ==
      torch.Size([3]); num_graphs=0 -> torch.Size([0]); omitting num_graphs still
      yields torch.Size([]); a pre-existing "graphs" sub-TensorDict is returned
      untouched (contents and batch_size unchanged).
    status: pending
  - id: ac-002
    summary: write_energy derives B so every in-tree write is schema-conforming
    type: code
    pass_when: |
      tests/test_molpot/test_derivation/test_protocol.py::TestWriteEnergyGraphShape
      passes: after write_energy(batch, energy) with energy.shape == (4,),
      batch["graphs"].batch_size == torch.Size([4]) and batch["graphs","energy"]
      equals the input tensor elementwise; a 0-dim energy leaves torch.Size([]) and
      raises nothing; tests/test_molpot/test_derivation/test_readouts.py passes
      unmodified.
    status: pending
  - id: ac-003
    summary: no consumer reaches into the private MACE energy core
    type: code
    pass_when: |
      `rg -n "_compute_energy" scripts/ benchmarks/` returns zero matches;
      scripts/matpes_port/run_nve.py and benchmarks/bench_mace_matpes.py both call the
      public energy_core; run_nve.py no longer defines build_model and no longer
      imports json; `ruff check src/ && ruff format --check src/` exits 0 with no F401.
    status: pending
  - id: ac-004
    summary: run_nve.py CLI and graph.pt output stay byte-compatible
    type: runtime
    pass_when: |
      `python scripts/matpes_port/run_nve.py --help` still advertises exactly the 21
      flags --structure --system --dump-system --weights-dir --out --steps --dt
      --temperature --stride --precision --checkpoint-every --resume --flush-every
      --seed --device --fallback --threads --rebuild-every --capacity-factor
      --autocast-bf16 --compile with unchanged defaults; the model returned by
      MACEPotential.from_checkpoint still answers float(model.cutoff_fn.r_cut); a
      --system run writes a .graph.pt whose keys are exactly {Z, cell, edge_index,
      shifts, num_edges, rebuild_count, energy_scale} and whose printed single-point
      E and |F|max match the pre-re-point run to all printed digits.
    status: pending
  - id: ac-005
    summary: EnergyForceModel is fully removed with no production fallout
    type: code
    pass_when: |
      src/molpot/composition/energy_force.py and
      tests/test_molpot/test_composition/test_energy_force.py no longer exist;
      `rg -n "EnergyForceModel" src/ tests/ docs/ scripts/ benchmarks/ .claude/`
      returns zero matches; `python -c "import molpot"` succeeds;
      `python -m pytest tests/ -q` is green; `python scripts/check_test_mirror.py
      --strict-pinet` exits 0.
    status: pending
  - id: ac-006
    summary: regression example reproduces the hard-coded energy/force goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-07-cleanup.py` exits 0 and prints
      RESULT: PASS, reproducing the embedded literal goldens (total E in eV, max|F| and
      F[0] in eV/Ang, captured at chain tip 06 before the re-point) to <= 1e-9 eV and
      <= 1e-9 eV/Ang on fp64 CPU, and asserting
      ensure_graphs(..., num_graphs=3)["graphs"].batch_size == torch.Size([3]); the
      script imports only molnex packages, torch and tensordict — no mace-torch, ASE or
      e3nn, and no subprocess.
    status: pending
  - id: ac-007
    summary: bench_mace_matpes keeps its PASS/FAIL contract on the public core
    type: runtime
    pass_when: |
      `python benchmarks/bench_mace_matpes.py --n-atoms 32 --steps 2` runs to a
      `RESULT: PASS` or `RESULT: FAIL` line and keeps the 0/1 exit-code contract; on
      CUDA the compiled arm wraps the public energy_core in Compiler(cuda_graphs=True);
      on CPU that arm remains skipped by the existing device.type guard.
    status: pending
  - id: ac-008
    summary: three molzoo specs re-anchored, structure and copies intact
    type: docs
    pass_when: |
      Every module/file/symbol anchor in the header table, section 2, 3.1, 5, 6, 7.2, 8
      and Appendix A of src/molzoo/specs/{mace,mace_matpes,mace_omol}.md resolves to a
      path or symbol that exists after the restructure (no anchor points at a deleted
      flat module); each file gains exactly one new section 9 row pinning the
      restructure; the 10 section headings are byte-identical to their pre-edit text;
      `diff docs/molzoo/specs/mace_omol.md src/molzoo/specs/mace_omol.md` is empty.
    status: pending
  - id: ac-009
    summary: notes reflect the new layout and the EnergyForceModel removal
    type: docs
    pass_when: |
      .claude/notes/architecture.md no longer lists src/molzoo/mace.py or
      src/molzoo/mace_omol.py as the MACE homes, its known-gap item no longer claims
      an un-consolidated EnergyForceModel (the pooling double-home and O(N^2) neighbour
      kernel items survive), and .claude/notes/notes.md no longer states "PiNet via
      EnergyForceModel".
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** are the schema fix. They are split because ac-001 binds the
  generalized `ensure_graphs` signature while ac-002 binds the caller migration —
  a partial landing (parameter added, `write_energy` untouched) would leave the
  latent `batch["graphs"].batch_size[0]` IndexError in `molzoo/pinet/potential.py`
  open.
- **ac-003 / ac-004** are the consumer re-point. ac-003 is the static side (no
  private-core reference survives, no lint fallout from the collapsed
  `build_model`); ac-004 is the behavioural side and is the chain's
  "CLI byte-compatible" invariant written as a checkable bar.
- **ac-005** is the deletion. The grep-zero plus a green full suite plus the mirror
  gate is the whole evidence that this was dead code.
- **ac-006** freezes the refactor's numerical inertness as literals in
  `regressions/`.
- **ac-007** guards the perf script's contract, not its numbers; the speedup guard
  itself is unchanged and remains GPU-only.
- **ac-008 / ac-009** are documentation truth. ac-008 deliberately also binds
  "10 section headings byte-identical" because restructuring an encoder spec is a
  section-10.2 breaking change, and binds the `docs/` copy because the two copies
  were already found drifted.
---
