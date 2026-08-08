---
slug: mace-subpackage-restructure-01-blocks
criteria:
  - id: ac-001
    summary: MACE-only blocks land in the three new mace homes with unchanged symbol names
    type: code
    pass_when: |
      python -c "import molrep.interaction.mace as m; [getattr(m, n) for n in
      ('ConvTP','ConvTPSpec','InteractionBlock','InteractionSpec','DensityInteraction',
      'DensityResidualInteraction')]" exits 0; src/molrep/readout/mace.py defines
      LinearReadout, NonLinearReadout, NonLinearBiasReadout, ProductHead, ProductHeadSpec and
      src/molrep/embedding/mace.py defines EmbeddingSpec, EmbeddingBlock; and
      `grep -nE "^(class|def) " src/molrep/interaction/density.py
      src/molrep/readout/{scalar,product}.py` prints nothing (legacy modules are pure shims).
      residual.py / product_basis.py / element.py are byte-unchanged (deferred to -01b).
    status: pending
  - id: ac-002
    summary: legacy import paths still resolve to the moved objects, no import cycle
    type: runtime
    pass_when: |
      python -m pytest tests/test_molrep/test_reexport_compat.py -v passes: all 12 legacy
      attributes satisfy `old is new` (e.g. molrep.interaction.density.DensityInteraction is
      molrep.interaction.mace.density.DensityInteraction; molrep.readout.scalar.LinearReadout is
      molrep.readout.mace.LinearReadout; molzoo.mace.EmbeddingBlock is
      molrep.embedding.mace.EmbeddingBlock), molrep.interaction.__all__ and molrep.readout.__all__
      equal their pre-move lists, and `python -c "import molrep, molrep.interaction,
      molrep.readout, molzoo.mace, molzoo.mace_omol, molzoo.mace_matpes"` exits 0.
    status: pending
  - id: ac-003
    summary: pure move — no cuEquivariance construction site is altered
    type: code
    pass_when: |
      `git diff -M -C -U0 HEAD~ -- src/molrep/interaction/mace/ src/molrep/readout/mace.py
      src/molrep/embedding/mace.py src/molrep/interaction/product.py | grep -E
      "^[-+][^-+].*(cuet\.|cue\.Irreps|layout=|use_fallback|dtype=|config\.ftype|method=|
      MomentNormalizedMLP|scatter_sum)"` prints nothing, and the remaining +/- lines inside the
      moved bodies are import statements or module docstrings only. src/molzoo/{mace,mace_omol,
      mace_matpes}.py diffs touch import lines (and mace.py's re-export block) exclusively.
    status: pending
  - id: ac-004
    summary: shared blocks and the Allegro path are byte-unchanged
    type: code
    pass_when: |
      `git diff --exit-code HEAD~ -- src/molrep/interaction/contraction.py
      src/molrep/interaction/radial.py src/molrep/interaction/gate.py
      src/molrep/interaction/linear.py src/molrep/interaction/aggregation.py
      src/molrep/interaction/residual.py src/molrep/interaction/product_basis.py
      src/molrep/interaction/element.py src/molzoo/mace_omol.py
      src/molzoo/allegro.py` returns 0, and EquivariantPolynomialTP, irreps_from_l_max,
      sh_irreps_from_l_max are still *defined* (not re-exported) in
      src/molrep/interaction/product.py.
    status: pending
  - id: ac-005
    summary: exact test mirrors exist for every module in the new namespace and the gate is strict
    type: runtime
    pass_when: |
      tests/test_molrep/test_interaction/test_mace/{test_conv,test_block,test_density}.py,
      tests/test_molrep/test_readout/test_mace.py
      and tests/test_molrep/test_embedding/test_mace.py all exist and pass; scripts/check_test_mirror.py
      PINET_SPINE contains "molrep/interaction/mace/" and `python scripts/check_test_mirror.py
      --strict-pinet` exits 0.
    status: pending
  - id: ac-006
    summary: existing MACE suites show no new failures versus the pre-move baseline
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molzoo/test_mace.py tests/test_molzoo/test_mace_encoder.py
      tests/test_molzoo/test_mace_omol.py tests/test_molzoo/test_mace_matpes.py
      tests/test_molzoo/test_symmetry.py tests/test_molrep/test_interaction
      tests/test_molrep/test_readout -q` reports pass/fail/error counts equal to the baseline
      recorded on the parent commit before task 4; no test is newly skipped, xfailed or
      assertion-weakened to reach that state.
    status: pending
  - id: ac-007
    summary: regression example reproduces hard-coded pre-move goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-01-blocks.py` exits 0 on CPU float64,
      reproducing every embedded golden (per-block output sum and abs-max) within rtol 1e-12 and
      matching every embedded sorted state_dict() key list exactly; the file imports only molnex
      packages + torch (no ASE / e3nn / mace-torch, no subprocess) and carries a comment recording
      the capture command, parent commit sha, torch version and date.
    status: pending
  - id: ac-008
    summary: repo-wide format, lint and test gate stay green
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` exits 0 and `python -m pytest tests/ -v`
      shows no failure absent from the pre-move baseline.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-003 / ac-004** are the "it really was a move" contract: the new homes hold the
  code, the legacy files hold nothing but re-exports, and no cuEquivariance instantiation
  (`layout`, `use_fallback`, `dtype`, `method`) shifted a byte. ac-004 additionally pins the
  blast radius: shared blocks and `molzoo/allegro.py` must come out of this step untouched.
- **ac-002** protects every current consumer (`molzoo.mace_matpes:50`, `molzoo.mace_omol:36`,
  `tests/test_molzoo/test_mace.py:15,17`, `molrep/__init__.py:35`) during the shim window, and
  doubles as the import-cycle guard for the new `molrep.interaction.mace` package.
- **ac-005** turns the new namespace into hard-gated territory (`--strict-pinet`) and closes the
  pre-existing mirror gap for the scalar readouts (`residual.py` / `product_basis.py` mirrors
  land with -01b). Exact
  mirror paths are required because `check_test_mirror.py`'s fuzzy fallback would otherwise
  match `test_pinet/test_residual.py` for `mace/residual.py`.
- **ac-006** is deliberately baseline-relative: `tests/test_molzoo/test_symmetry.py` carries
  known pre-existing `setup_context` failures (documented in
  `.claude/specs/cuet-force-doublebackward.md`). Comparing counts keeps that debt visible instead
  of laundering it with skip marks.
- **ac-007** establishes `regressions/` for the whole chain. It stays `type: runtime` (verified by
  `/mol:impl` at delivery) because the goldens are this repo's own pre-move baseline, not an
  external benchmark.
- **ac-008** is the delivery gate.
