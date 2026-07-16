# MolNex — Evolving Decisions

Capture non-obvious decisions, trade-offs, and provisional rules here via `/mol:note`.
When an entry stabilises, `/mol:note` promotes it into `CLAUDE.md` and deletes it here.

This file is read by every `mol:*` skill / agent before running checks.

---

## TensorDict — no subclass for batch data (2026-05-18)

**Context.** `AtomData / EdgeData / GraphData / GraphBatch` were four empty
`TensorDict` subclasses that added zero functionality (no new methods, no
overrides) — only docstring-schema and type tags. They blocked `torch.compile`,
increased coupling to unstable tensordict subclass APIs, and created confusion
for newcomers needing to understand four type names for what is really just a
nested dict of tensors.

**Decision.** Removed all four subclasses (spec `tensordict-cleanup`).
Post-collate batch is now a plain `tensordict.TensorDict` with three nested
namespaces (`atoms`, `edges`, `graphs`). Schema contracts (field names,
shapes, `batch_size`) are documented in CLAUDE.md, not enforced by Python types.
Encoders are encouraged to inherit `TensorDictModuleBase` for `in_keys/out_keys`
validation, but `nn.Module` with `forward(td: TensorDict) -> TensorDict` is
also valid.

**Status.** stable (promoted to CLAUDE.md).

---

## Industrial module layout + test mirror (2026-07-16)

**Context.** PiNet train/infer work was blocked by god-files (`molzoo/pinet.py`
756 LOC, `molrep/interaction/pinet.py` 11 classes), dual homes
(heads/pooling/export), and tests that crossed package boundaries
(`test_molzoo` owning `PadMolecularBatch` and molrep layer checks).

**Decision — package ownership (one-way):**

```
molix  → infra (data, trainer, compile, export, md)
molrep → pure representation (embedding / interaction / readout)
molpot → physics (heads / derivation / potentials)
molzoo → recipes that wire molrep blocks into encoders (+ temporary
         potential façades under molzoo.pinet.potential for import
         stability; long-term home is molpot)
```

Hard rules:
1. `molrep` must not import `molpot` or `molzoo`.
2. `molpot` must not import `molzoo` (accept encoder Protocol / tensors).
3. One physics concept → one owner package (no duplicate CosineCutoff, etc.).
4. Prefer explicit `nn.Linear(in, out)` over `LazyLinear` on any export /
   functorch / compile path.

**Decision — source ↔ test mirror:**

```
src/<pkg>/<area>/<module>.py
  → tests/test_<pkg>/test_<area>/test_<module>.py
```

- One source module ↔ one primary test module.
- One public class ↔ dedicated test class (and preferred: dedicated file when
  the source file still hosts multiple public types during migration).
- Cross-package integration tests go under `tests/regression/` or an explicit
  `test_*_integration.py` name — never as a substitute for unit mirrors.
- Gate: `python scripts/check_test_mirror.py --strict-pinet`.

**PiNet spine layout (landed):**

```
src/molrep/interaction/pinet/{ff,message,residual,blocks}.py
src/molzoo/pinet/{spec,geometry,encoder,potential,properties}.py
tests/test_molrep/test_interaction/test_pinet/...
tests/test_molzoo/test_pinet/...
tests/test_molix/test_data/test_tasks/test_pad.py
```

**Status.** active (PiNet spine done; full-repo mirror is incremental).
