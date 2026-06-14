---
name: molzoo-spec
description: Create, update, or append a benchmark-log row to a MolZoo encoder spec. Use when the user types `/molzoo-spec <encoder> ...` and wants to scaffold, edit, or log against `src/molzoo/specs/<encoder>.md`.
argument-hint: <encoder> [--log <k=v ...>] [--paper <url>] [--ref <org>/<repo>@<sha>]
user-invocable: true
---

# molzoo-spec

A MolZoo spec is a readable engineering contract. It must be precise enough to
drive implementation and audits, but it is also part of the rendered
documentation. Do not write it as a raw audit dump.

The reader should learn, in this order:

1. what the encoder owns
2. what it reads and writes
3. the forward equations and tensor shapes
4. the configuration contract
5. how the implementation maps to the paper/reference
6. which MolNex adaptations are intentional
7. how the contract is validated

Long paper-code crosswalks, benchmark logs, and drift policy belong after the
core contract, not before it.

## File Layout

The canonical source file is:

```text
src/molzoo/specs/<encoder>.md
```

The rendered documentation copy is:

```text
docs/molzoo/specs/<encoder>.md
```

Both files must be byte-identical after every spec edit. Verify with:

```bash
diff -u src/molzoo/specs/<encoder>.md docs/molzoo/specs/<encoder>.md
```

Every public spec must also be present in `zensical.toml` under:

```toml
{ "MolZoo" = [
    ...
    { "Spec" = [
        { "<Encoder> Spec" = "molzoo/specs/<encoder>.md" }
    ]}
]}
```

If the docs copy changes, run `zensical build --clean`.

## Spec-First Principle

Specs are written from the paper and pinned reference implementation before
new encoder code is written. Implementation is a translation of the spec. If
the implementer must invent behavior, the spec is incomplete.

When code, spec, paper, and reference disagree:

- if the spec conflicts with the paper/reference and code is already correct,
  patch the spec
- if code conflicts with a correctly transcribed spec, patch the code (this is
  the `molzoo-auditor` agent's territory — it prints a `⚠️` verdict; this skill
  does not edit code)
- if the reference pin is missing, state that explicitly in §9 and require a
  follow-up audit to pin it

## Argument Parsing

Parse `$ARGUMENTS` left-to-right.

1. The first positional token is `<encoder>`. It must match `^[a-z][a-z0-9_]*$`;
   if it does not, abort and ask the user to rename. Do not silently coerce.
2. If the remaining tokens contain `--log`, dispatch to **log** mode and treat
   every following `k=v` token as a log field (see §"Mode: log").
3. Otherwise, mode is determined by filesystem state:
   - `src/molzoo/specs/<encoder>.md` does **not** exist → **create** mode
     (requires `--paper`; abort with a usage hint if missing).
   - `src/molzoo/specs/<encoder>.md` **does** exist → **update** mode.
   Announce the inferred mode in one sentence before acting.
4. Collect optional context flags:
   - `--paper <url>`: required for **create**; ignored with a warning in other
     modes.
   - `--ref <org>/<repo>@<sha>`: meaningful for **create** and **update**;
     ignored with a warning in **log**.
5. If `<encoder>` is missing entirely, refuse with a one-line usage hint.

## Modes

- **create**: spec missing; copy `template.md` to both spec locations and seed
  the header / §1 / §9 from the paper + reference pin.
- **update**: spec exists; fill placeholders, reconcile drift between spec /
  code / paper / reference, and refresh stale anchors. The same operation
  covers both first-time fill (status `draft` → `partial`) and later repair
  (status `partial`/`stable` with placeholders or contradictions).
- **log**: append a benchmark/training row to §7.4.

## Writing Rules

### Readability

- Put public contract and equations before crosswalk tables.
- Use stable anchors such as class names, method names, TensorDict keys, module
  symbols, and test paths.
- Avoid exact code line numbers in public docs. If exact line numbers are used
  during an audit, verify them in the same turn and remove them if they are not
  necessary.
- Stable specs must not contain `<...>` placeholders. If a claim is not yet
  supported, write `not claimed` and explain the missing evidence.
- Keep changelogs and run logs short. Detailed audit chatter belongs in chat or
  appendices, not in the main contract.

### Consistency Guard

Before finishing any spec edit, search for contradictions:

- If the spec says there is no extra scalar MLP, no section may introduce a
  `scalar_embed_mlp` or `ScalarMLP` in that path.
- If the spec says tensor products directly target output irreps, no equation
  may include a post-TP `W_proj` or `cuet.Linear`.
- If the code has optional outputs such as `edge_tensor_features`, the public
  contract and system boundary must mention when they are written.
- If a config field exists in `AllegroSpec`, it must appear in §4.
- If a TensorDict key appears in code or tests, it must appear in §2 or be
  explicitly marked internal.

### Source and Docs Sync

After editing one spec copy, mirror the same content into the other copy and
verify `diff -u` is empty. Never let `src/molzoo/specs/<encoder>.md` and
`docs/molzoo/specs/<encoder>.md` drift.

## Mode: create

Preconditions:

- spec files do not exist
- `--paper` is provided
- reference repo is provided with `--ref` or can be resolved and pinned

Actions:

1. Fetch the paper metadata and reference implementation. Do not invent
   citations.
2. Copy `template.md` (sibling of this `SKILL.md`) to both
   `src/molzoo/specs/<encoder>.md` and `docs/molzoo/specs/<encoder>.md`,
   substituting `<Encoder>` / `<encoder>` placeholders in the header.
3. Fill only the header, §1, §2 skeleton, §9 rows that are known, and an
   appendix maintenance entry. Status starts `draft`.
4. Add the docs copy to `zensical.toml` under `MolZoo -> Spec`.
5. Do not create or modify `src/molzoo/<encoder>.py`.

Print the mandatory next step:

```text
/molzoo-spec <encoder>     # → update mode: fill §2 / §3 / §5 from paper + ref
```

## Mode: update

Use update mode whenever the spec already exists. The same actions cover both
first-time fill (placeholders → content) and subsequent repair (drift,
contradictions, stale anchors).

Actions:

1. Read the spec, encoder code (if it exists), relevant tests, and downstream
   consumers.
2. Read the paper and pinned reference source directly. Do not paraphrase from
   memory.
3. Populate or correct §3 equations and §5 reference crosswalk from those
   sources.
4. Fix stale line-number references by replacing them with stable anchors.
5. Remove contradictions between the math contract, crosswalk, adaptations,
   and code.
6. Confirm optional outputs and current config fields are documented.
7. Replace unsupported reproduction placeholders with `not claimed` language.
8. Bump status: `draft` → `partial` once §2 / §3.1 / §5 are filled; `partial`
   → `stable` only when every §2 row has a non-`unknown` status.
9. Mirror source/docs specs and verify the diff is empty.

Update mode may patch the spec, the docs mirror, and nearby documentation that
references old section numbers. It must not write encoder code and must not
silently change encoder behavior. Drift discovered between **code** and a
correctly-transcribed spec is reported, not fixed — point the user at the
`molzoo-auditor` agent.

## Mode: log

Accepted keys:

```text
run_id date commit dirty dataset config steps train_mae val_mae fwd_ms bwd_ms compiled note
```

Defaults:

- `run_id`: previous row + 1
- `date`: today
- `commit`: `git rev-parse --short HEAD`
- `dirty`: 0 or 1 from `git status --porcelain`

Append the row to §7.4 in both spec copies. Do not reformat earlier rows.

## Lookup Behavior (no separate mode)

When the user asks a question about an encoder's spec content (e.g. "why is
our Bessel RBF normalised?", "what does §3.2 say about the tensor product?"):

1. Always `Read` `src/molzoo/specs/<encoder>.md` first. Never answer from
   memory.
2. Quote matched sections verbatim when short; cite §-numbers for long
   sections.
3. If no section answers the question, say the spec has a gap and recommend
   the `molzoo-auditor` agent. Do not invent the missing answer.
4. If the spec status is `draft` (placeholders unfilled), refuse and instruct:
   "spec status=draft; run `/molzoo-spec <enc>` (update mode) to fill §2 /
   §3.1 / §5 first."

This is a behavioural contract, not a dispatched mode — questions about spec
content do not invoke this skill via `/molzoo-spec`; they are answered inline
in the calling conversation under these rules.

## Hard Rules

- Do not author encoder code while §3 or §5 still contain placeholders.
- Do not use the crosswalk as the first explanatory section.
- Do not leave public specs with `<...>` placeholders after status is marked
  `stable`.
- Do not cite code line numbers unless you verified them in the same operation.
- Do not write a docs-only spec. `src` and `docs` copies must match.
- Do not add a MolZoo `Explanation` page for model theory. Theory belongs in
  the model user guide; exact implementation contracts belong in `Spec`.
- When asked about spec content, follow §"Lookup Behavior" — always Read,
  never paraphrase, refuse on a miss.

## Spec Template

The 10-section template lives in `template.md` next to this file. Read it from
disk in create mode; do not paraphrase it from memory.
