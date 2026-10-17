# MolCrafts 0.17 plan (2026-10-08)

## Working rules
- **One repo at a time.** Finish a repo, all the way to green and merged, before starting the next.
- **Fork CI first.** Push to Roy-Kid's fork, wait for green there, then open the PR to MolCrafts. Merge only when green.
- **Never bypass:** no `--no-verify`, no `--admin`, no installs into `~/work/.x86_64`.
- **No compatibility code before 1.0:** no shims, no migration notes, no changelogs in the 0.x repos.
- **`dev` tracks partners' `dev`.** Pins appear only in release commits.
- **Release order:** molrs, then molcfg / mollog (independent), then molpy, then molpack.
- **Nightly order:** molrs, then molpy.

## Step 0. Close out 0.16
- [x] molrs 0.16.0: crates.io, PyPI, npm, GitHub Release; `master` merged back to `dev`.
- [x] molcfg 1.5.1, mollog 1.3.1: PyPI and GitHub Release.
- [x] molpy 0.16.0: PyPI and GitHub Release; merged back to `dev`.
- [ ] molpack 0.4.0. PR #43, then tag, then merge back to `dev`, where `MOLRS_REF=dev` again.
- [ ] Manually start the molrs and molpy nightly runs on `master`, so the dashboard shows 0.16.0.
- [ ] Bump `dev` to 0.17.0 in molrs and molpy together, keeping the shared major.minor rule; molpy pins `molcrafts-molrs>=0.17.0,<0.18`. Bump molpack `dev` to 0.5.0, pinned to molrs 0.17.

## Step 1. Per-project uv venvs (environment)
The runbook is at `…/molcrafts/.venv-migration.md` and the staged config at `~/.hpc/*.new`.
- No global tools and no global Python. Every command runs as `uv run` or `uvx`. Venvs are named per architecture: `.venv-$(uname -m)`.
- Order:
  1. Revise the staged hooks to use `uv run` only.
  2. Build each repo's venv.
  3. Switch `HPC_VENV_ARCHES="aarch64"` (needs the user's go-ahead, timed so it doesn't disrupt the Atomiverse session).
  4. Rename `.x86_64` to `.x86_64.retired`, and delete it after 2 weeks.
- Keep `.aarch64` until the research projects have their own venvs.
- Done when: every molcrafts repo's hooks and tests pass with no `.x86_64` on PATH.

## Step 2. Downstream catches up with 0.16 (one repo at a time)
1. **molvis.** Move to the molrs 0.16 stream readers, `io.<fmt>.<Fmt>Reader`, and `readFrame(index)`.
2. **molnex.** Port the gaps the other session's `ir/molnex-gaps` branch addresses, and coordinate with that session first. Covers the CPU torch index plus a relock, and the `build_env.sh` that copies `.x86_64`.
3. **molhub.** `dev` is already ported. The `feat/mm-param-val` owner still has to rebase; the overlap list is in the molhub port report.
4. **Atomiverse.** Merge `compat/molrs-016` (local, unpushed) once the other session's work lands. Re-pin the gitlink to molrs v0.16.0, and follow the history-rewrite note `.ci-split/ATOMIVERSE-HISTORY-REWRITE.md`.

Done when each repo builds and tests against molrs/molpy 0.16 (or `dev`), with green CI.

## Step 3. molpy on mollog + molcfg (the 0.17 feature)
- **Dependencies:** molpy depends on `molcrafts-mollog` and `molcrafts-molcfg`. `dev` tracks their `master`, since they have no `dev`; add them to `partners.env`.
- **Logging:** engines and wrappers (LAMMPS, GROMACS, OpenMM, CP2K, AmberTools: antechamber, tleap, parmchk2, prepgen, sander) log through mollog, as structured records. No `print`, and no ad-hoc `logging.getLogger` configuration.
- **Configuration:** engine and wrapper settings resolve through molcfg layers: package defaults, then user, then project, then run. Executable paths, environment specs (conda/venv) and resource settings live there.
- **Release order:** mollog and molcfg before molpy, whenever molpy needs a new API from them.
- **Done when:** no engine or wrapper reads config or prints logs on its own. Tests cover the layering, and docs show one config file driving a run.

## Step 4. Force-field IR verification closes
- **class2 against LAMMPS.** Build a LAMMPS with the CLASS2 package, on a compute node, under `.build-alloc`. Run the class2 `bond_angle` case of `ff_ir_extension_lammps_check.sh`.
- **MACE.** Get the official MACE weights and run `regressions/mace-official-weights.py` once.
- **Done when:** the IR completeness proof has no skipped cells.

## Step 5. Honest quality reporting
Report only; no thresholds.
- **PR summaries.** `test / python` and `test / rust` write the pass, fail and skip counts and the coverage figure to the GitHub step summary.
- **Dashboard.** Nightly records on `master` keep feeding it. molpack's Packmol parity results are submitted as a record too.
- **Docstrings.** Fix the 3 griffe docstring/signature mismatches (`molrs.conformer`, `molrs.builder`, `molrs.compute`).

## Later, not scheduled
- The `test / browser` replacement for molvis, if UI coverage is wanted again, as a separate non-unit suite.
- Open org decisions for the user: the org Actions setting for private-repo fork PRs, the 4 blocked private PRs, the GitHub Support request for the Atomiverse `refs/pull/2-4`, and Team plan vs. public repos.
