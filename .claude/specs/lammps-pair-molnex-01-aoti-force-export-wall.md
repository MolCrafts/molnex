---
title: pair_style molnex — AOTI force-export wall + TorchScript route forward
status: investigation-complete
chain: lammps-pair-molnex
created: 2026-06-23
---

# pair_style molnex — AOTI force-export wall + TorchScript route forward

## Summary

Goal was to ship the fine-tuned MACE-OMol potential into LAMMPS as a native
`pair_style molnex` backed by an AOT-Inductor `.so` (the existing
`molcrafts/molnex/interface/lammps` plugin is AOTI-only), so MD runs the model
in-process with no Python and no `fix external` round-trip — then drive CSVR-
thermostatted NaCl(aq) MD for MSD/diffusion.

**This route is blocked at the framework level, not by our code.** AOT Inductor
exports the MACE-OMol *energy* bit-exact but exports the *forces as identically
zero*, because the `cuequivariance` fused tensor-product ops have no
export-traceable backward. Forces in this model are `F = -dE/dx` computed by
`torch.autograd.grad`/`torch.func.grad` *inside* `forward`; AOTI can trace the
forward energy graph but cannot lower the autodiff backward through the cuet
fused kernels, so the gradient collapses to 0.

Recommended path forward: **TorchScript + libtorch `pair_style`** (the upstream
`pair_style mace` pattern), where the C++ side holds the scripted module and
calls `.backward()` at runtime — autodiff happens live in libtorch instead of
being baked into a `.so` at export time. This sidesteps the cuet-backward-export
wall entirely. This spec records the AOTI investigation so we do not re-walk it,
and scopes the TorchScript replacement.

## Investigation — the AOTI wall chain

Reproduced on the `.aarch64` GPU venv (torch 2.12.0+cu130, GH200), full
MACE-OMol foundation (1024 feat, l_max=3, 3 interactions), NaCl(aq) box.
Gate script: `$CLAUDE_JOB_DIR/tmp/aoti_gate.py` (eager-vs-AOTI E/F diff, hard
fail on `max|dF| > tol`). Each wall, in the order hit:

| # | Symptom | Cause | Action |
|---|---------|-------|--------|
| 1 | `Unsupported: torch.autograd.grad with trace_autograd_ops=False` | dynamo/export refuses to trace autograd ops by default | set `torch._dynamo.config.trace_autograd_ops = True` |
| 2 | `Unsupported: returning intermediate with requires_grad_()` | force path returned a tensor that had `requires_grad_()` set on an intermediate | switch force to a clean leaf-input formulation |
| 3 | (user) "不是说用functorch吗" — still on `torch.autograd.grad` | autograd.grad path is the fragile one for export | switch force to `torch.func.grad` (functorch), energy as pure fn of positions |
| 4 | `InductorError: LoweringException: 'Constant' object has no attribute 'data'` on `einsum('abc,->abc', x, scalar)` | cuet emits a scalar-broadcast einsum the Inductor lowering can't handle | monkeypatch `torch.einsum('abc,->abc', x, s)` → `x * s` |
| 5 | **HARD WALL.** AOTI compiles & loads. Energy matches bit-exact (`eager -128066.5225` == `aoti -128066.5225`). **Forces `Fmax=0.0000`** vs eager `Fmax≈9.56e5`; `max|dF|≈9.6e5`. | cuequivariance fused tensor-product ops have **no export-traceable backward** — the autodiff graph through them does not survive AOTI lowering, so `-dE/dx` lowers to zero | **none available within AOTI** — abandon AOTI for force export |

Gate verdicts on record (job ids): v2 `140000` wall #2, v3 `140004` wall #2/#3,
v5 `140011` wall #4 (functorch), v6 `140017` wall #5 (einsum patched →
`E` exact, `Fmax=0.0000`, GATE FAIL).

Energy-only export is unaffected — wall #5 is specific to the differentiated
(force) graph.

### Why this is not our bug

- Eager forces are correct and finite (`Fmax≈9.56e5` on the perturbed gen-0
  box; large but that box is a bootstrap perturbation, not equilibrated).
- The energy graph exports bit-exact, so the forward port is sound.
- The zero is introduced precisely at the autodiff-through-cuet boundary —
  the same class of issue as `cuet-force-doublebackward` (cuet equivariant
  ops being hostile to the autodiff machinery), here at *export* rather than
  *double-backward* time.

## Code touched during investigation (keep / revert decision)

- `src/molix/export.py` — added `torch._dynamo.config.trace_autograd_ops = True`
  and `torch.no_grad()` → `torch.enable_grad()` around `aot_compile`. **Keep.**
  These are correct and necessary for *any* future autodiff-force export, and
  are inert for energy-only exports. They are not the blocker.
- `torch.einsum` monkeypatch (wall #4) — lived only in the gate script, **not**
  in `src/`. Do not upstream; it is a symptom workaround for a path we are
  abandoning.
- `src/molzoo/mace_omol.py` force path uses `torch.autograd.grad`
  (`energy_forces`, ~line 221); `_compute_energy` is already a pure fn of
  positions, so a `torch.func.grad` reformulation is cheap if needed. Unchanged
  by this spec.

## Design — TorchScript + libtorch pair_style (route forward)

Mirror upstream `pair_style mace`:

1. **Export**: `torch.jit.script`/`torch.jit.trace` the MACE-OMol module to a
   `.pt` TorchScript archive holding weights + graph. No backward baked in —
   the scripted module exposes the *energy* forward only.
2. **Runtime (C++)**: `pair_style molnex` loads the archive via libtorch, builds
   the neighbor-list tensors (positions as a leaf with `requires_grad=True`),
   runs the forward to get energy, then calls `energy.backward()` *in C++*;
   `pos.grad` gives `-F`. Autodiff runs live in libtorch, where cuet's eager
   backward works — never goes through Inductor/AOTI lowering.
3. **Protocol**: define the tensor contract at the C++↔model seam — inputs
   (`Z`, `pos`, `edge_index`/neighbor pairs, `cell`/shifts, `total_charge`,
   `total_spin`), outputs (`energy`, optionally per-atom). Charge/spin are
   per-system scalars (MACE-OMol is charge/spin-conditioned) and must be passed
   through, defaulting to `(0, 1)` (neutral singlet).
4. **CSVR + MSD**: once forces are live, standard LAMMPS `fix temp/csvr` +
   per-species `compute msd` for the NaCl(aq) diffusion measurement. No custom
   integrator needed.

The existing `interface/lammps` AOTI plugin is **not reusable** for forces —
it loads a `.so` produced by AOTI and so inherits wall #5. The TorchScript
pair_style is a separate C++ target.

## Files

- `src/molix/export.py` — keep the two-line autograd-export enablement (above);
  add a `export_torchscript(model, ...)` entry point when the route is built.
  (No change required by *this* spec — it is investigation-of-record + scope.)
- `interface/lammps/` — new TorchScript pair_style target lives here alongside
  the existing AOTI plugin; do not delete the AOTI plugin (still valid for
  energy-only / non-cuet models).
- `.claude/specs/lammps-pair-molnex-01-...` — this file.

## Tasks (for the route-forward spec, not done here)

1. `export_torchscript(model, path)` in `src/molix/export.py` — script/trace
   MACE-OMol energy forward to `.pt`; round-trip test E bit-exact vs eager.
2. C++ `pair_style molnex` (libtorch): load `.pt`, neighbor-list → tensors,
   forward + `backward()` → forces. Unit: forces match eager `-dE/dx` to tol.
3. Wire CSVR (`fix temp/csvr`) + per-species `compute msd`; NaCl(aq) run →
   D(Na⁺), D(Cl⁻), D(H₂O).

## Testing

- **AOTI wall is reproducible** (regression guard so we do not re-attempt
  blindly): `aoti_gate.py` asserts AOTI forces ≠ eager forces for a cuet model
  → documents the wall; flip to "fixed" only if a future torch lowers the cuet
  backward.
- **TorchScript route**: energy round-trip bit-exact; forces from C++
  `backward()` match eager `-dE/dx` within port tolerance (≤1e-4 eV·Å, the
  bar from `mace-omol-port-02`).

## Out of scope

- Re-attempting AOTI force export (blocked until cuet ships export-traceable
  backward, or the model drops cuet on the export path).
- The `fix external` eager-Python fallback (`nacl_md/lammps_msd.py`) — works but
  is the slow round-trip the user explicitly rejected ("仍然使用.so").
- Energy-only AOTI export (already works; not the deliverable).
