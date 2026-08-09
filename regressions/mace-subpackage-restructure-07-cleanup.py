"""Refactor invariance of the public MACE energy core (chain step 07-cleanup).

Chain step 07 closes three tails: the two in-tree consumers
(`scripts/matpes_port/run_nve.py`, `benchmarks/bench_mace_matpes.py`) stop
reaching into the private core `model._compute_energy` and call the public
`energy_core` landed in step 04; `molpot.composition.EnergyForceModel` is
deleted; and `molpot.derivation.protocol.ensure_graphs` learns a `num_graphs`
argument so the `graphs` namespace it builds carries the schema-conforming
`batch_size=[B]`. The physics is supposed to be untouched — this file is the
standalone check of that claim (spec
`.claude/specs/mace-subpackage-restructure-07-cleanup.md`).

Scenario (public API only), two sections:

1. **Energy and forces through the public core.** A hard-coded 6-atom water
   dimer is pushed through `MACEMatpes.energy_core` inside a literal copy of
   `run_nve._matpes_energy_forces`' closure shape — position leaf with
   `requires_grad_(True)`, energy under `torch.enable_grad()`, and
   `autograd_forces_from_energy` **outside** any compile region (no `Compiler`
   here: `Compiler(cuda_graphs=True)` needs a GPU, and what is being guarded is
   the numbers, not the capture). The positional argument order
   `(pos, Z, edge_index, batch, num_graphs, shifts)` is the part of the
   re-point that could silently rot, so it is reproduced verbatim. Total
   energy, `max|F|` and `F[0]` are compared against goldens captured through
   the *private* core the re-point replaced.
2. **`ensure_graphs` schema.** The one-line public-surface change of this step:
   `ensure_graphs(TensorDict(batch_size=[]), num_graphs=3)["graphs"].batch_size`
   must be `torch.Size([3])`, the `"graphs": batch_size=[B]` shape CLAUDE.md
   specifies and `src/molzoo/pinet/potential.py` already reads `[0]` from.

Random weights under a fixed seed, and an `E0` table (`[-1.0, -8.0]` eV/atom)
that is literal rather than physical: what is pinned is *refactor invariance*,
not accuracy against a reference implementation. The chain's parity against
upstream MACE is a separate, already-recorded verdict (`src/molzoo/specs/
mace_matpes.md` §7.1) and cannot be re-run in-tree — mace-torch and e3nn are
not importable here by policy.

Goldens
-------
    oracle   : molnex itself. No third-party oracle (no mace-torch, no e3nn,
               no ASE), no network, no subprocess.
    path     : captured on the PRIVATE core `model._compute_energy` — the path
               task 3 of this spec re-points away from — so the literals below
               predate the change they are guarding. In the same capture the
               public `energy_core` was run on the identical inputs and came
               out bit-equal (`|dE| = 0.0`, `max|dF| = 0.0`,
               `torch.equal` True for both), which is expected rather than
               lucky: `MACEMatpes._compute_energy.__func__ is
               MACEPotential.energy_core` was also asserted True at capture
               time. The private name is deliberately *not* touched by this
               script — public API only.
    commit   : e825a51 (e825a515342a5f5f41b70d6443bab5f5ee957299), chain tip 06
               "refactor(molzoo): cut over to the mace subpackage; retire flat
               modules", `src/` clean.
    command  : PYTHONPATH=src python capture_goldens_07.py
    torch    : 2.12.1+cpu (tensordict 0.13.0)
    date     : 2026-08-09
    device   : CPU, float64 (`molix.config.set_precision("fp64")` before any
               construction), `use_fallback=True` — no
               `cuequivariance-ops-torch` wheel on this host, so the pure-torch
               cuEquivariance path is the only one available.
    observed : repeat runs at a fixed thread count are bit-identical, and the
               energy is bit-identical at every thread count tried. The
               *forces* are not: re-measured on this host at
               `OMP_NUM_THREADS` 1 / 2 / 4 / 8, `max|F|` moves by up to
               3.2e-12 eV·A^-1 against the literals below (worst case
               `OMP_NUM_THREADS=1`; 2, 4 and 8 agree with each other to
               4.5e-13). The goldens are the default-thread values and
               reproduce exactly there. So: do **not** tighten this file to
               `torch.equal`. The spec's 1e-9 band is ~300x above that
               reduction-order jitter and still ~2000 ULP below anything a
               real behaviour change would produce, which is exactly the
               separation it was chosen for.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-07-cleanup.py
"""

from __future__ import annotations

import sys
import textwrap
import traceback
from collections.abc import Callable

import torch

from molix import config

# Must precede every construction: the layers bake `config.ftype` in at
# __init__ time, so switching precision afterwards silently leaves an fp32
# model behind and the goldens below stop meaning anything.
config.set_precision("fp64")

from tensordict import TensorDict  # noqa: E402

from molpot.derivation.force import autograd_forces_from_energy  # noqa: E402
from molpot.derivation.protocol import ensure_graphs  # noqa: E402
from molzoo.mace import MACEMatpes  # noqa: E402

# ---------------------------------------------------------------------------
# The system. A 6-atom non-periodic water dimer (A), O-O 3.2 A, so that both
# intra- and inter-molecular edges fall inside the 4.0 A cutoff. Literal
# coordinates: a golden whose geometry is generated is not a golden.
# ---------------------------------------------------------------------------

#: Cartesian positions in Angstrom, in `ATOMIC_NUMBERS` order.
POSITIONS: list[list[float]] = [
    [0.00000, 0.00000, 0.00000],  # O
    [0.95720, 0.00000, 0.00000],  # H
    [-0.23999, 0.92663, 0.00000],  # H
    [3.20000, 0.10000, 0.20000],  # O
    [3.80000, 0.85000, -0.10000],  # H
    [3.60000, -0.70000, 0.60000],  # H
]

#: Per-atom atomic numbers.
ATOMIC_NUMBERS: list[int] = [8, 1, 1, 8, 1, 1]

#: Number of graphs `B` in the batch: one molecule-pair, so `batch = zeros(N)`.
NUM_GRAPHS = 1

# ---------------------------------------------------------------------------
# The model. Small MACE-MatPES built through the public keyword constructor.
# ---------------------------------------------------------------------------

#: Element table (z-table), strictly ascending as `torch.searchsorted` needs.
Z_TABLE: list[int] = [1, 8]

#: Reference energies `E0` in eV/atom, in `Z_TABLE` order. Arbitrary but
#: literal, and non-zero so the isolated-atom reference path contributes to the
#: total instead of being a silent no-op.
ATOMIC_ENERGIES: list[float] = [-1.0, -8.0]

#: Cutoff radius in Angstrom; also the neighbour-list radius below.
R_MAX = 4.0

# ---------------------------------------------------------------------------
# Goldens. See the module docstring for provenance.
# ---------------------------------------------------------------------------

#: Edge count of the `r < R_MAX` ordered-pair graph on `POSITIONS` (structural,
#: so exact): every pair of the 6 atoms except the four O-H/H-H pairs that
#: straddle the two molecules at more than 4 A.
GOLDEN_NUM_EDGES = 26

#: Total energy in eV (sum over the `(B,)` per-graph energies).
GOLDEN_ENERGY = -4014.3426025127233

#: Largest force component magnitude in eV/A.
GOLDEN_MAX_FORCE = 2833.787986314126

#: Force on atom 0 (the first O) in eV/A.
GOLDEN_FORCE_ATOM0: list[float] = [
    -2833.787986314126,
    -2349.1750081421415,
    -548.5925878141118,
]

#: Refactor-invariance tolerance on energies, eV (spec "Domain basis").
ENERGY_TOL = 1e-9

#: Refactor-invariance tolerance on forces, eV/A (spec "Domain basis").
FORCE_TOL = 1e-9


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def count(self, name: str, got: int, want: int) -> None:
        """Assert an integer golden exactly -- no tolerance applies to a count."""
        ok = got == want
        if not ok:
            self.failures.append(f"{name}: got {got}, want {want}")
        print(f"  {name:<30} {got:>24}  {'ok' if ok else 'FAILED'}")

    def close(self, name: str, got: float, want: float, tol: float, unit: str) -> None:
        """Assert a float golden within the spec's refactor-invariance band."""
        delta = abs(got - want)
        ok = delta <= tol
        if not ok:
            self.failures.append(
                f"{name}: got {got!r}, want {want!r} ({unit}); |delta|={delta:.3e} > {tol:g}"
            )
        print(f"  {name:<30} {got!r:>24}  |d|={delta:.3e} {unit:<9} {'ok' if ok else 'FAILED'}")

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (a shape, a `batch_size`, ...)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        print(f"  {name:<30} {'ok' if holds else 'FAILED':>24}")


def build_edges(pos: torch.Tensor, r_max: float) -> torch.Tensor:
    """Full bidirectional neighbour graph as `(E, 2)`, `[:,0]`=source, `[:,1]`=target.

    Non-periodic, so no `shifts` accompany it. `torch.cdist` over all ordered
    pairs is O(N^2) and fine at N=6 -- the production path uses
    `molix.md.PeriodicNeighborList`, which this file deliberately does not pull
    in: it would add a moving part between the goldens and the core under test.

    Args:
        pos: Positions `(N, 3)` in Angstrom.
        r_max: Cutoff radius in Angstrom.

    Returns:
        Edge index `(E, 2)`, self-pairs excluded.
    """
    dist = torch.cdist(pos, pos)
    mask = (dist < r_max) & ~torch.eye(pos.shape[0], dtype=torch.bool)
    return torch.nonzero(mask, as_tuple=False)


def build_model() -> MACEMatpes:
    """Small MACE-MatPES with seeded random weights, through the public kwargs.

    The unpassed spec defaults are part of the golden as much as the passed
    keywords are; at capture time they resolved to `num_polynomial_cutoff=5`,
    `scale=1.0`, `shift=0.0`, `max_hidden_l=1`, `radial_mlp=[64, 64, 64]`,
    `interaction='density'`, `readout='per_layer'`,
    `distance_transform='agnesi'`, `pair_repulsion='zbl'`,
    `conditioning='none'`, for a total of 35792 parameters.

    Returns:
        The model in `eval()` mode.
    """
    torch.manual_seed(0)
    return MACEMatpes(
        atomic_numbers=Z_TABLE,
        atomic_energies=ATOMIC_ENERGIES,
        r_max=R_MAX,
        num_bessel=8,
        l_max=2,
        num_features=16,
        num_interactions=2,
        correlation=3,
        mlp_dim=8,
        use_fallback=True,
    ).eval()


def energy_forces(
    core: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None],
        torch.Tensor,
    ],
    pos: torch.Tensor,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`run_nve._matpes_energy_forces`' closure shape, minus `Compiler`.

    Reproduced rather than imported: the script under `scripts/` is not a
    public API, and the point is to pin the *shape* of the call the re-point
    left behind -- leaf `requires_grad_`, energy under `torch.enable_grad()`,
    `autograd.grad` outside any compile region, and the positional order
    `(pos, Z, edge_index, batch, num_graphs, shifts)` with `shifts=None` for a
    non-periodic cell.

    Args:
        core: The public `energy_core`, per-graph energy `(B,)` out.
        pos: Positions `(N, 3)`; detached and re-leafed here.
        Z: Atomic numbers `(N,)`.
        edge_index: Edge index `(E, 2)`.
        batch: Graph membership `(N,)`.

    Returns:
        Per-graph energies `(B,)` in eV and forces `(N, 3)` in eV/A, detached.
    """
    leaf = pos.detach().requires_grad_(True)
    with torch.enable_grad():
        energy = core(leaf, Z, edge_index, batch, NUM_GRAPHS, None)
    forces = autograd_forces_from_energy(energy, leaf)
    return energy.detach(), forces.detach()


def check_energy_core(checker: Checker) -> None:
    """Section 1 -- the public core reproduces the pre-re-point goldens.

    Args:
        checker: Failure collector.
    """
    print("Section 1 - public energy_core vs pre-re-point goldens (fp64, CPU, fallback)")
    model = build_model()
    pos = torch.tensor(POSITIONS, dtype=config.ftype)
    Z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long)
    edge_index = build_edges(pos, R_MAX)
    batch = torch.zeros(Z.shape[0], dtype=torch.long)

    checker.count("n_atoms", int(pos.shape[0]), len(ATOMIC_NUMBERS))
    checker.count("n_edges", int(edge_index.shape[0]), GOLDEN_NUM_EDGES)

    energy, forces = energy_forces(model.energy_core, pos, Z, edge_index, batch)

    checker.truth(
        "energy.shape",
        tuple(energy.shape) == (NUM_GRAPHS,),
        f"energy_core returned {tuple(energy.shape)}, want ({NUM_GRAPHS},) per-graph energies",
    )
    checker.truth(
        "forces.shape",
        tuple(forces.shape) == (len(ATOMIC_NUMBERS), 3),
        f"got {tuple(forces.shape)}, want ({len(ATOMIC_NUMBERS)}, 3)",
    )
    checker.close("E_total", float(energy.sum()), GOLDEN_ENERGY, ENERGY_TOL, "eV")
    checker.close("max|F|", float(forces.abs().max()), GOLDEN_MAX_FORCE, FORCE_TOL, "eV/A")
    for index, (axis, want) in enumerate(zip("xyz", GOLDEN_FORCE_ATOM0)):
        checker.close(f"F[0].{axis}", float(forces[0][index]), want, FORCE_TOL, "eV/A")


def check_ensure_graphs(checker: Checker) -> None:
    """Section 2 -- `ensure_graphs` builds a schema-conforming `graphs` namespace.

    Args:
        checker: Failure collector.
    """
    print("\nSection 2 - ensure_graphs(num_graphs=3) graphs schema")
    batch = ensure_graphs(TensorDict(batch_size=[]), num_graphs=3)
    got = batch["graphs"].batch_size
    checker.truth(
        "graphs.batch_size",
        got == torch.Size([3]),
        f"got {got}, want torch.Size([3]) -- the CLAUDE.md graphs schema is batch_size=[B]; "
        "a consumer reading batch['graphs'].batch_size[0] raises IndexError on torch.Size([])",
    )


def main() -> int:
    """Run both sections and report a single PASS/FAIL verdict.

    A section that *raises* is a section that failed: the signatures this file
    pins (`energy_core`'s six positional parameters, `ensure_graphs`'
    `num_graphs` keyword) drift by `TypeError` rather than by a wrong number,
    so an escaping exception would otherwise kill the run before the
    `RESULT:` line the caller reads. The traceback is printed, then folded
    into the verdict.

    Returns:
        `0` on PASS, `1` on FAIL.
    """
    print("molnex regression - mace-subpackage-restructure-07-cleanup")
    print(f"torch={torch.__version__} device=cpu dtype={config.ftype} use_fallback=True\n")

    checker = Checker()
    for section in (check_energy_core, check_ensure_graphs):
        try:
            section(checker)
        except Exception:
            checker.failures.append(
                f"{section.__name__} raised:\n"
                f"{textwrap.indent(traceback.format_exc().rstrip(), '    ')}"
            )

    if checker.failures:
        print("\nThe 07-cleanup contract is broken:")
        for failure in checker.failures:
            print(f"  {failure}")
        print("RESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
