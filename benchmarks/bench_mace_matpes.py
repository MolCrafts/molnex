"""MACE-MatPES perf guard: the eager energy+force op vs the compiled op.

Times ``MaceMatpes(spec, wrt=("atoms.pos",))`` — the op every MD step runs —
eagerly and under ``Compiler(cuda_graphs=True)`` after the one eager forward
that spends the element gate (the ``run_nve.py`` production path). Random
weights (perf only, no physics); a periodic random-dense system builds real
``(E, 2)`` edges + imaged displacements through :class:`molix.md.NeighborList`.
The cuEquivariance kernels are pinned by :mod:`molrep.cueq`; there is no
fallback arm to compare against. Run on a GPU node:

    python benchmarks/bench_mace_matpes.py            # defaults: N=192, fp32
    python benchmarks/bench_mace_matpes.py --fp64 --steps 30
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from tensordict import TensorDict

from molix import config
from molix.compile import Compiler
from molix.md import NeighborList
from molzoo.mace import MaceMatpes, MACEMatpesSpec

_Z_TABLE = [1, 6, 7, 8, 14, 26]  # small table; dims below are the MatPES-class ones

#: ~0.045 atoms/A^3 — condensed-phase-ish density, for a realistic edge count.
DENSITY = 0.045
#: Model ``r_max`` *and* neighbour-list cutoff; the two must stay equal.
CUTOFF = 6.0


def _box_length(n_atoms: int) -> float:
    """Cubic box edge in Angstrom holding ``n_atoms`` at :data:`DENSITY`."""
    return float((n_atoms / DENSITY) ** (1.0 / 3.0))


def _min_n_atoms() -> int:
    """Smallest ``n_atoms`` whose box half-width strictly exceeds :data:`CUTOFF`.

    Minimum image requires ``cutoff <= box / 2``; below that
    :class:`molix.md.NeighborList` raises.
    """
    return math.floor(DENSITY * (2.0 * CUTOFF) ** 3) + 1


def _build_model() -> MaceMatpes:
    torch.manual_seed(0)
    spec = MACEMatpesSpec(
        atomic_numbers=_Z_TABLE,
        atomic_energies=[0.0] * len(_Z_TABLE),
        r_max=CUTOFF,
        num_bessel=10,
        num_polynomial_cutoff=5,
        l_max=3,
        num_features=128,
        max_hidden_l=1,
        num_interactions=2,
        correlation=3,
        mlp_dim=16,
    )
    return MaceMatpes(spec, wrt=("atoms.pos",)).eval()


def _system(n_atoms: int, device: torch.device) -> TensorDict:
    """A periodic random-dense system as the post-collate batch the op reads."""
    torch.manual_seed(1)
    box = _box_length(n_atoms)
    pos = torch.rand(n_atoms, 3, dtype=config.ftype, device=device) * box
    cell = torch.eye(3, dtype=config.ftype, device=device) * box
    Z = _Z_TABLE[0] + torch.zeros(n_atoms, dtype=torch.long, device=device)
    Z[::3] = _Z_TABLE[2]
    Z[::5] = _Z_TABLE[3]
    neighbors = NeighborList(cell=cell, cutoff=CUTOFF, positions=pos)
    pair_index = neighbors.pair_index
    pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + neighbors.shifts
    print(f"system: N={n_atoms} E={neighbors.num_pairs} (capacity {neighbors.capacity})")
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": Z,
                    "pos": pos,
                    "batch": torch.zeros(n_atoms, dtype=torch.long, device=device),
                },
                batch_size=[n_atoms],
            ),
            "pairs": TensorDict(
                {"pair_index": pair_index, "pair_diff": pair_diff, "shifts": neighbors.shifts},
                batch_size=[pair_index.shape[0]],
            ),
            "graphs": TensorDict({"cell": cell.unsqueeze(0)}, batch_size=[1]),
        },
        batch_size=[],
    )


def _time(fn, steps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-atoms",
        type=int,
        default=192,
        help=f"atoms in the cubic box; must exceed {_min_n_atoms() - 1} so the box "
        f"half-width stays above the {CUTOFF} A cutoff",
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fp64", action="store_true")
    args = ap.parse_args()

    # Fail here, not 40 frames deep inside NeighborList: minimum image
    # needs cutoff <= box/2, and the box is derived from --n-atoms at DENSITY.
    box = _box_length(args.n_atoms)
    if box / 2.0 <= CUTOFF:
        ap.error(
            f"--n-atoms {args.n_atoms} gives a {box:.2f} A box at {DENSITY} atoms/A^3, "
            f"whose half-width {box / 2.0:.2f} A does not exceed the {CUTOFF} A cutoff; "
            f"the minimum-image neighbour list would miss periodic images. "
            f"Use --n-atoms {_min_n_atoms()} or more."
        )

    config.set_precision("fp64" if args.fp64 else "fp32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} dtype={config.ftype}")

    frame = _system(args.n_atoms, device)
    model = _build_model().to(device)

    with torch.no_grad():
        eager = _time(lambda: model(frame)["atoms", "forces"], args.steps)
    print(f"eager energy+forces: {eager:9.3f} ms/step")

    if device.type == "cuda":
        # The op is the compile boundary; the eager forwards above spent the
        # element gate, so the trace never host-syncs.
        compiled = Compiler(cuda_graphs=True)(model)
        with torch.no_grad():
            fast = _time(lambda: compiled(frame)["atoms", "forces"], args.steps)
        print(f"compiled op (cuda graphs): {fast:9.3f} ms/step ({eager / fast:.2f}x eager)")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
