"""Public-API force-pass parity scenario for chain step mace-subpackage-restructure-03.

Chain step 03 extracts the two batch-level force passes duplicated in
``molzoo/pinet/potential.py`` and ``molpot/derivation/modes/`` into the shared
``molpot.derivation.kernels`` (``grad_force_pass`` / ``func_force_pass``). The
spec declares that refactor **zero-behaviour-change**, so this script pins the
only thing that can prove it end to end: a fixed-seed ``PiNetPotential`` must
keep producing bit-identical energies and forces through both public force
methods after the rebinding.

Scenario (public API only): construct ``PiNetPotential(..., compute_forces=True,
method="func")`` and the same model with ``method="grad"``, feed one fixed
4-atom / 8-edge batch, and compare ``graphs.energy`` (eV) and ``atoms.forces``
(eV/Å) against the hard-coded literals below at ``atol=1e-12, rtol=0``. A third
``compute_forces=False`` instance (bit-identical ``state_dict``) supplies the
energies for a central-difference check of the forces — the domain leg, since
"the kernels still return ``F = -∂E/∂r``" is a physics claim, not a diff claim.

Goldens
-------
``GOLDEN_ENERGY`` / ``GOLDEN_FORCES`` were captured from the **pre-rebinding**
``PiNetPotential`` (its own local ``_pipeline_ef_func`` / ``_pipeline_ef_grad``
copies) — an in-repo self-oracle at the commit below, never re-imported at run
time. ``method="func"`` and ``method="grad"`` produced bit-identical values at
capture, so one table covers both and the script additionally asserts the two
methods agree exactly.

    capture script : scratchpad ``capture_03_goldens.py`` — same construction,
                     seed and batch as this file, printing ``.tolist()``
    capture command: PYTHONPATH=src:. python capture_03_goldens.py
    commit         : 03c0e85 (03c0e85b91f64cc09a1576139eb5576fe7cc4ab2)
    torch          : 2.12.1+cpu
    date           : 2026-08-08
    device / dtype : CPU, float64 (``config.set_precision("fp64")``)
    oracle         : molzoo.pinet.PiNetPotential at 03c0e85 — in-repo
                     self-oracle; no ASE / e3nn / mace-torch, no network,
                     no subprocess
    observed       : func-vs-grad deviation 0.0 (bit-identical); autograd-vs
                     -central-difference deviation 1.3e-11 eV/Å at h = 1e-5 Å

The batch is built inline (the same ``edge_diff = pos[dst] - pos[src]``,
``edge_dist = ‖edge_diff‖`` post-collate schema that ``tests/conftest.py``
produces) so the script stays standalone — ``PYTHONPATH=src`` is enough, the
repo root is not needed on the path.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-03-kernels.py
"""

from __future__ import annotations

import sys
from typing import Literal

import torch

from molix import config

config.set_precision("fp64")  # before any module construction

from tensordict import TensorDict  # noqa: E402

from molzoo.pinet import PiNetPotential  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixed system: 4 atoms (H, C, N, O), 8 directed edges, one graph.
# --------------------------------------------------------------------------- #
POS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.1, 0.1, 0.0],
        [0.3, 1.2, 0.2],
        [1.4, 1.1, -0.1],
    ],
    dtype=torch.float64,
)
Z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
EDGE_INDEX = torch.tensor(
    [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
    dtype=torch.long,
)
BATCH = torch.zeros(4, dtype=torch.long)

# --------------------------------------------------------------------------- #
# Hard-coded goldens (see module docstring for provenance).
# --------------------------------------------------------------------------- #
GOLDEN_ENERGY: list[float] = [0.8332936477597297]  # eV
GOLDEN_FORCES: list[list[float]] = [  # eV/Å
    [-0.005663236275122078, -0.005354983795618726, -0.0003581090284556928],
    [0.0035887033693461586, -0.004020850505374732, -0.0002506004328096707],
    [-0.003822775289657374, 0.005858775873778405, 0.0017552896792324322],
    [0.005897308195433293, 0.0035170584272150533, -0.0011465802179670686],
]

ATOL = 1e-12  # exact band: the refactor must not move a single bit
FD_ATOL = 1e-6  # numerical band for the central-difference domain check
FD_STEP = 1e-5  # Å


def make_batch(pos: torch.Tensor) -> TensorDict:
    """Post-collate batch for ``pos`` ``(4, 3)`` Å (atoms / edges / graphs)."""
    edge_diff = pos[EDGE_INDEX[:, 1]] - pos[EDGE_INDEX[:, 0]]
    edge_dist = edge_diff.norm(dim=-1).clamp(min=1e-6)
    num_atoms = torch.tensor([pos.shape[0]], dtype=torch.long)
    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=BATCH, batch_size=[pos.shape[0]]),
        edges=TensorDict(
            edge_index=EDGE_INDEX,
            edge_diff=edge_diff,
            edge_dist=edge_dist,
            batch_size=[EDGE_INDEX.shape[0]],
        ),
        graphs=TensorDict(num_atoms=num_atoms, batch_size=[1]),
        batch_size=[],
    )


def build(method: Literal["func", "grad"], *, compute_forces: bool = True) -> PiNetPotential:
    """Seed-0 PiNet potential — identical weights for every ``method``."""
    torch.manual_seed(0)
    return PiNetPotential(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=3,
        hidden_dim=16,
        compute_forces=compute_forces,
        method=method,
    ).eval()


class Checker:
    """Collect deviations so one run reports every failure, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def close(self, name: str, got: torch.Tensor, want: torch.Tensor, atol: float) -> None:
        if got.shape != want.shape:
            self.failures.append(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
            return
        deviation = (got - want).abs().max().item()
        if not deviation <= atol:
            self.failures.append(f"{name}: max|Δ| = {deviation:.3e} > atol {atol:.1e}")


def check_method(checker: Checker, method: Literal["func", "grad"]) -> TensorDict:
    """Run one public force method and compare energy + forces to the goldens."""
    result = build(method)(make_batch(POS.clone()))
    checker.close(
        f"{method}.energy",
        result["graphs", "energy"].detach(),
        torch.tensor(GOLDEN_ENERGY, dtype=torch.float64),
        ATOL,
    )
    checker.close(
        f"{method}.forces",
        result["atoms", "forces"].detach(),
        torch.tensor(GOLDEN_FORCES, dtype=torch.float64),
        ATOL,
    )
    return result


def check_finite_differences(checker: Checker, forces: torch.Tensor) -> None:
    """Domain leg: ``F = -dE/dr`` by central differences at ``h = 1e-5`` Å."""
    energy_model = build("func", compute_forces=False)
    numerical = torch.zeros_like(POS)
    with torch.no_grad():
        for atom in range(POS.shape[0]):
            for axis in range(POS.shape[1]):
                plus = POS.clone()
                plus[atom, axis] += FD_STEP
                minus = POS.clone()
                minus[atom, axis] -= FD_STEP
                e_plus = energy_model(make_batch(plus))["graphs", "energy"].sum()
                e_minus = energy_model(make_batch(minus))["graphs", "energy"].sum()
                numerical[atom, axis] = -(e_plus - e_minus) / (2.0 * FD_STEP)
    checker.close("finite_difference.forces", forces, numerical, FD_ATOL)


def main() -> int:
    """Verify both force methods against the goldens and against the physics."""
    checker = Checker()

    func_result = check_method(checker, "func")
    grad_result = check_method(checker, "grad")

    checker.close(
        "func_vs_grad.energy",
        func_result["graphs", "energy"].detach(),
        grad_result["graphs", "energy"].detach(),
        0.0,
    )
    checker.close(
        "func_vs_grad.forces",
        func_result["atoms", "forces"].detach(),
        grad_result["atoms", "forces"].detach(),
        0.0,
    )

    check_finite_differences(checker, func_result["atoms", "forces"].detach())

    if checker.failures:
        print("FAILED — PiNet force passes moved after the kernel rebinding:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
