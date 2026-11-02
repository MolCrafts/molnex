"""Regression: the official MACE checkpoints load and evaluate in tree.

Spec: `mace-subpackage-restructure-05-checkpoint`.

Moved out of the unit suite (formerly
``tests/test_molzoo/test_mace/test_checkpoint.py``, skipped there whenever the
offline weights were absent, i.e. always in CI). Full-size foundation models
and an offline asset make this a golden regression run, not a unit test; the
remap logic itself stays unit-tested on tiny seeded models.

Pins:

1. **MatPES bit parity.** ``MACEPotential.from_checkpoint`` on
   ``MACE-matpes-r2scan-omat-ft`` places exactly the tensors of the
   hand-written "construct + ``MATPES_REMAP``" path, with the published
   ``128 / 1 / 16`` dimensions stated literally so ``from_checkpoint`` must
   derive them from ``hidden_irreps`` / ``MLP_irreps`` unaided.
2. **OMol load.** ``OMOL_REMAP`` fills all 104 ``nn.Parameter`` tensors of
   ``MACE-omol-0-extra-large-1024`` and leaves no checkpoint key unhoused.
3. **OMol surface.** Energy, ``max|F|`` and ``F[0]`` of a five-atom neutral
   singlet (``O H H C H``) match the captured goldens to 1e-6 eV / 1e-4 eV/A.
   This is a stability lock, not upstream parity: nothing here compares
   against ``mace-torch`` or ``e3nn``.

Capture:
    command: MOLNEX_MACE_WEIGHTS_DIR=<dir> OMP_NUM_THREADS=1 \\
             PYTHONPATH=src python regressions/mace-official-weights.py
    assets:  $MOLNEX_MACE_WEIGHTS_DIR/matpes_r2scan_config.json,
             matpes_r2scan_cueq_state.pt, omol_cueq_state.pt (sha256
             074c86154a1d709f..., dumped from OMOL-cueq.model, sha256
             8735a524e99fe80c..., by the convert_omol_to_cueq_state.py stored
             beside it)
    date:    2026-08-09 (goldens), moved here 2026-10-08
    torch:   2.12.1+cpu, cuequivariance 0.10.0
    device:  cpu, fp64, use_fallback=True, OMP_NUM_THREADS=1

Tolerances are the repo's numerical ones, not exact: the CPU reduction order
inside the cuEquivariance fallback follows ``torch.get_num_threads()``;
{1, 4, 16, 48} threads disagree by 3.2e-08 eV / 9.3e-08 eV/A. Run at a small
thread count: at 48 threads the five-atom forward is ~80 s instead of ~0.6 s.

Exits 2 when ``$MOLNEX_MACE_WEIGHTS_DIR`` or an asset is missing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

WEIGHTS_DIR_ENV = "MOLNEX_MACE_WEIGHTS_DIR"
MATPES_CONFIG = "matpes_r2scan_config.json"
MATPES_WEIGHTS = "matpes_r2scan_cueq_state.pt"
OMOL_WEIGHTS = "omol_cueq_state.pt"

OFFICIAL_NUM_FEATURES = 128
OFFICIAL_MAX_HIDDEN_L = 1
OFFICIAL_MLP_DIM = 16

OMOL_PARAMETER_COUNT = 104
OMOL_UNEXPECTED_KEYS: list[str] = []
OMOL_GOLDEN_ENERGY = -3122.454566006894
OMOL_GOLDEN_MAX_FORCE = 5.397722165018621
OMOL_GOLDEN_FIRST_FORCE = (-5.397722165018621, -3.9298849841895853, -3.931997122634094)
OMOL_ENERGY_TOL = 1e-6
OMOL_FORCE_TOL = 1e-4

CLUSTER_POS = [
    [0.00, 0.00, 0.00],
    [0.95, 0.00, 0.00],
    [-0.24, 0.93, 0.00],
    [0.00, 0.00, 1.40],
    [1.20, 1.10, 0.60],
]
CLUSTER_Z = [8, 1, 1, 6, 1]


def omol_cluster() -> TensorDict:
    """The five-atom cluster, every ordered pair an edge, neutral singlet."""
    n = len(CLUSTER_Z)
    pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j])
    edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    return TensorDict(
        atoms=TensorDict(
            Z=torch.tensor(CLUSTER_Z),
            pos=pos,
            batch=torch.zeros(n, dtype=torch.long),
            batch_size=[n],
        ),
        edges=TensorDict(
            edge_index=edge_index,
            edge_diff=edge_diff,
            edge_dist=edge_diff.norm(dim=-1),
            batch_size=[edge_index.shape[0]],
        ),
        graphs=TensorDict(
            num_atoms=torch.tensor([n]),
            total_charge=torch.zeros(1, dtype=torch.long),
            total_spin=torch.ones(1, dtype=torch.long),
            batch_size=[1],
        ),
        batch_size=[],
    )


def matpes_bit_parity(directory: Path) -> None:
    from molzoo.mace.checkpoint import MATPES_REMAP
    from molzoo.mace.potential import MACEPotential
    from molzoo.mace.spec import MACEMatpesSpec

    config_path = directory / MATPES_CONFIG
    weights_path = directory / MATPES_WEIGHTS
    official = json.loads(config_path.read_text())
    manual = MACEPotential(
        MACEMatpesSpec(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official["r_max"],
            num_bessel=official["num_bessel"],
            num_polynomial_cutoff=official["num_polynomial_cutoff"],
            l_max=official["max_ell"],
            num_features=OFFICIAL_NUM_FEATURES,
            max_hidden_l=OFFICIAL_MAX_HIDDEN_L,
            num_interactions=official["num_interactions"],
            correlation=official["correlation"],
            mlp_dim=OFFICIAL_MLP_DIM,
            radial_mlp=official["radial_MLP"],
            scale=official["atomic_inter_scale"],
            shift=official["atomic_inter_shift"],
        ),
        use_fallback=True,
    )
    MATPES_REMAP.load(manual, torch.load(weights_path, map_location="cpu", weights_only=True))
    built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

    reference = manual.state_dict()
    produced = built.state_dict()
    assert set(produced) == set(reference), set(produced) ^ set(reference)
    for name, want in reference.items():
        assert torch.equal(produced[name], want), name
    print(f"  matpes: from_checkpoint == manual remap on {len(reference)} tensors")


def omol(directory: Path) -> None:
    from molzoo.mace.checkpoint import OMOL_REMAP
    from molzoo.mace.potential import MACEPotential
    from molzoo.mace.spec import MACEOMolSpec

    state = torch.load(directory / OMOL_WEIGHTS, map_location="cpu", weights_only=True)
    spec = MACEOMolSpec(
        atomic_numbers=state["atomic_numbers"].tolist(),
        atomic_energies=state["atomic_energies_fn.atomic_energies"].flatten().tolist(),
        scale=float(state["scale_shift.scale"]),
        shift=float(state["scale_shift.shift"]),
        use_fallback=True,
    )
    model = MACEPotential(spec, use_fallback=True)
    OMOL_REMAP.load(model, state)
    model.eval()

    renamed = OMOL_REMAP.rename(state)
    parameters = {name for name, _ in model.named_parameters()}
    assert len(parameters) == OMOL_PARAMETER_COUNT, len(parameters)
    assert parameters <= set(renamed), parameters - set(renamed)
    unhoused = sorted(set(renamed) - set(model.state_dict()))
    assert unhoused == OMOL_UNEXPECTED_KEYS, unhoused
    print(f"  omol: {len(parameters)} learnables filled, no unhoused keys")

    result = model(omol_cluster())
    energy = result["graphs", "energy"].item()
    forces = result["atoms", "forces"]
    assert abs(energy - OMOL_GOLDEN_ENERGY) <= OMOL_ENERGY_TOL, energy
    max_force = forces.abs().max().item()
    assert abs(max_force - OMOL_GOLDEN_MAX_FORCE) <= OMOL_FORCE_TOL, max_force
    first = forces[0].tolist()
    assert all(
        abs(a - b) <= OMOL_FORCE_TOL for a, b in zip(first, OMOL_GOLDEN_FIRST_FORCE, strict=True)
    ), first
    print(f"  omol: E = {energy:.9f} eV, max|F| = {max_force:.6f} eV/A")


def main() -> int:
    from molix import config

    directory = os.environ.get(WEIGHTS_DIR_ENV)
    missing = [
        name
        for name in (MATPES_CONFIG, MATPES_WEIGHTS, OMOL_WEIGHTS)
        if directory is None or not (Path(directory) / name).is_file()
    ]
    if missing:
        print(f"SKIP: set {WEIGHTS_DIR_ENV} to a directory holding {missing}", file=sys.stderr)
        return 2

    # cuEquivariance freezes its working precision at construction.
    config.set_precision("fp64")
    matpes_bit_parity(Path(directory))
    omol(Path(directory))
    print("mace-official-weights: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
