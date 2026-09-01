"""MacePolar vs official MACE-POLAR-1-M water E/F (hard-coded goldens).

Requires the 2026-08-30 Polar dump and an out-of-tree
``convert_e3nn_cueq`` state (e3nn Linear weights that stay ``(N,)``).
Skip when either file is absent. No mace/e3nn/ASE imported here.

    PYTHONPATH=src python regressions/mace-polar-official-water.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from molix import config
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar

config.set_precision("fp64")

_DUMP = Path(os.environ.get("MOLNEX_POLAR_DUMP", "/tmp/mace-polar/MACE-POLAR-1-M.model"))
_CUEQ = Path(os.environ.get("MOLNEX_POLAR_CUEQ_STATE", "/tmp/polar_cueq_state.pt"))

# Official e3nn PolarMACE + MACECalculator fp64, isolated water, Q=0, M=1.
# Captured 2026-08-30 from /tmp/mace-polar/MACE-POLAR-1-M.model.
_ENERGY = -2079.8642433944215
_FORCES = [
    [1.68658860e-04, 3.67987548e-07, 1.84531545e-01],
    [-8.31143851e-05, 2.01580453e-01, -9.23147259e-02],
    [-8.55444747e-05, -2.01580821e-01, -9.22168195e-02],
]


def main() -> None:
    if not _DUMP.is_file() or not _CUEQ.is_file():
        print("skip: Polar dump or cueq overlay state not present")
        return
    pos = torch.tensor(
        [
            [0.000000, 0.000000, 0.117300],
            [0.000000, 0.757200, -0.469200],
            [0.000000, -0.757200, -0.469200],
        ],
        dtype=torch.float64,
    )
    z = torch.tensor([8, 1, 1], dtype=torch.long)
    n = 3
    pairs = [[i, j] for i in range(n) for j in range(n) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    batch = torch.zeros(n, dtype=torch.long)
    model = MacePolar.load(_DUMP, format="official", use_fallback=True, compute_forces=True)
    cueq_sd = torch.load(_CUEQ, map_location="cpu", weights_only=True)
    CheckpointRemap(polar_keys, on_unexpected="return").load(model, cueq_sd)
    model.eval()
    out = model.energy_forces(
        pos,
        z,
        edge_index,
        batch,
        1,
        edge_diff=edge_diff,
        total_charge=torch.zeros(1, dtype=torch.float64),
        total_spin=torch.ones(1, dtype=torch.float64),
    )
    energy = float(out["energy"].detach().reshape(-1)[0])
    forces = out["forces"].detach()
    gold_f = torch.tensor(_FORCES, dtype=torch.float64)
    d_e = abs(energy - _ENERGY)
    d_f = float((forces.cpu() - gold_f).abs().max())
    assert d_e < 1.0e-6, f"dE={d_e} eV"
    assert d_f < 1.0e-6, f"max|dF|={d_f} eV/Å"
    print(f"mace-polar-official-water: ok  dE={d_e:.3e} eV  max|dF|={d_f:.3e} eV/Å")


if __name__ == "__main__":
    main()
