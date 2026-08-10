#!/usr/bin/env python
"""Regression: molecule-centered bond-harmonic multi-conf energy residuals."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molix.core.losses.molecular import center_by_group  # noqa: E402
from molix.core.metrics import MoleculeCenteredRMSE  # noqa: E402
from molpot.potentials.bonds import BondHarmonic  # noqa: E402


def main() -> None:
    # k=2, r0=1, r in {1, 1.5, 2} → E = {0, 0.25, 1.0}
    pot = BondHarmonic(
        k=torch.tensor([2.0], dtype=torch.float64),
        r0=torch.tensor([1.0], dtype=torch.float64),
    )
    rs = [1.0, 1.5, 2.0]
    energies = []
    for r in rs:
        pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
        e = pot(
            pos=pos,
            bond_index=torch.tensor([[0], [1]], dtype=torch.long),
            bond_types=torch.tensor([0], dtype=torch.long),
        )
        energies.append(float(e))
    assert energies == [0.0, 0.25, 1.0] or all(
        abs(a - b) < 1e-12 for a, b in zip(energies, [0.0, 0.25, 1.0], strict=True)
    )
    e_t = torch.tensor(energies, dtype=torch.float64)
    groups = torch.tensor([0, 0, 0])
    centered = center_by_group(e_t, groups)
    # mean = (0+0.25+1)/3 = 1.25/3
    mean = e_t.mean()
    assert torch.allclose(centered, e_t - mean)
    m = MoleculeCenteredRMSE()
    m.update(e_t, e_t, groups)
    assert float(m.compute()) == 0.0
    print("mm-param-val-07-mm-energy: OK")


if __name__ == "__main__":
    main()
