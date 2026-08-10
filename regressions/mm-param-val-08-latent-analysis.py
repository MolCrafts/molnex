#!/usr/bin/env python
"""Regression: hard-coded purity goldens 1.0 / 0.0 / None + artifact keys."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molrep.analysis import (  # noqa: E402
    AtomLatentTable,
    LatentAnalysisArtifacts,
    NearestNeighbourTypePurity,
)


def main() -> None:
    sep = AtomLatentTable(
        torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]]),
        ["m"] * 4,
        torch.tensor([0, 0, 1, 1]),
    )
    assert NearestNeighbourTypePurity(k=1).score(sep).mean_purity == 1.0
    alt = AtomLatentTable(
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        ["m"] * 4,
        torch.tensor([0, 1, 0, 1]),
    )
    assert NearestNeighbourTypePurity(k=1).score(alt).mean_purity == 0.0
    unl = AtomLatentTable(torch.zeros(2, 2), ["m", "m"], torch.tensor([-1, -1]))
    assert NearestNeighbourTypePurity(k=1).score(unl).mean_purity is None
    with tempfile.TemporaryDirectory() as td:
        m = LatentAnalysisArtifacts(td).write(sep)
        assert m["nn_type_purity_mean"] == 1.0
        assert m["n_atoms"] == 4
    print("mm-param-val-08-latent-analysis: OK")


if __name__ == "__main__":
    main()
