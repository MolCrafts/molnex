"""Public-API regression for PesBundle (one graph, many confs).

Spec: ``universal-class-i-04-bundle`` (ac-008).

Hard-coded water Å geometries. Topology view has no pos; both confs share
``bonds.bond_index``.

    capture: PYTHONPATH=src python regressions/universal-class-i-04-bundle.py
    date: 2026-08-13
"""

from __future__ import annotations

import sys

import torch
from tensordict import TensorDict

from molix.data import PesBundle

_POS0 = torch.tensor(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9266, 0.0]],
    dtype=torch.float64,
)
_POS1 = torch.tensor(
    [[0.0, 0.0, 0.0], [1.02, 0.0, 0.0], [-0.25, 0.97, 0.0]],
    dtype=torch.float64,
)


def main() -> int:
    topology = TensorDict(
        {
            "atoms": TensorDict(
                {"Z": torch.tensor([8, 1, 1], dtype=torch.long)},
                batch_size=[3],
            ),
            "bonds": TensorDict(
                {"bond_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long)},
                batch_size=[],
            ),
        },
        batch_size=[],
    )
    bundle = PesBundle(topology, torch.stack([_POS0, _POS1]), group_id="water")
    topo = bundle.topology_batch()
    leaves = set(topo.keys(include_nested=True, leaves_only=True))
    if "pos" in leaves or ("atoms", "pos") in leaves:
        raise AssertionError("topology_batch must not contain pos")
    c0 = bundle.conformation_batch(0)
    c1 = bundle.conformation_batch(1)
    if not torch.equal(c0["bonds", "bond_index"], c1["bonds", "bond_index"]):
        raise AssertionError("bond_index must be shared")
    if not torch.allclose(c0["atoms", "pos"], _POS0, atol=1e-8):
        raise AssertionError("pos0 mismatch")
    if not torch.allclose(c1["atoms", "pos"], _POS1, atol=1e-8):
        raise AssertionError("pos1 mismatch")
    print("universal-class-i-04-bundle: water two-conf topology/pos OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
