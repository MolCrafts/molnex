"""Public-API regression for Universal Class-I ChemEncoder.

Spec: `universal-class-i-01-encoder` (ac-007).

Hard-coded goldens only — no third-party oracle. Pins:

1. Same covalent topology, two conformations (zeros vs random ``pos``) →
   five :class:`~molrep.chem.features.ChemEmbeddings` leaves
   ``allclose(atol=1e-7)``; max abs diff is ``0.0``
2. Reverse ``bonds.atomi`` / ``bonds.atomj`` → ``bonds.chem_features``
   ``allclose(rtol=1e-5, atol=1e-6)``
3. Feature shapes ``(4, 8)``, ``(3, 8)``, ``(2, 8)``, ``(1, 8)``, ``(1, 8)``
4. ``forward`` writes no energy / force keys

Provenance
----------
    topology         : tests/test_molrep/test_chem/conftest.py ``mini_batch``
                       O/H/H/C, 3 bonds / 2 angles / 1 proper / 1 improper
    capture command  : PYTHONPATH=src python regressions/universal-class-i-01-encoder.py
    date             : 2026-08-13
    torch            : 2.12.1+cpu (python 3.14.5)
    device/precision : cpu, float32 (default ``config.ftype``)
    oracle           : none — identity / shape / key-absence goldens
"""

from __future__ import annotations

import sys

import torch
from tensordict import TensorDict

_SHAPE_ATOM = (4, 8)
_SHAPE_BOND = (3, 8)
_SHAPE_ANGLE = (2, 8)
_SHAPE_PROPER = (1, 8)
_SHAPE_IMPROPER = (1, 8)
_MAX_ABS_POS_INVARIANCE = 0.0


def _mini_batch(pos: torch.Tensor) -> TensorDict:
    """Four-atom O/H/H/C valence fixture (same topology as chem ``mini_batch``)."""
    z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
    n = z.shape[0]
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": z,
                    "pos": pos,
                    "batch": torch.zeros(n, dtype=torch.long),
                },
                batch_size=[n],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 0, 0], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                    "bond_types": torch.tensor([1, 1, 1], dtype=torch.long),
                },
                batch_size=[3],
            ),
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([1, 1], dtype=torch.long),
                    "atomj": torch.tensor([0, 0], dtype=torch.long),
                    "atomk": torch.tensor([2, 3], dtype=torch.long),
                },
                batch_size=[2],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([1], dtype=torch.long),
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([3], dtype=torch.long),
                    "atoml": torch.tensor([2], dtype=torch.long),
                },
                batch_size=[1],
            ),
            "impropers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),  # center
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def main() -> int:
    from molrep.chem import ChemEmbeddings, ChemEncoder

    torch.manual_seed(0)

    enc = ChemEncoder(
        atom_dim=8,
        bond_dim=8,
        angle_dim=8,
        proper_dim=8,
        improper_dim=8,
        n_blocks=5,
    )
    enc.eval()

    batch_zero = _mini_batch(torch.zeros(4, 3))
    batch_rand = _mini_batch(torch.randn(4, 3))

    emb_zero = enc.compose(batch_zero)
    emb_rand = enc.compose(batch_rand)
    assert isinstance(emb_zero, ChemEmbeddings)
    assert isinstance(emb_rand, ChemEmbeddings)

    leaves = (
        ("atom", emb_zero.atom, emb_rand.atom),
        ("bond", emb_zero.bond, emb_rand.bond),
        ("angle", emb_zero.angle, emb_rand.angle),
        ("proper", emb_zero.proper, emb_rand.proper),
        ("improper", emb_zero.improper, emb_rand.improper),
    )
    for name, left, right in leaves:
        max_abs = (left - right).abs().max().item()
        assert max_abs == _MAX_ABS_POS_INVARIANCE, f"{name} max abs {max_abs}"
        assert torch.allclose(left, right, atol=1e-7), name

    assert emb_zero.atom.shape == _SHAPE_ATOM
    assert emb_zero.bond.shape == _SHAPE_BOND
    assert emb_zero.angle.shape == _SHAPE_ANGLE
    assert emb_zero.proper.shape == _SHAPE_PROPER
    assert emb_zero.improper.shape == _SHAPE_IMPROPER

    written = enc(batch_zero.clone())
    assert written["atoms", "chem_features"].shape == _SHAPE_ATOM
    assert written["bonds", "chem_features"].shape == _SHAPE_BOND
    assert written["angles", "chem_features"].shape == _SHAPE_ANGLE
    assert written["propers", "chem_features"].shape == _SHAPE_PROPER
    assert written["impropers", "chem_features"].shape == _SHAPE_IMPROPER

    nested = {str(key) for key in written.keys(include_nested=True)}
    assert not any("energy" in key for key in nested)
    assert not any("force" in key for key in nested)

    reversed_bonds = batch_zero.clone()
    reversed_bonds["bonds", "atomi"] = batch_zero["bonds", "atomj"].clone()
    reversed_bonds["bonds", "atomj"] = batch_zero["bonds", "atomi"].clone()
    reversed_out = enc(reversed_bonds)
    assert torch.allclose(
        written["bonds", "chem_features"],
        reversed_out["bonds", "chem_features"],
        rtol=1e-5,
        atol=1e-6,
    )

    print("universal-class-i-01-encoder: all hard-coded goldens OK")
    print(
        "  atom {0}, bond {1}, angle {2}, proper {3}, improper {4}".format(
            tuple(emb_zero.atom.shape),
            tuple(emb_zero.bond.shape),
            tuple(emb_zero.angle.shape),
            tuple(emb_zero.proper.shape),
            tuple(emb_zero.improper.shape),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
