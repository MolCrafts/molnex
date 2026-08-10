"""Public-API regression for continuous chemical perception encoder.

Spec: `learnable-classical-ff-04-chem-encoder`.

Hard-coded goldens only — no third-party oracle. Pins:

1. AtomChemEmbedding Z → (N, D_a); reuses JointEmbedding
2. Bond / angle / proper reverse symmetries; improper outer-swap (center fixed)
3. ChemEncoder write_batch keys + ChemEmbeddings counts
4. molzoo ChemPerception recipe forwards without energy keys
5. No molpot imports under molrep.chem / molzoo.chem

Provenance
----------
    capture command  : PYTHONPATH=src python regressions/learnable-classical-ff-04-chem-encoder.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch
from tensordict import TensorDict


def main() -> int:
    from molrep.chem import (
        AngleContext,
        AtomChemEmbedding,
        BondChemEmbedding,
        ChemEmbeddings,
        ChemEncoder,
        ImproperContext,
        ProperContext,
    )
    from molrep.embedding.node import JointEmbedding
    from molzoo.chem import ChemPerception, ChemPerceptionSpec

    torch.manual_seed(0)

    # --- 1. Atom embedding ---
    atom_emb = AtomChemEmbedding(atom_dim=8, num_elements=20)
    assert isinstance(atom_emb.joint, JointEmbedding)
    z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
    h = atom_emb(z)
    assert h.shape == (4, 8)

    # --- 2. Bond symmetry ---
    bond_emb = BondChemEmbedding(atom_dim=8, bond_dim=6)
    atomi = torch.tensor([0, 0, 1], dtype=torch.long)
    atomj = torch.tensor([1, 2, 3], dtype=torch.long)
    assert torch.allclose(
        bond_emb(h, atomi, atomj),
        bond_emb(h, atomj, atomi),
        rtol=1e-5,
        atol=1e-6,
    )

    # --- 3. Angle reverse ---
    ang = AngleContext(atom_dim=8, angle_dim=6)
    ai = torch.tensor([1], dtype=torch.long)
    aj = torch.tensor([0], dtype=torch.long)
    ak = torch.tensor([2], dtype=torch.long)
    assert torch.allclose(ang(h, ai, aj, ak), ang(h, ak, aj, ai), rtol=1e-5, atol=1e-6)

    # --- 4. Proper reverse ---
    prop = ProperContext(atom_dim=8, proper_dim=6)
    pi = torch.tensor([1], dtype=torch.long)
    pj = torch.tensor([0], dtype=torch.long)
    pk = torch.tensor([3], dtype=torch.long)
    pl = torch.tensor([2], dtype=torch.long)
    assert torch.allclose(
        prop(h, pi, pj, pk, pl),
        prop(h, pl, pk, pj, pi),
        rtol=1e-5,
        atol=1e-6,
    )

    # --- 5. Improper outer swap (center fixed at atomi) ---
    imp = ImproperContext(atom_dim=8, improper_dim=6)
    center = torch.tensor([0], dtype=torch.long)
    j = torch.tensor([1], dtype=torch.long)
    k = torch.tensor([2], dtype=torch.long)
    l = torch.tensor([3], dtype=torch.long)
    base = imp(h, center, j, k, l)
    assert torch.allclose(base, imp(h, center, k, j, l), rtol=1e-5, atol=1e-6)
    assert torch.allclose(base, imp(h, center, j, l, k), rtol=1e-5, atol=1e-6)

    # --- 6. ChemEncoder I/O ---
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": z,
                    "batch": torch.zeros(4, dtype=torch.long),
                },
                batch_size=[4],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 0, 0], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
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
                    "atomi": pi,
                    "atomj": pj,
                    "atomk": pk,
                    "atoml": pl,
                },
                batch_size=[1],
            ),
            "impropers": TensorDict(
                {
                    "atomi": center,
                    "atomj": j,
                    "atomk": k,
                    "atoml": l,
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    enc = ChemEncoder(
        atom_dim=8,
        bond_dim=6,
        angle_dim=6,
        proper_dim=6,
        improper_dim=6,
        num_elements=20,
    )
    out = enc(batch)
    emb = enc.embeddings(out)
    assert isinstance(emb, ChemEmbeddings)
    assert emb.atom.shape == (4, 8)
    assert emb.bond.shape == (3, 6)
    assert emb.angle.shape == (2, 6)
    assert emb.proper.shape == (1, 6)
    assert emb.improper.shape == (1, 6)

    # --- 7. molzoo recipe ---
    spec = ChemPerceptionSpec(
        atom_dim=8,
        bond_dim=6,
        angle_dim=6,
        proper_dim=6,
        improper_dim=6,
        num_elements=20,
    )
    model = ChemPerception(spec=spec)
    out2 = model(batch)
    assert out2["atoms", "chem_features"].shape == (4, 8)
    nested = {str(k) for k in out2.keys(include_nested=True)}
    assert not any("energy" in k for k in nested)

    # --- 8. Import boundary ---
    root = Path(__file__).resolve().parents[1] / "src"
    for rel in ("molrep/chem", "molzoo/chem"):
        for path in (root / rel).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molpot"), path
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("molpot"), path

    print("learnable-classical-ff-04-chem-encoder: all hard-coded goldens OK")
    print(f"  atom features {tuple(emb.atom.shape)}, bond {tuple(emb.bond.shape)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — standalone regression script
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
