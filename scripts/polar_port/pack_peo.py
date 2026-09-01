"""Pack short PEO oligomers (diglyme) into a cubic PBC box.

molpy-only: SmilesReader + Packmol. Writes a system.pt the NPT runner can
load without molpy (Z, pos, mass, cell).

    PYTHONPATH=src python scripts/polar_port/pack_peo.py --out /tmp/peo_box.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from molpy import Element
from molpy.io.data.smiles import SmilesReader
from molpy.pack import InsideBoxConstraint, Packmol, Target

# Diglyme = CH3-O-(CH2CH2O)2-CH3, the smallest PEO-like oligomer with two
# ether repeats. SMILES is molrs-backed; hydrogens are explicit after embed.
_PEO_SMILES = "COCCOCCOC"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/peo_box.pt"))
    parser.add_argument("--n-mol", type=int, default=16)
    parser.add_argument("--box", type=float, default=18.0, help="cubic box edge in Å")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--packmol",
        type=Path,
        default=Path("/nobackup/proj/disk/teoroo/personal/jicli594/work/.x86_64/bin/packmol"),
    )
    args = parser.parse_args()

    frame = SmilesReader(_PEO_SMILES, add_hydrogens=True, optimize=True, seed=0).read()
    n_per = len(frame["atoms"]["x"])
    box = InsideBoxConstraint(
        length=np.array([args.box, args.box, args.box], dtype=np.float64),
        origin=np.zeros(3, dtype=np.float64),
    )
    packed = Packmol(executable=str(args.packmol))(
        [Target(frame, number=args.n_mol, constraint=box, name="peo")],
        max_steps=400,
        seed=args.seed,
        tolerance=2.0,
        pbc=[args.box, args.box, args.box],
    )
    atoms = packed["atoms"]
    symbols = [str(s) for s in atoms["element"]]
    pos = torch.tensor(np.stack([atoms["x"], atoms["y"], atoms["z"]], axis=1), dtype=torch.float64)
    elements = [Element(sym) for sym in symbols]
    z = torch.tensor([el.number for el in elements], dtype=torch.long)
    # Packmol's Frame mass column is 0 for C/O; take masses from molpy Element.
    mass = torch.tensor([float(el.mass) for el in elements], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * float(args.box)
    payload = {
        "Z": z,
        "pos": pos,
        "mass": mass,
        "cell": cell,
        "smiles": _PEO_SMILES,
        "n_mol": int(args.n_mol),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}: {int(z.numel())} atoms ({args.n_mol} × {n_per}), box {args.box:g} Å")


if __name__ == "__main__":
    main()
