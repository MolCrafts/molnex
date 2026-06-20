"""Verify mace e3nn OMOL vs its cueq-converted twin give identical E/F on CPU.

This establishes the cueq model (cuet primitives, ir_mul — the same stack molnex
uses) as a faithful reference for the molnex port.
"""
import os
import sys
import tempfile

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

PATH = "/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/MACE-omol-0-extra-large-1024.model"

from ase import Atoms  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402
from mace.cli.convert_e3nn_cueq import run as to_cueq  # noqa: E402

# small molecule (water); OMOL wants charge/spin metadata
atoms = Atoms(
    "H2O",
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
)
atoms.info["charge"] = 0
atoms.info["spin"] = 1

# --- e3nn reference ---
e3nn_calc = MACECalculator(model_paths=PATH, device="cpu", default_dtype="float64", head="omol")
atoms.calc = e3nn_calc
e_e3nn = atoms.get_potential_energy()
f_e3nn = atoms.get_forces()

# --- convert to cueq, save, build calc ---
src = torch.load(PATH, map_location="cpu", weights_only=False).double()
cueq = to_cueq(src, device="cpu", return_model=True)
tmp = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
torch.save(cueq, tmp.name)
cueq_calc = MACECalculator(model_paths=tmp.name, device="cpu", default_dtype="float64", head="omol")
atoms.calc = cueq_calc
e_cueq = atoms.get_potential_energy()
f_cueq = atoms.get_forces()
os.unlink(tmp.name)

de = abs(e_e3nn - e_cueq)
df = np.abs(f_e3nn - f_cueq).max()
print(f"E(e3nn) = {e_e3nn:.8f} eV")
print(f"E(cueq) = {e_cueq:.8f} eV")
print(f"|dE|        = {de:.3e} eV")
print(f"max|dF|     = {df:.3e} eV/Ang")
ok = de < 1e-5 and df < 1e-5
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
