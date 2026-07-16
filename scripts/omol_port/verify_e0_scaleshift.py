"""CPU unit test: molpot AtomicReferenceEnergy + GlobalRescale vs official mace."""
import importlib.util
import sys
import types

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"

molix = types.ModuleType("molix")
config_mod = types.SimpleNamespace(ftype=torch.float64)
molix.config = config_mod
sys.modules["molix"] = molix
sys.modules["molix.config"] = config_mod


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


energy = load("molpot_energy", f"{SRC}/molpot/heads/energy.py")
rescale = load("molpot_rescale", f"{SRC}/molpot/heads/rescale.py")

from mace.modules.blocks import AtomicEnergiesBlock, ScaleShiftBlock  # noqa: E402

# ---- E0 / AtomicReferenceEnergy ----
z_table = [1, 6, 8, 7]                       # H, C, O, N (element-table order)
e0 = torch.tensor([-13.6, -1029.0, -2042.0, -1485.0], dtype=torch.float64)
Z = torch.tensor([1, 8, 6, 6, 7, 1, 8])      # some atoms

ours_e0 = energy.AtomicReferenceEnergy(atomic_energies=e0, atomic_numbers=z_table)
oe = ours_e0(Z)

ref_block = AtomicEnergiesBlock(e0)          # indexed by element-table position
# build one-hot of Z over the element table order
z_index = {z: i for i, z in enumerate(z_table)}
idx = torch.tensor([z_index[int(z)] for z in Z])
one_hot = F.one_hot(idx, num_classes=len(z_table)).to(torch.float64)
re = ref_block(one_hot).squeeze(-1)  # mace returns (N,1); ours (N,)

e0_diff = (oe - re).abs().max().item()

# ---- ScaleShift / GlobalRescale (single head) ----
scale, shift = 0.731, -1.234
x = torch.randn(7, dtype=torch.float64) * 10
ours_ss = rescale.GlobalRescale(scale=scale, shift=shift)
ox = ours_ss(x)
ref_ss = ScaleShiftBlock(scale=scale, shift=shift)
head = torch.zeros(7, dtype=torch.long)
rx = ref_ss(x, head)
ss_diff = (ox - rx).abs().max().item()

print(f"AtomicReferenceEnergy (E0) max|diff| = {e0_diff:.3e}")
print(f"GlobalRescale (scale/shift) max|diff| = {ss_diff:.3e}")
ok = e0_diff < 1e-12 and ss_diff < 1e-12
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
