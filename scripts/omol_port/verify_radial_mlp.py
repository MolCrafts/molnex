"""CPU test: molrep.interaction.RadialMLP vs official mace RadialMLP (direct copy)."""
import importlib.util
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"

molix = types.ModuleType("molix")
molix.config = types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"] = molix
sys.modules["molix.config"] = molix.config


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


radial = load("mr_radial", f"{SRC}/molrep/interaction/radial.py")
from mace.modules.radial import RadialMLP as MaceRMLP  # noqa: E402

ch = [2056, 128, 128, 128, 4096]  # OMOL conv_tp_weights shape
ours = radial.RadialMLP(ch).double()
ref = MaceRMLP(ch).double()
ours.load_state_dict(ref.state_dict())
ours.eval()
ref.eval()

x = torch.randn(20, 2056, dtype=torch.float64)
with torch.no_grad():
    d = (ours(x) - ref(x)).abs().max().item()
print(f"RadialMLP max|diff| = {d:.3e}")
print("RESULT:", "PASS" if d < 1e-12 else "FAIL")
sys.exit(0 if d < 1e-12 else 1)
