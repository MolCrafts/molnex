"""CPU unit test: molrep MACEBesselBasis + PolynomialCutoff vs official mace.

Loads the plain-torch molrep modules in isolation (stubbing `molix.config`)
so no cuequivariance / molcfg is needed, and checks bit-level agreement with
mace.modules.radial on float64.
"""
import importlib.util
import sys
import types

import torch

torch.set_default_dtype(torch.float64)

SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"

# --- stub molix.config so radial.py / cutoff.py import cleanly on CPU ---
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


radial = load("molrep_radial", f"{SRC}/molrep/embedding/radial.py")
cutoff = load("molrep_cutoff", f"{SRC}/molrep/embedding/cutoff.py")

from mace.modules.radial import BesselBasis as RefBessel  # noqa: E402
from mace.modules.radial import PolynomialCutoff as RefCutoff  # noqa: E402

R_MAX = 6.0
N = 8
r = torch.linspace(0.05, R_MAX + 1.0, 500, dtype=torch.float64)  # include r > r_max

# ---- Bessel ----  (BesselRBF in MACE-faithful mode: no norm, no eps, trainable)
ours_b = radial.BesselRBF(
    r_cut=R_MAX, num_radial=N, normalize=False, eps=0.0, trainable=True
)
ref_b = RefBessel(r_max=R_MAX, num_basis=N, trainable=True)
# init must already match (same formula); measure BEFORE any copy
init_diff = (ours_b.freqs.detach() - ref_b.bessel_weights.detach()).abs().max().item()
ob = ours_b(r)
rb = ref_b(r.unsqueeze(-1))
bessel_diff = (ob - rb).abs().max().item()

# ---- PolynomialCutoff ----
ours_c = cutoff.PolynomialCutoff(r_cut=R_MAX, exponent=6)
ref_c = RefCutoff(r_max=R_MAX, p=6)
oc = ours_c(r)
rc = ref_c(r)
cut_diff = (oc - rc).abs().max().item()

# Note: ref cutoff does NOT zero out r > r_max (no mask); ours masks.
# Check agreement on r < r_max region (physical edges only).
in_cut = r < R_MAX
cut_diff_incut = (oc[in_cut] - rc[in_cut]).abs().max().item()

print(f"bessel_weights init max|diff|   = {init_diff:.3e}")
print(f"Bessel    max|diff| (all r)     = {bessel_diff:.3e}")
print(f"PolyCutoff max|diff| (r<r_max)  = {cut_diff_incut:.3e}")
print(f"PolyCutoff max|diff| (all r)    = {cut_diff:.3e}  (ours masks r>=r_max, ref does not)")
print(f"ref cutoff at r=r_max+0.5       = {ref_c(torch.tensor([R_MAX+0.5]))[0].item():.3e}")
print(f"our cutoff at r=r_max+0.5       = {ours_c(torch.tensor([R_MAX+0.5]))[0].item():.3e}")

ok = bessel_diff < 1e-12 and cut_diff_incut < 1e-12 and init_diff < 1e-12
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
