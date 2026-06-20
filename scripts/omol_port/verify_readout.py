import importlib.util, sys, types, torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
SRC="/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"
molix=types.ModuleType("molix"); molix.config=types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"]=molix; sys.modules["molix.config"]=molix.config
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
pkg=types.ModuleType("mro"); pkg.__path__=[f"{SRC}/molrep/readout"]; sys.modules["mro"]=pkg
sc=load("mro.scalar", f"{SRC}/molrep/readout/scalar.py")
cueq=torch.load("/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/OMOL-cueq.model",map_location="cpu",weights_only=False).double()
r=cueq.readouts[0]
ours=sc.NonLinearBiasReadout(irreps_in="1024x0e", mlp_dim=16).double()
miss,unexp=ours.load_state_dict(r.state_dict(),strict=False)
mp=[k for k in miss if k.endswith((".weight",".bias"))]
N=7; x=torch.randn(N,1024,dtype=torch.float64)
with torch.no_grad():
    yo=ours(x); yr=r(x, torch.zeros(N,dtype=torch.long))
print("miss",mp,"out",tuple(yo.shape),tuple(yr.shape))
print(f"NonLinearBiasReadout max|diff| = {(yo-yr).abs().max():.3e}")
