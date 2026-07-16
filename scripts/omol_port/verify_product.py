import importlib.util, sys, types, torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
SRC="/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"
molix=types.ModuleType("molix"); molix.config=types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"]=molix; sys.modules["molix.config"]=molix.config
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
pkg=types.ModuleType("mi"); pkg.__path__=[f"{SRC}/molrep/interaction"]; sys.modules["mi"]=pkg
pb=load("mi.product_basis", f"{SRC}/molrep/interaction/product_basis.py")
cueq=torch.load("/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/OMOL-cueq.model",map_location="cpu",weights_only=False).double()
for li,p in enumerate(cueq.products):
    sc_m=p.symmetric_contractions
    iin=str(sc_m.irreps_in); iout=str(sc_m.irreps_out)
    ours=pb.EquivariantProductBasis(node_feats_irreps=iin,target_irreps=iout,correlation=sc_m.contraction_degree,num_elements=1,use_sc=True).double()
    miss,unexp=ours.load_state_dict(p.state_dict(),strict=False)
    mp=[k for k in miss if k.endswith(".weight")]
    import cuequivariance as cue
    N=6; irdim=cue.Irreps("O3",iin).dim; mul=[mi.mul for mi in cue.Irreps("O3",iin)][0]
    nf=torch.randn(N, irdim//mul, mul, dtype=torch.float64)  # (N, ir_dim, mul)
    scv=torch.randn(N, cue.Irreps("O3",iout).dim, dtype=torch.float64)
    na=torch.zeros(N,83,dtype=torch.float64); na[torch.arange(N),torch.randint(0,83,(N,))]=1.0
    with torch.no_grad():
        yo=ours(nf,scv,na); yr=p(nf,scv,na)
    print(f"product{li}: deg={sc_m.contraction_degree} miss{mp} max|diff|={(yo-yr).abs().max():.2e}")
