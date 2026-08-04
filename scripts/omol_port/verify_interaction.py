"""CPU test: molrep.ResidualInteraction vs cueq OMOL interactions[0]."""
import importlib.util
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"
CUEQ = "/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/OMOL-cueq.model"

molix = types.ModuleType("molix")
molix.config = types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"] = molix
sys.modules["molix.config"] = molix.config


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# package shim so residual.py's `from .gate import` / `from .radial import` resolve
pkg = types.ModuleType("molrep_inter")
pkg.__path__ = [f"{SRC}/molrep/interaction"]
sys.modules["molrep_inter"] = pkg
load("molrep_inter.gate", f"{SRC}/molrep/interaction/gate.py")
load("molrep_inter.radial", f"{SRC}/molrep/interaction/radial.py")
res = load("molrep_inter.residual", f"{SRC}/molrep/interaction/residual.py")

it = torch.load(CUEQ, map_location="cpu", weights_only=False).double().interactions[0]

ours = res.ResidualInteraction(
    node_attrs_irreps=str(it.node_attrs_irreps),
    node_feats_irreps=str(it.node_feats_irreps),
    edge_attrs_irreps=str(it.edge_attrs_irreps),
    edge_feats_irreps=str(it.edge_feats_irreps),
    edge_irreps=str(it.edge_irreps),
    target_irreps=str(it.target_irreps),
    hidden_irreps=str(it.hidden_irreps),
    radial_mlp=list(it.radial_MLP),
).double()

missing, unexpected = ours.load_state_dict(it.state_dict(), strict=False)
miss_params = [
    k for k in missing if k.endswith((".weight", ".bias")) or k in ("alpha", "beta")
]
print("missing learnable keys:", miss_params[:20])

ours.eval()
N, E = 7, 20
n_attrs = torch.zeros(N, 83, dtype=torch.float64)
n_attrs[torch.arange(N), torch.randint(0, 83, (N,))] = 1.0
n_feats = torch.randn(N, it.node_feats_irreps.dim, dtype=torch.float64)
e_attrs = torch.randn(E, it.edge_attrs_irreps.dim, dtype=torch.float64)
e_feats = torch.randn(E, 8, dtype=torch.float64)
edge_index = torch.randint(0, N, (2, E))
cutoff = torch.rand(E, 1, dtype=torch.float64)

with torch.no_grad():
    m_ours, sc_ours = ours(n_attrs, n_feats, e_attrs, e_feats, edge_index, cutoff)
    m_ref, sc_ref = it(
        node_attrs=n_attrs, node_feats=n_feats, edge_attrs=e_attrs,
        edge_feats=e_feats, edge_index=edge_index, cutoff=cutoff, first_layer=True,
    )

dm = (m_ours - m_ref).abs().max().item()
ds = (sc_ours - sc_ref).abs().max().item()
print("message shape", tuple(m_ours.shape), "ref", tuple(m_ref.shape))
print(f"message max|diff| = {dm:.3e}")
print(f"skip    max|diff| = {ds:.3e}")
ok = dm < 1e-9 and ds < 1e-9 and not miss_params
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
