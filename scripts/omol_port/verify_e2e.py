import importlib.util, sys, types, torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"
molix = types.ModuleType("molix"); molix.config = types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"] = molix; sys.modules["molix.config"] = molix.config


def pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m


for n, p in [("molrep", f"{SRC}/molrep"), ("molrep.embedding", f"{SRC}/molrep/embedding"),
             ("molrep.interaction", f"{SRC}/molrep/interaction"), ("molrep.readout", f"{SRC}/molrep/readout"),
             ("molpot", f"{SRC}/molpot"), ("molpot.heads", f"{SRC}/molpot/heads"),
             ("molzoo", f"{SRC}/molzoo")]:
    pkg(n, p)


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m


for name, sub in [("molrep.embedding.angular", "molrep/embedding/angular.py"),
                  ("molrep.embedding.cutoff", "molrep/embedding/cutoff.py"),
                  ("molrep.embedding.node", "molrep/embedding/node.py"),
                  ("molrep.embedding.radial", "molrep/embedding/radial.py"),
                  ("molrep.interaction.gate", "molrep/interaction/gate.py"),
                  ("molrep.interaction.radial", "molrep/interaction/radial.py"),
                  ("molrep.interaction.product_basis", "molrep/interaction/product_basis.py"),
                  ("molrep.interaction.residual", "molrep/interaction/residual.py"),
                  ("molrep.readout.scalar", "molrep/readout/scalar.py"),
                  ("molpot.heads.energy", "molpot/heads/energy.py"),
                  ("molpot.heads.rescale", "molpot/heads/rescale.py")]:
    load(name, f"{SRC}/{sub}")
mo = load("molzoo.mace_omol", f"{SRC}/molzoo/mace_omol.py")
MACEOMol, load_omol_state_dict = mo.MACEOMol, mo.load_omol_state_dict

cueq = torch.load("/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/OMOL-cueq.model",
                  map_location="cpu", weights_only=False).double()
ztab = cueq.atomic_numbers.tolist()
ae = cueq.atomic_energies_fn.atomic_energies.flatten().double()
mdl = MACEOMol(atomic_numbers=ztab, atomic_energies=ae,
               scale=float(cueq.scale_shift.scale), shift=float(cueq.scale_shift.shift)).double()
miss, unexp = load_omol_state_dict(mdl, cueq.state_dict())
print("missing learnable:", miss[:15])

pos = torch.tensor([[0., 0., 0.], [1.0, 0., 0.], [0., 1.1, 0.], [0.5, 0.5, 0.9]], dtype=torch.float64)
Z = torch.tensor([6, 1, 8, 7]); N = 4
s_, d_ = [], []
for i in range(N):
    for j in range(N):
        if i != j and (pos[i] - pos[j]).norm() < 6.0:
            s_.append(i); d_.append(j)
ei = torch.tensor([s_, d_]); batch = torch.zeros(N, dtype=torch.long)
tc = torch.tensor([0.]); ts = torch.tensor([1.])

out = mdl.energy_forces(pos, Z, ei, batch, tc, ts, compute_forces=True)
E_mine = out["energy"].item(); F_mine = out["forces"]

zi = torch.searchsorted(cueq.atomic_numbers, Z)
na = torch.zeros(N, len(ztab), dtype=torch.float64); na[torch.arange(N), zi] = 1.0
posg = pos.clone().requires_grad_(True)
data = {"positions": posg, "node_attrs": na, "edge_index": ei,
        "shifts": torch.zeros(ei.shape[1], 3, dtype=torch.float64),
        "unit_shifts": torch.zeros(ei.shape[1], 3, dtype=torch.float64),
        "batch": batch, "ptr": torch.tensor([0, N]), "cell": torch.zeros(3, 3, dtype=torch.float64),
        "head": torch.tensor([0]), "total_charge": tc, "total_spin": ts}
res = cueq(data, compute_force=True, training=False)
E_ref = res["energy"].item(); F_ref = res["forces"]
print(f"E_mine={E_mine:.8f}  E_ref={E_ref:.8f}  |dE|={abs(E_mine - E_ref):.3e}")
print(f"max|dF|={(F_mine - F_ref).abs().max().item():.3e}")
print("RESULT:", "PASS" if abs(E_mine - E_ref) < 1e-5 and (F_mine - F_ref).abs().max().item() < 1e-4 and not miss else "FAIL")
