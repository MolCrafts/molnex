"""CPU test: molrep.JointFeatureEmbedding vs OMOL's real joint_embedding submodule."""
import importlib.util
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

SRC = "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/src"
PATH = "/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/MACE-omol-0-extra-large-1024.model"

molix = types.ModuleType("molix")
molix.config = types.SimpleNamespace(ftype=torch.float64)
sys.modules["molix"] = molix
sys.modules["molix.config"] = molix.config


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so pydantic can resolve module globals
    spec.loader.exec_module(mod)
    return mod


node = load("molrep_node", f"{SRC}/molrep/embedding/node.py")

# load OMOL, grab the real joint_embedding
src = torch.load(PATH, map_location="cpu", weights_only=False).double()
je = src.joint_embedding
print("OMOL joint_embedding.specs:", je.specs)
print("embedders:", {k: type(v).__name__ for k, v in je.embedders.items()})

# build matching specs for ours
specs = []
for name, spec in je.specs.items():
    specs.append(
        node.JointFeatureSpec(
            name=name,
            kind=spec["type"],
            emb_dim=spec["emb_dim"],
            num_classes=spec.get("num_classes"),
            per=spec.get("per", "graph"),
            offset=spec.get("offset", 0),
        )
    )
out_dim = je.project[0].weight.shape[0]
ours = node.JointFeatureEmbedding(feature_specs=specs, out_dim=out_dim).double()

# copy weights (direct, matching key names)
ours.load_state_dict(je.state_dict())
ours.eval()

# build inputs: 3 graphs, 7 atoms
batch = torch.tensor([0, 0, 1, 1, 1, 2, 2])
B = 3
feats = {}
for name, spec in je.specs.items():
    if spec["type"] == "categorical":
        # value range pre-offset: choose valid post-offset indices
        off = spec.get("offset", 0)
        nc = spec["num_classes"]
        feats[name] = torch.randint(-off, nc - off, (B,))
    else:
        feats[name] = torch.randn(B)

with torch.no_grad():
    y_ours = ours(batch, **feats)
    # OMOL forward signature: (batch, features_dict)
    y_ref = je(batch, {k: v for k, v in feats.items()})

diff = (y_ours - y_ref).abs().max().item()
print("out shape", tuple(y_ours.shape))
print(f"JointFeatureEmbedding max|diff| = {diff:.3e}")
print("RESULT:", "PASS" if diff < 1e-10 else "FAIL")
sys.exit(0 if diff < 1e-10 else 1)
