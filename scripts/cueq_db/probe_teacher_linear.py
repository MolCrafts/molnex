"""Crack the contradiction: the teacher's cuet.Linear (l>0) double-backwards,
a freshly-built one does not. Extract a working teacher Linear, run the
double-backward probe on it vs a fresh clone, and dump the structural diff.

Run in NAIVE mode (no fused-ops LD_LIBRARY_PATH) to isolate the instantiation
difference (the teacher's 101/103 was measured in naive mode)."""

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = "/nobackup/proj/disk/teoroo/personal/jicli594/work/mace_models/OMOL-cueq.model"
teacher = torch.load(T, map_location=dev, weights_only=False).to(dev).double()


def db_probe(lin, din):
    """force-like double-backward: input=f(v); g=dE/dv; backprop g^2 -> weight grad."""
    v = torch.randn(8, 3, dtype=torch.float64, device=dev, requires_grad=True)
    x = torch.tanh(v.repeat(1, (din + 2) // 3)[:, :din].contiguous())  # (8, din), nonlinear in v
    E = lin(x).sum()
    g = torch.autograd.grad(E, v, create_graph=True)[0]
    for p in lin.parameters():
        p.grad = None
    g.pow(2).mean().backward()
    gl1 = sum(float(p.grad.abs().sum()) for p in lin.parameters() if p.grad is not None)
    return gl1


def describe(lin, tag):
    iin = getattr(lin, "irreps_in", None)
    iout = getattr(lin, "irreps_out", None)
    wsh = [tuple(p.shape) for p in lin.parameters()]
    print(f"\n[{tag}] {type(lin).__module__}.{type(lin).__name__}")
    print(f"   irreps_in={iin}  irreps_out={iout}")
    print(f"   weight shapes={wsh}")
    print(f"   vars keys={sorted(k for k in vars(lin) if not k.startswith('_'))}")
    # dig into the wrapped polynomial / method if present
    attrs = ("f", "module", "linear", "_linear", "tp", "transpose_in", "transpose_out", "layout")
    for attr in attrs:
        if hasattr(lin, attr):
            a = getattr(lin, attr)
            print(f"   .{attr} = {type(a).__name__ if hasattr(a, '__class__') else a}")


# find teacher cuet.Linear modules with l>0 in irreps_in
cands = []
for name, m in teacher.named_modules():
    if isinstance(m, cuet.Linear):
        iin = getattr(m, "irreps_in", None)
        has_l = iin is not None and any(ir.ir.l > 0 for ir in cue.Irreps(iin)) if iin else False
        cands.append((name, m, has_l, iin))
print(f"teacher has {len(cands)} cuet.Linear; l>0 ones:")
for name, m, has_l, iin in cands:
    if has_l:
        print(f"   {name}: {iin}")

# pick the first l>0 teacher Linear
tname, tlin, _, tiin = next((c for c in cands if c[2]), (None, None, None, None))
if tlin is None:
    print("no l>0 teacher Linear found")
    raise SystemExit
din = cue.Irreps(tiin).dim
describe(tlin, f"TEACHER {tname}")
print(f"   double-backward weight-grad L1 = {db_probe(tlin, din):.3e}")

# fresh clone with same irreps, our molrep style
fresh = cuet.Linear(
    cue.Irreps(tlin.irreps_in), cue.Irreps(tlin.irreps_out), layout=cue.ir_mul, dtype=torch.float64
).to(dev)
describe(fresh, "FRESH (molrep-style, layout=ir_mul)")
print(f"   double-backward weight-grad L1 = {db_probe(fresh, din):.3e}")
