"""PiNet2 EF benchmark — paper config (JCTC 2024 PiNN suite), real rMD17 aspirin.

Architecture = PiNN-documented PiNet defaults:
  depth=5, rc=6.0, gaussian basis n_basis=10,
  pp_nodes=[16,16,16,16], pi_nodes=[16], ii_nodes=[16,16,16,16], out/hidden=16.
Model variants (paper): PiNet (rank=1, invariant) / PiNet2-P3 (rank=3,
+vector) / PiNet2-P5 (rank=5, +tensor).
Data: real rMD17 aspirin (21 atoms), NeighborList(rc=6.0), batch_size=32
(the experimental setup: ~30-step epoch over the ~950-config rMD17 split).
Regime: EF *training* — F=-dE/dpos, create_graph=True → double backward.

Three tables:
  (1) wall ms/step:   variant × precision × compiler
  (2) accuracy:       energy/force MAE vs the fp64 eager reference, per
                      variant × precision  (feeds the quantization→MD study)
  (3) correctness:    cudagraphs-vs-eager gradient cosine + rel-L2

Run on GH200:  python benchmarks/bench_pinet_ef_full.py --batch-size 32
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from molix.data import NeighborList, Pipeline
from molix.data.collate import collate_molecules
from molix.datasets import RevMD17Source
from molix.profiler._utils import _fmt_table
from molzoo.pinet import PiNet, PiNetPotential

RMD17_ROOT = "/home/jicli594/work/pinet-training/data"
VARIANTS = {"PiNet(r1)": 1, "PiNet2-P3(r3)": 3, "PiNet2-P5(r5)": 5}
PRECISIONS = ["fp32", "fp64", "bf16-mixed", "fp16-mixed"]
COMPILERS = ["eager", "inductor", "cudagraphs", "reduce-overhead"]
_KW = {
    "inductor": {"backend": "inductor"},
    "cudagraphs": {"backend": "cudagraphs"},
    "reduce-overhead": {"mode": "reduce-overhead"},
}


def _build(rank: int) -> PiNetPotential:
    enc = PiNet(
        atom_types=[1, 6, 7, 8],  # rMD17 aspirin: H, C, N, O
        r_max=6.0,
        basis_type="gaussian",
        n_basis=10,
        pp_nodes=[16, 16, 16, 16],
        pi_nodes=[16],
        ii_nodes=[16, 16, 16, 16],
        depth=5,
        rank=rank,
    )
    return PiNetPotential(encoder=enc, hidden_dim=16, compute_forces=True).cuda()


def _ef_loss(out):
    return out["energy"].float().pow(2).mean() + out["forces"].float().pow(2).mean()


class _Autocast(torch.nn.Module):
    def __init__(self, m, dtype):
        super().__init__()
        self.module = m
        self.dtype = dtype

    def forward(self, b):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(b)


def _inner(m):
    return m.module if isinstance(m, _Autocast) else m


def _wrap(model, precision):
    if precision == "fp32":
        return model
    if precision == "fp64":
        return model.double()
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return _Autocast(model, dtype)


def build_aspirin_batch(batch_size: int, cache_dir: Path):
    """One real rMD17-aspirin batch of *batch_size* configs on cuda (fp32 pos)."""
    src = RevMD17Source(RMD17_ROOT, molecule="aspirin", total=max(batch_size * 4, 256))
    pipe = Pipeline("rmd17-aspirin-rc6").add(NeighborList(cutoff=6.0, pbc=False)).build()
    packed = pipe.cache(src, base_dir=str(cache_dir), fit_source=src)
    ds = packed.dataset(mmap=True)
    samples = [ds[i] for i in range(batch_size)]
    batch = collate_molecules(samples, RevMD17Source.TARGET_SCHEMA).to("cuda")
    return batch


def _cast_batch(batch, dtype):
    b = batch.copy()
    b["atoms", "pos"] = batch["atoms", "pos"].to(dtype).detach().clone()
    return b


def _run(model, batch):
    inner = _inner(model)
    inner.train()
    inner.zero_grad(set_to_none=True)
    b = batch.copy()
    b["atoms", "pos"] = batch["atoms", "pos"].detach().clone()
    out = model(b)
    _ef_loss(out).backward()
    grads = {
        n: p.grad.detach().double().clone()
        for n, p in inner.named_parameters()
        if p.grad is not None
    }
    return out["energy"].detach().double(), out["forces"].detach().double(), grads


def _time(model, batch, n_warmup=12, n_steps=40) -> float:
    for _ in range(n_warmup):
        _run(model, batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _run(model, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_steps


def _grad_diff(ga, gb):
    names = [n for n in ga if n in gb]
    a = torch.cat([ga[n].flatten() for n in names])
    b = torch.cat([gb[n].flatten() for n in names])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((b - a).norm() / (a.norm() + 1e-30)).item()
    return cos, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", default="/dev/shm/pinet_aspirin_cache")
    args = ap.parse_args()
    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
    print(f"rMD17 aspirin, batch_size={args.batch_size}, rc=6.0, depth=5, gaussian n_basis=10")

    fp32_batch = build_aspirin_batch(args.batch_size, Path(args.cache_dir))
    n_atoms = fp32_batch["atoms", "Z"].shape[0]
    n_edges = fp32_batch["edges", "edge_index"].shape[0]
    print(f"batch: {n_atoms} atoms, {n_edges} edges\n")

    speed, accuracy, correct = [], [], []

    for vname, rank in VARIANTS.items():
        # fp64 eager reference (energy/forces) for the accuracy table
        ref_model = _wrap(_build(rank), "fp64")
        ref_batch = _cast_batch(fp32_batch, torch.float64)
        ref_e, ref_f, _ = _run(ref_model, ref_batch)
        ref_state = copy.deepcopy(_inner(ref_model).state_dict())  # fp64 weights
        del ref_model
        torch.cuda.empty_cache()

        for precision in PRECISIONS:
            dtype = torch.float64 if precision == "fp64" else torch.float32
            batch = _cast_batch(fp32_batch, dtype)
            row = {"variant": vname, "precision": precision}
            eager_grads = None

            for comp in COMPILERS:
                tag = f"{vname} {precision:11s} {comp}"
                try:
                    model = _build(rank)
                    _run(
                        _wrap(model, "fp32"), _cast_batch(fp32_batch, torch.float32)
                    )  # materialise
                    model.load_state_dict({k: v.float() for k, v in ref_state.items()})
                    wrapped = _wrap(model, precision)
                    if comp != "eager":
                        _inner(wrapped).compile_energy(**_KW[comp])
                    ms = _time(wrapped, batch)
                    row[comp] = f"{ms:.2f}"
                    print(f"  [ok]   {tag}  {ms:.2f} ms")
                    if comp == "eager":
                        e, f, eager_grads = _run(wrapped, batch)
                        f_rel = ((f - ref_f).norm() / (ref_f.norm() + 1e-30)).item()
                        accuracy.append(
                            {
                                "variant": vname,
                                "precision": precision,
                                "E_MAE": f"{(e - ref_e).abs().mean().item():.2e}",
                                "F_MAE": f"{(f - ref_f).abs().mean().item():.2e}",
                                "F_relL2": f"{f_rel:.2e}",
                            }
                        )
                    if comp == "cudagraphs" and eager_grads is not None:
                        for _ in range(3):
                            _run(wrapped, batch)
                        _, _, cg = _run(wrapped, batch)
                        cos, rel = _grad_diff(eager_grads, cg)
                        correct.append(
                            {
                                "variant": vname,
                                "precision": precision,
                                "cosine": f"{cos:.6f}",
                                "grad_relL2": f"{rel:.2e}",
                            }
                        )
                except Exception as e:  # noqa: BLE001
                    row[comp] = "FAIL"
                    print(f"  [FAIL] {tag}  {type(e).__name__}: {str(e).splitlines()[0][:48]}")
                finally:
                    torch.compiler.reset()
                    torch.cuda.empty_cache()
            speed.append(row)

    print("\n=== (1) EF-train wall ms/step  (variant × precision × compiler) ===")
    print(_fmt_table(speed, ["variant", "precision", *COMPILERS], col_width=9))
    print("\n=== (2) accuracy vs fp64 eager (energy/force MAE, eager fwd) ===")
    print(_fmt_table(accuracy, ["variant", "precision", "E_MAE", "F_MAE", "F_relL2"], col_width=9))
    print("\n=== (3) cudagraphs vs eager gradient correctness ===")
    print(_fmt_table(correct, ["variant", "precision", "cosine", "grad_relL2"], col_width=9))


if __name__ == "__main__":
    main()
