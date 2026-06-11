"""EF training with fixed-shape (ghost-padded) batches so cudagraphs can run.

CUDA graphs need static shapes; rMD17 batches vary only in edge count (single
molecule → fixed atom count). Fix: pad every batch to a constant edge count
E_PAD with DEAD edges — real_atom→ghost, where one ghost atom sits far away
(>rc) so the cosine cutoff zeros the contribution exactly. The ghost's per-atom
energy is routed to a discard ("trash") graph; a wrapper slices model outputs
back to the real atom/graph counts, so loss/metrics are unaffected.

A fp64 correctness gate asserts padded == unpadded energy/forces before any
timing/training is trusted. Then trains eager vs compile_energy(cudagraphs)
from the same seed and compares E/F MAE/RMSE.

Run on GH200:  python benchmarks/train_pinet_ef_padded.py --epochs 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch._functorch.config as _fc

_fc.donated_buffer = False  # let cudagraphs survive the create_graph double backward

from tensordict import TensorDict  # noqa: E402

from molix.core.losses import energy_force_mse  # noqa: E402
from molix.core.metrics import MAE, RMSE  # noqa: E402
from molix.core.seed import seed_everything  # noqa: E402
from molix.data import AtomicDress, NeighborList, Pipeline  # noqa: E402
from molix.data.collate import collate_molecules  # noqa: E402
from molix.datasets import RevMD17Source  # noqa: E402
from molzoo.pinet import PiNet, PiNetPotential  # noqa: E402

RMD17_ROOT = "/home/jicli594/work/pinet-training/data"
GHOST_POS = 1.0e3  # far enough that every edge to it exceeds rc=6


def build_model() -> PiNetPotential:
    enc = PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=6.0,
        basis_type="gaussian",
        n_basis=10,
        pp_nodes=[16, 16, 16, 16],
        pi_nodes=[16],
        ii_nodes=[16, 16, 16, 16],
        depth=5,
        rank=3,
    )
    return PiNetPotential(encoder=enc, hidden_dim=16, compute_forces=True)


class PaddedWrap(torch.nn.Module):
    """Run inner on a ghost-padded batch; slice outputs back to real size."""

    def __init__(self, inner: PiNetPotential, n_real_atoms: int, n_real_graphs: int) -> None:
        super().__init__()
        self.inner = inner
        self.n_real_atoms = n_real_atoms
        self.n_real_graphs = n_real_graphs

    def forward(self, batch):
        out = self.inner(batch)
        return {
            "energy": out["energy"][: self.n_real_graphs],
            "forces": out["forces"][: self.n_real_atoms],
        }

    def compile_energy(self, **kw):
        self.inner.compile_energy(**kw)


def pad_batch(batch: TensorDict, e_pad: int) -> TensorDict:
    """Append one far ghost atom (trash graph) and pad edges to *e_pad* dead edges."""
    Z = batch["atoms", "Z"]
    pos = batch["atoms", "pos"]
    ab = batch["atoms", "batch"]
    n, g = Z.shape[0], int(ab.max().item()) + 1
    dev = pos.device

    ghost_idx = n
    Zp = torch.cat([Z, Z.new_ones(1)])
    posp = torch.cat([pos, pos.new_full((1, 3), GHOST_POS)])
    abp = torch.cat([ab, ab.new_full((1,), g)])  # ghost → trash graph g
    atoms = TensorDict({"Z": Zp, "pos": posp, "batch": abp}, batch_size=[n + 1])

    ei = batch["edges", "edge_index"]
    e = ei.shape[0]
    if e > e_pad:
        raise ValueError(f"batch has {e} edges > e_pad={e_pad}; raise e_pad")
    n_dead = e_pad - e
    dead = torch.tensor([[0, ghost_idx]], device=dev).repeat(n_dead, 1)  # real0→ghost
    eip = torch.cat([ei, dead], dim=0)
    edges = TensorDict({"edge_index": eip}, batch_size=[e_pad])

    num_atoms = batch["graphs", "num_atoms"]
    graphs_d = {"num_atoms": torch.cat([num_atoms, num_atoms.new_ones(1)])}
    for k in batch["graphs"].keys():
        if k != "num_atoms":
            v = batch["graphs", k]
            pad = v.new_zeros((1, *v.shape[1:])) if v.ndim else v.new_zeros(1)
            graphs_d[k] = torch.cat([v, pad])
    graphs = TensorDict(graphs_d, batch_size=[g + 1])
    return TensorDict({"atoms": atoms, "edges": edges, "graphs": graphs}, batch_size=[]).to(
        dev, non_blocking=True
    )


class PaddedDataModule:
    """Yields fixed-shape ghost-padded batches; drop_last keeps the shape constant."""

    def __init__(self, train_ds, val_ds, batch_size, e_pad, schema, device="cpu", seed=0):
        self.train_ds, self.val_ds = train_ds, val_ds
        self.bs, self.e_pad, self.schema, self.seed = batch_size, e_pad, schema, seed
        self.device = device
        self._epoch = 0

    def setup(self, stage="fit"):
        pass

    def on_epoch_start(self, epoch):
        self._epoch = epoch

    def _loader(self, ds, shuffle):
        n = len(ds)
        idx = list(range(n))
        if shuffle:
            g = torch.Generator().manual_seed(self.seed + self._epoch)
            idx = torch.randperm(n, generator=g).tolist()
        for s in range(0, n - self.bs + 1, self.bs):  # drop_last
            chunk = idx[s : s + self.bs]
            # move to device BEFORE padding so pad_batch builds all its tensors
            # (ghost atom, dead edges) on-device → fully on-device batch, no
            # reliance on the Trainer's per-batch move for this hand-built TD.
            batch = collate_molecules([ds[i] for i in chunk], self.schema).to(self.device)
            yield pad_batch(batch, self.e_pad)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)


def build_data(cache_dir: Path, n_total: int):
    src = RevMD17Source(RMD17_ROOT, molecule="aspirin", total=n_total)
    pipe = (
        Pipeline("rmd17-aspirin-ef")
        .add(AtomicDress(elements=[1, 6, 7, 8], target_key="energy", output_key="energy"))
        .add(NeighborList(cutoff=6.0, pbc=False))
        .build()
    )
    packed = pipe.cache(src, base_dir=str(cache_dir), fit_source=src)
    ds = packed.dataset(mmap=True)
    return ds.split(sizes=(n_total - 50, 50), seed=42)


def correctness_gate(train_ds, schema, e_pad, device):
    """fp64: padded+sliced model output must equal the unpadded output."""
    seed_everything(0)
    model = build_model().to(device).double()
    samples = [train_ds[i] for i in range(8)]
    raw = collate_molecules(samples, schema).to(device)
    raw["atoms", "pos"] = raw["atoms", "pos"].double()
    model.train()
    model(raw)  # materialise
    n_atoms, n_graphs = raw["atoms", "Z"].shape[0], 8
    padded = pad_batch(raw, e_pad)
    wrap = PaddedWrap(model, n_atoms, n_graphs)

    out_raw = model(raw.copy())
    out_pad = wrap(padded.copy())
    de = (out_raw["energy"] - out_pad["energy"]).abs().max().item()
    df = (out_raw["forces"] - out_pad["forces"]).abs().max().item()
    print(f"correctness gate (fp64): max|ΔE|={de:.2e}  max|ΔF|={df:.2e}")
    ok = de < 1e-8 and df < 1e-8
    print(f"  → padding exact: {ok}")
    return ok


def _metric_hooks():
    from molix.hooks.scalar import MetricsHook

    return [
        MetricsHook(
            metrics=[MAE(), RMSE()],
            pred_key=("predictions", "energy"),
            target_key=("graphs", "energy"),
            name_prefix="E_",
        ),
        MetricsHook(
            metrics=[MAE(), RMSE()],
            pred_key=("predictions", "forces"),
            target_key=("atoms", "forces"),
            name_prefix="F_",
        ),
    ]


def train_once(mode, train_ds, val_ds, e_pad, schema, epochs, lambda_f, device):
    from molix.core.hook import BaseHook
    from molix.core.trainer import Trainer

    class Rec(BaseHook):
        def __init__(self):
            self.rows = []

        def on_epoch_end(self, trainer, state):
            def g(p):
                v = state.get(p)
                return float(v) if v is not None else None

            self.rows.append(
                {
                    "epoch": int(state.get("epoch", 0)),
                    "val_E_MAE": g("eval/E_MAE"),
                    "val_E_RMSE": g("eval/E_RMSE"),
                    "val_F_MAE": g("eval/F_MAE"),
                    "val_F_RMSE": g("eval/F_RMSE"),
                }
            )

    seed_everything(0)
    inner = build_model().to(device)
    dm = PaddedDataModule(train_ds, val_ds, 32, e_pad, schema, device=device)
    warm = next(iter(dm.train_dataloader())).to(device)
    n_real_atoms = 32 * 21  # aspirin
    inner.train()
    inner(warm)  # materialise lazy params/buffers on the padded shape
    inner.to(device)  # sweep lazily-created buffers (cutoff r_cut etc.) to device
    if mode == "cudagraphs":
        inner.compile_energy(backend="cudagraphs")
    model = PaddedWrap(inner, n_real_atoms, 32)

    rec = Rec()
    trainer = Trainer(
        model=model,
        loss_fn=energy_force_mse(lambda_F=lambda_f, per_atom=True),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
        hooks=[*_metric_hooks(), rec],
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train(datamodule=dm, max_epochs=epochs)
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "mode": mode,
        "wall_s": time.perf_counter() - t0,
        "final": rec.rows[-1] if rec.rows else {},
        "curve": rec.rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-total", type=int, default=1000)
    ap.add_argument("--lambda-f", type=float, default=100.0)
    ap.add_argument("--e-pad", type=int, default=13000)
    ap.add_argument("--cache-dir", default="/dev/shm/pinet_ef_pad_cache")
    ap.add_argument("--out", default="pinet_ef_padded_compare.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device={device}")
    schema = RevMD17Source.TARGET_SCHEMA
    train_ds, val_ds = build_data(Path(args.cache_dir), args.n_total)

    if not correctness_gate(train_ds, schema, args.e_pad, device):
        raise SystemExit("padding corrupts physics — aborting before training")

    results = {}
    for mode in ("eager", "cudagraphs"):
        print(f"\n{'=' * 60}\nTRAIN {mode} (padded e_pad={args.e_pad})\n{'=' * 60}")
        results[mode] = train_once(
            mode, train_ds, val_ds, args.e_pad, schema, args.epochs, args.lambda_f, device
        )
        print(f"  {mode}: {results[mode]['final']}  ({results[mode]['wall_s']:.1f}s)")

    Path(args.out).write_text(json.dumps(results, indent=2))
    e, c = results["eager"]["final"], results["cudagraphs"]["final"]
    print(f"\n{'=' * 60}\nFINAL — eager vs cudagraphs(padded)\n{'=' * 60}")
    print(f"{'metric':<12}{'eager':>13}{'cudagraphs':>13}{'rel.diff':>11}")
    for k in ("val_E_MAE", "val_E_RMSE", "val_F_MAE", "val_F_RMSE"):
        if e.get(k) and c.get(k):
            print(f"{k:<12}{e[k]:>13.4e}{c[k]:>13.4e}{abs(c[k] - e[k]) / abs(e[k]):>11.1%}")
    sp = results["eager"]["wall_s"] / max(results["cudagraphs"]["wall_s"], 1e-9)
    print(
        f"\nwall: eager {results['eager']['wall_s']:.1f}s  "
        f"cudagraphs {results['cudagraphs']['wall_s']:.1f}s  ({sp:.1f}× speedup)"
    )


if __name__ == "__main__":
    main()
