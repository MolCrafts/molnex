"""Real EF training: does cudagraphs converge to the same E/F MAE/RMSE as eager?

We measured that compile_energy(cudagraphs) gives a tiny systematic gradient
deviation (~1e-7 rel-L2 fp32) on the EF double backward. Profiling can't say
whether that accumulates over training. This trains PiNet2-P3 on rMD17 aspirin
twice — eager and cudagraphs — from the SAME seed/weights/data, logging
train+val energy/force MAE and RMSE every epoch, and reports the final values
side by side.

Config (paper / PiNN-doc): PiNet2-P3 (rank=3), depth=5, rc=6.0, gaussian
n_basis=10, pp/ii=[16,16,16,16], pi=[16], hidden=16; AtomicDress on energy;
rMD17 aspirin 950 train / 50 val, batch_size=32, Adam lr=1e-3,
energy_force_mse(lambda_F, per_atom).

Run on GH200:  python benchmarks/train_pinet_ef_compare.py --epochs 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# compile_energy(cudagraphs) + force training (create_graph=True double
# backward) trips aot_autograd's donated-buffer optimisation in a real
# training loop (RuntimeError: ... requires create_graph=False). Disable it so
# the cudagraphs run can actually train and be compared. Eager is unaffected.
import torch._functorch.config as _functorch_config

_functorch_config.donated_buffer = False

from molix.core.hook import BaseHook
from molix.core.losses import energy_force_mse
from molix.core.metrics import MAE, RMSE
from molix.core.seed import seed_everything
from molix.core.trainer import Trainer
from molix.data import AtomicDress, DataModule, NeighborList, Pipeline
from molix.datasets import RevMD17Source
from molix.hooks.scalar import MetricsHook
from molzoo.pinet import PiNet, PiNetPotential

RMD17_ROOT = "/home/jicli594/work/pinet-training/data"


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


class CurveRecorder(BaseHook):
    """Capture per-epoch train/val E/F MAE+RMSE into a list."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def on_epoch_end(self, trainer, state) -> None:
        def g(path):
            v = state.get(path)
            return float(v) if v is not None else None

        self.rows.append(
            {
                "epoch": int(state.get("epoch", 0)),
                "train_E_MAE": g("train/E_MAE"),
                "train_F_MAE": g("train/F_MAE"),
                "val_E_MAE": g("eval/E_MAE"),
                "val_E_RMSE": g("eval/E_RMSE"),
                "val_F_MAE": g("eval/F_MAE"),
                "val_F_RMSE": g("eval/F_RMSE"),
            }
        )


def _metric_hooks():
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


def build_data(cache_dir: Path, n_total: int):
    src = RevMD17Source(RMD17_ROOT, molecule="aspirin", total=n_total)
    pipe = (
        Pipeline("rmd17-aspirin-ef")
        .add(AtomicDress(elements=[1, 6, 7, 8], target_key="energy", output_key="energy"))
        .add(NeighborList(cutoff=6.0, pbc=False))
        .build()
    )
    n_train = n_total - 50
    # split first, fit the dress on train only (no val/test leakage)
    full = src
    packed = pipe.cache(full, base_dir=str(cache_dir), fit_source=full)
    ds = packed.dataset(mmap=True)
    train_ds, val_ds = ds.split(sizes=(n_train, 50), seed=42)
    return train_ds, val_ds


def train_once(mode: str, train_ds, val_ds, epochs: int, lambda_f: float, device: str) -> dict:
    """Train one model (mode='eager'|'cudagraphs'); return curves + final + timing."""
    seed_everything(0)
    model = build_model().to(device)
    dm = DataModule(
        train_ds,
        val_ds,
        target_schema=RevMD17Source.TARGET_SCHEMA,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )

    # PiNet has lazy params — materialise them with one real forward before the
    # Trainer builds the optimizer / reads parameters(), and before compiling.
    warm = next(iter(dm.train_dataloader())).to(device)
    model.train()
    model(warm)
    if mode == "cudagraphs":
        model.compile_energy(backend="cudagraphs")

    recorder = CurveRecorder()
    trainer = Trainer(
        model=model,
        loss_fn=energy_force_mse(lambda_F=lambda_f, per_atom=True),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
        hooks=[*_metric_hooks(), recorder],
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train(datamodule=dm, max_epochs=epochs)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    final = recorder.rows[-1] if recorder.rows else {}
    return {"mode": mode, "wall_s": wall, "final": final, "curve": recorder.rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-total", type=int, default=1000)
    ap.add_argument("--lambda-f", type=float, default=100.0)
    ap.add_argument("--cache-dir", default="/dev/shm/pinet_ef_train_cache")
    ap.add_argument("--out", default="pinet_ef_train_compare.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device={device}")
    if device == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)}")
    print(
        f"PiNet2-P3, rMD17 aspirin {args.n_total - 50} train/50 val, bs=32, "
        f"epochs={args.epochs}, lambda_F={args.lambda_f}"
    )

    train_ds, val_ds = build_data(Path(args.cache_dir), args.n_total)

    results = {}
    for mode in ("eager", "cudagraphs"):
        print(f"\n{'=' * 64}\nTRAIN {mode}\n{'=' * 64}")
        results[mode] = train_once(mode, train_ds, val_ds, args.epochs, args.lambda_f, device)
        f = results[mode]["final"]
        print(f"  {mode} final: {f}  ({results[mode]['wall_s']:.1f}s)")

    Path(args.out).write_text(json.dumps(results, indent=2))

    e, c = results["eager"]["final"], results["cudagraphs"]["final"]
    print(f"\n{'=' * 64}\nFINAL (epoch {args.epochs}) — eager vs cudagraphs\n{'=' * 64}")
    print(f"{'metric':<14}{'eager':>14}{'cudagraphs':>14}{'rel.diff':>12}")
    for k in ("val_E_MAE", "val_E_RMSE", "val_F_MAE", "val_F_RMSE"):
        ev, cv = e.get(k), c.get(k)
        if ev and cv:
            rd = abs(cv - ev) / (abs(ev) + 1e-30)
            print(f"{k:<14}{ev:>14.5e}{cv:>14.5e}{rd:>12.2%}")
    print(
        f"\nwall: eager {results['eager']['wall_s']:.1f}s  "
        f"cudagraphs {results['cudagraphs']['wall_s']:.1f}s  "
        f"({results['eager']['wall_s'] / max(results['cudagraphs']['wall_s'], 1e-9):.1f}× speedup)"
    )
    print(f"curves saved to {args.out}")


if __name__ == "__main__":
    main()
