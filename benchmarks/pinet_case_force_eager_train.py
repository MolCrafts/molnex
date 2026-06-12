from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pinet_gpu_diagnostics import bench, clone_batch, energy_force_loss, setup_case, write_case_result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graphs", type=int, default=32)
    p.add_argument("--atoms", type=int, default=16)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    batch, model = setup_case(graphs=args.graphs, atoms=args.atoms, compute_forces=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        out = model(clone_batch(batch), compute_forces=True)
        energy_force_loss(out, batch).backward()
        opt.step()

    row = bench("force_eager_train_step", step, warmup=args.warmup, iters=args.iters)
    write_case_result(args.out, graphs=args.graphs, atoms=args.atoms, batch=batch, row=row)


if __name__ == "__main__":
    main()
