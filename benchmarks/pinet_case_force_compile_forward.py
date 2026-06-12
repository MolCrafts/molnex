from __future__ import annotations

import argparse
from pathlib import Path

from pinet_gpu_diagnostics import bench, clone_batch, setup_case, write_case_result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graphs", type=int, default=32)
    p.add_argument("--atoms", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    batch, model = setup_case(graphs=args.graphs, atoms=args.atoms, compute_forces=True)
    model.compile_energy(backend="inductor")
    model.eval()
    row = bench("force_compile_energy_forward_eval", lambda: model(clone_batch(batch), compute_forces=True), warmup=args.warmup, iters=args.iters)
    write_case_result(args.out, graphs=args.graphs, atoms=args.atoms, batch=batch, row=row)


if __name__ == "__main__":
    main()
