"""DataLoader throughput profile for the real QM9 and RevMD17 datasets.

Builds each dataset's pipeline cache once (NeighborList edges), wraps it in an
``MmapDataset``, and runs :class:`~molix.profiler.DataLoaderProfiler` across a
few ``(batch_size, num_workers)`` settings — measuring per-batch wall time and
atoms/graphs throughput, i.e. how fast the data path can feed the trainer.

QM9 is many tiny rigid molecules (≈18 atoms); RevMD17 aspirin is one molecule's
MD trajectory (21 atoms, energy+forces targets) — different collate/target
shapes, so the comparison shows how target layout and molecule size affect
dataloader throughput.

Run (point --qm9-dir / --rmd17-dir at dirs holding qm9.tar.bz2 / rmd17_*.npz):
    python benchmarks/bench_dataloader_datasets.py \
        --qm9-dir /path/data --rmd17-dir /path/data --n 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from molix.data import NeighborList, Pipeline
from molix.datasets import QM9Source, RevMD17Source
from molix.profiler import DataLoaderProfiler

CONFIGS = [(32, 0), (32, 4), (128, 4)]


def _build_dataset(source, schema_tag: str, cache_dir: Path):
    """Run the NeighborList pipeline once and return an MmapDataset."""
    pipe = Pipeline(schema_tag).add(NeighborList(cutoff=5.0, pbc=False)).build()
    packed = pipe.cache(source, base_dir=cache_dir, fit_source=source)
    return packed.dataset(mmap=True)


def _profile(name: str, dataset, schema, n_batches: int) -> None:
    print(f"\n{'=' * 74}\n{name}  ({len(dataset)} samples)\n{'=' * 74}")
    for batch_size, num_workers in CONFIGS:
        prof = DataLoaderProfiler(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            target_schema=schema,
        )
        result = prof.run(dataset, n_batches=n_batches, n_warmup=5)
        print(f"\n--- batch_size={batch_size}  num_workers={num_workers} ---")
        result.print_report()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qm9-dir", default="/home/jicli594/work/pinet-training/data")
    ap.add_argument("--rmd17-dir", default="/home/jicli594/work/pinet-training/data")
    ap.add_argument("--cache-dir", default=None, help="cache base dir (default: tmp)")
    ap.add_argument("--n", type=int, default=4000, help="samples per dataset")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--only", choices=["qm9", "rmd17"], default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.cwd() / ".bench_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "qm9"):
        qm9 = QM9Source(args.qm9_dir, total=args.n)
        ds = _build_dataset(qm9, "qm9", cache_dir / "qm9")
        _profile("QM9", ds, QM9Source.TARGET_SCHEMA, args.n_batches)

    if args.only in (None, "rmd17"):
        rmd = RevMD17Source(args.rmd17_dir, molecule="aspirin", total=args.n)
        ds = _build_dataset(rmd, "rmd17_aspirin", cache_dir / "rmd17")
        _profile("RevMD17 (aspirin)", ds, RevMD17Source.TARGET_SCHEMA, args.n_batches)


if __name__ == "__main__":
    main()
