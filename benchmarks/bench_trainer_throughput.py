"""Trainer-throughput benchmark: model fwd/bwd/opt + end-to-end Trainer loop.

Measures the per-step compute that dominates training throughput, on CPU and
GPU, using the public profiler workload contract plus a real ``Trainer.train()``
loop over mock data for the end-to-end steps/sec (data + hooks + Step protocol).

Run:
    python benchmarks/bench_trainer_throughput.py --device cpu
    python benchmarks/bench_trainer_throughput.py --device cuda

The model is a MACE encoder (representative O(E) equivariant GNN); the loss is
``node_features.sum()`` so no readout head is needed — the measured cost is the
encoder forward + backward + optimizer step, i.e. the trainer's hot path.
"""

from __future__ import annotations

import argparse
import time

import torch
from _profile_workload import profile_training

from molix.profiler import MockBatch
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.spec import MACEMatpesSpec

SIZES = {
    "small": {"n_atoms": 64, "n_pairs": 256, "n_graphs": 4},
    "medium": {"n_atoms": 512, "n_pairs": 4096, "n_graphs": 16},
    "large": {"n_atoms": 2048, "n_pairs": 16384, "n_graphs": 64},
}


def build_model() -> MACEEncoder:
    """A small MACE encoder (7 elements, 16 features, 2 interactions)."""
    return MACEEncoder(
        MACEMatpesSpec(
            atomic_numbers=list(range(1, 8)),
            atomic_energies=[0.0] * 7,
            r_max=5.0,
            num_bessel=8,
            l_max=1,
            num_features=16,
            max_hidden_l=1,
            num_interactions=2,
            correlation=2,
        ),
        dtype=torch.float32,
    )


def _loss(output, batch=None):
    """Scalar loss on the encoder output — exercises the full backward graph."""
    return output["atoms", "node_features"].sum()


def bench_module(device: str, n_steps: int) -> None:
    """Profile a complete MACE training workload over each input size."""
    print(f"\n{'=' * 78}\nProfiler — MACE fwd+bwd+opt on {device}\n{'=' * 78}")
    for name, dims in SIZES.items():
        model = build_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        factory = MockBatch(
            n_atoms=dims["n_atoms"],
            n_pairs=dims["n_pairs"],
            n_graphs=dims["n_graphs"],
            atomic_numbers=7,
            device=device,
            seed=0,
        )
        warmup = 5 if device == "cpu" else 10
        result = profile_training(
            model, _loss, opt, factory, warmup=warmup, steps=n_steps, device=device
        )
        print(f"\n--- size={name} {dims} ---")
        print(result.model_dump_json(indent=2))


def bench_trainer_loop(device: str, n_steps: int) -> None:
    """End-to-end Trainer.train() steps/sec including data + hooks."""
    from molix.core.trainer import Trainer
    from molix.hooks.scalar import StepSpeedHook

    print(f"\n{'=' * 78}\nEnd-to-end Trainer loop on {device}\n{'=' * 78}")
    dims = SIZES["medium"]
    factory = MockBatch(
        n_atoms=dims["n_atoms"],
        n_pairs=dims["n_pairs"],
        n_graphs=dims["n_graphs"],
        atomic_numbers=7,
        device=device,
        seed=0,
    )

    class _MockDataModule:
        """Yields *n_steps* pre-generated batches per epoch (one epoch)."""

        def __init__(self, n: int):
            self._batches = [factory() for _ in range(n)]

        def setup(self, stage: str = "fit") -> None:
            pass

        def on_epoch_start(self, epoch: int) -> None:
            pass

        def train_dataloader(self):
            return iter(self._batches)

        def val_dataloader(self):
            return iter(self._batches[:1])

    model = build_model().to(device)
    trainer = Trainer(
        model=model,
        loss_fn=lambda pred, batch: pred["atoms", "node_features"].sum(),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-3),
        hooks=[StepSpeedHook()],
        device=device,
    )
    dm = _MockDataModule(n_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = trainer.train(datamodule=dm, max_epochs=1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    steps = state.global_step
    print(
        f"\nend-to-end: {steps} steps in {elapsed:.3f}s "
        f"= {steps / elapsed:.2f} steps/s "
        f"({dims['n_atoms'] * steps / elapsed:.0f} atoms/s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    print(f"torch {torch.__version__} | device={dev}")
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"threads={torch.get_num_threads()}")

    bench_module(dev, args.steps)
    bench_trainer_loop(dev, args.steps)


if __name__ == "__main__":
    main()
