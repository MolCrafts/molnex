"""Trainer-throughput benchmark: model fwd/bwd/opt + end-to-end Trainer loop.

Measures the per-step compute that dominates training throughput, on CPU and
GPU, using ``molix.profiler.ModuleProfiler`` (CUDA-event timing, peak memory,
real GPU-active time via ``torch.profiler``) plus a real ``Trainer.train()``
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

from molix.profiler import MockBatch, ModuleProfiler
from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo.mace import MACE

SIZES = {
    "small": {"n_atoms": 64, "n_edges": 256, "n_graphs": 4},
    "medium": {"n_atoms": 512, "n_edges": 4096, "n_graphs": 16},
    "large": {"n_atoms": 2048, "n_edges": 16384, "n_graphs": 64},
}


def build_model() -> MACE:
    """A small MACE encoder (5 elements, 16 features, 2 interactions)."""
    return MACE(
        node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=8, emb_dim=16)],
        num_elements=8,
        num_features=16,
        r_max=5.0,
        num_bessel=8,
        l_max=1,
        num_interactions=2,
        correlation=2,
    )


def _loss(output, batch=None):
    """Scalar loss on the encoder output — exercises the full backward graph."""
    return output["atoms", "node_features"].sum()


def bench_module(device: str, n_steps: int) -> None:
    """ModuleProfiler over each size: fwd/bwd/opt, throughput, peak mem."""
    print(f"\n{'=' * 78}\nModuleProfiler — MACE fwd+bwd+opt on {device}\n{'=' * 78}")
    for name, dims in SIZES.items():
        model = build_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        factory = MockBatch(
            n_atoms=dims["n_atoms"],
            n_edges=dims["n_edges"],
            n_graphs=dims["n_graphs"],
            atomic_numbers=7,
            device=device,
            seed=0,
        )
        prof = ModuleProfiler(model, loss_fn=_loss, device=device, optimizer=opt)
        warmup = 5 if device == "cpu" else 10
        result = prof.run(factory, n_steps=n_steps, n_warmup=warmup)
        print(f"\n--- size={name} {dims} ---")
        result.print_report()


def bench_trainer_loop(device: str, n_steps: int) -> None:
    """End-to-end Trainer.train() steps/sec including data + hooks."""
    from molix.core.trainer import Trainer
    from molix.hooks.scalar import StepSpeedHook

    print(f"\n{'=' * 78}\nEnd-to-end Trainer loop on {device}\n{'=' * 78}")
    dims = SIZES["medium"]
    factory = MockBatch(
        n_atoms=dims["n_atoms"],
        n_edges=dims["n_edges"],
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
