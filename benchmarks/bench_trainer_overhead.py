"""Trainer *machinery* overhead — isolated from model compute.

Uses a trivial model (one parameter, ~zero FLOPs) so the measured per-step
time is almost entirely Trainer loop overhead: Step-protocol dispatch, hook
calls, ``batch_to``, TrainState namespace-validated writes, per-batch
``next(model.parameters()).device`` walks, ``config`` lookups, and the
single-process DDP no-op checks.

Three measurements:
  A  raw loop      — zero_grad / forward / loss / backward / step, no Trainer
  B  Trainer       — same model+data, no hooks
  C  Trainer+hooks — adds N no-op hooks to price hook dispatch

overhead(Trainer)  = B - A         (pure loop machinery)
overhead(per hook) = (C - B) / N   (per-hook-callback dispatch)

Then cProfile of B attributes the overhead to concrete Trainer methods.

Run:  python benchmarks/bench_trainer_overhead.py --steps 4000
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.core.hook import BaseHook
from molix.core.trainer import Trainer


class TinyModel(nn.Module):
    """One scalar parameter; forward is a single multiply — compute ≈ 0."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, batch: TensorDict) -> torch.Tensor:
        return batch["atoms", "pos"].sum() * self.w.sum()


def _loss(pred: torch.Tensor, batch: TensorDict) -> torch.Tensor:
    return pred * pred


def _make_batch(n_atoms: int = 32, n_edges: int = 128, n_graphs: int = 4) -> TensorDict:
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": torch.ones(n_atoms, dtype=torch.long),
                    "pos": torch.randn(n_atoms, 3),
                    "batch": torch.arange(n_graphs).repeat_interleave(n_atoms // n_graphs),
                },
                batch_size=[n_atoms],
            ),
            "edges": TensorDict(
                {
                    "edge_index": torch.randint(0, n_atoms, (n_edges, 2)),
                    "bond_diff": torch.randn(n_edges, 3),
                    "bond_dist": torch.rand(n_edges),
                },
                batch_size=[n_edges],
            ),
        },
        batch_size=[],
    )


class _NoopHook(BaseHook):
    """Subscribes to the two per-batch callbacks but does nothing."""

    def on_train_batch_start(self, trainer, state, batch):  # noqa: D102
        pass

    def on_train_batch_end(self, trainer, state, batch, outputs):  # noqa: D102
        pass


class _Batches:
    """Fixed list of pre-built batches replayed each epoch (one epoch)."""

    def __init__(self, batch: TensorDict, n: int) -> None:
        self._b = [batch] * n

    def setup(self, stage: str = "fit") -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def train_dataloader(self):
        return iter(self._b)

    def val_dataloader(self):
        return iter(self._b[:1])


def _raw_loop(model, opt, batches: list[TensorDict]) -> None:
    """Irreducible per-step work, no Trainer/Step/hook machinery."""
    model.train()
    for batch in batches:
        opt.zero_grad()
        pred = model(batch)
        loss = _loss(pred, batch)
        loss.backward()
        opt.step()


def _time(fn, *a) -> float:
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hooks", type=int, default=4)
    args = ap.parse_args()
    n = args.steps
    batch = _make_batch()
    torch.set_num_threads(1)  # pure Python-dispatch overhead, no BLAS noise

    # warmup (JIT autograd graph, allocator)
    m = TinyModel()
    _raw_loop(m, torch.optim.SGD(m.parameters(), lr=0.0), [batch] * 50)

    # A — raw loop
    m = TinyModel()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    a = _time(_raw_loop, m, opt, [batch] * n)

    # B — Trainer, no hooks
    tb = Trainer(
        model=TinyModel(),
        loss_fn=_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
        hooks=[],
    )
    b = _time(lambda: tb.train(datamodule=_Batches(batch, n), max_epochs=1))

    # C — Trainer + N no-op hooks
    tc = Trainer(
        model=TinyModel(),
        loss_fn=_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
        hooks=[_NoopHook() for _ in range(args.hooks)],
    )
    c = _time(lambda: tc.train(datamodule=_Batches(batch, n), max_epochs=1))

    us = 1e6 / n
    print(f"\n{'=' * 64}\nTrainer machinery overhead  (n={n} steps, 1 thread)\n{'=' * 64}")
    print(f"  A raw loop          {a * us:8.2f} us/step   ({n / a:8.0f} step/s)")
    print(f"  B Trainer no-hooks  {b * us:8.2f} us/step   ({n / b:8.0f} step/s)")
    print(f"  C Trainer +{args.hooks} hooks  {c * us:8.2f} us/step   ({n / c:8.0f} step/s)")
    print(f"\n  Trainer overhead    {(b - a) * us:8.2f} us/step   ({(b - a) / b * 100:.1f}% of B)")
    print(
        f"  per hook-callback   {(c - b) * us / (2 * args.hooks):8.2f} us/step  (2 callbacks/step)"
    )

    # cProfile attribution of B
    print(f"\n{'=' * 64}\ncProfile — Trainer.train (no hooks), top by cumulative\n{'=' * 64}")
    pr = cProfile.Profile()
    tp = Trainer(
        model=TinyModel(),
        loss_fn=_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
        hooks=[],
    )
    pr.enable()
    tp.train(datamodule=_Batches(batch, n), max_epochs=1)
    pr.disable()
    s = StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("tottime")
    st.print_stats(25)
    # keep molix/own-code rows + header
    for line in s.getvalue().splitlines():
        if any(k in line for k in ("molix", "tottime", "function calls", "tensordict")) or (
            "/" not in line and "{" in line and "built-in" in line
        ):
            print(line)


if __name__ == "__main__":
    main()
