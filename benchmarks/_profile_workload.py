"""Small benchmark-only adapters for the public profiler workload contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from molix.profiler import Profiler, ProfilerConfig, ProfileSpec, Report


class TrainingWorkload(nn.Module):
    """Represent forward, loss, backward and optimizer work as one module call."""

    def __init__(
        self, model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def get_extra_state(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    def forward(self, batch: Any) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch)
        loss = self.loss_fn(output, batch)
        loss.backward()
        self.optimizer.step()
        return loss.detach()


class SingleBatchLoader:
    """Expose one synthetic batch through the dataloader class contract."""

    def __init__(self, batch_source: Callable[[], Any]) -> None:
        self.batch_source = batch_source

    def __iter__(self):
        yield self.batch_source()


def profile_training(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    batch_factory: Callable[[], Any],
    *,
    warmup: int,
    steps: int,
    device: str,
) -> Report:
    """Profile a training workload while keeping its semantics in ``TrainingWorkload``."""
    config = ProfilerConfig(
        device=device,
        warmup=warmup,
        steps=steps,
        transfer_batch=False,
        memory="cuda" if torch.device(device).type == "cuda" else "none",
        activities=("cpu", "cuda") if torch.device(device).type == "cuda" else ("cpu",),
        hotspot_steps=min(5, steps),
        hotspot_limit=20,
    )
    return Profiler().profile(
        ProfileSpec(
            "training-workload",
            TrainingWorkload,
            SingleBatchLoader,
            model_args=(model, loss_fn, optimizer),
            dataloader_args=(batch_factory,),
        ),
        config,
    )
