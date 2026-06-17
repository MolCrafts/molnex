"""Model adapters for the generic ``pair_style molnex`` LAMMPS interface.

The LAMMPS C++ pair style talks to an AOT-Inductor ``.so`` through
:class:`molnex.interface.ModelRunner` with a *flat tensor* calling convention —
``TensorDict`` never crosses into C++. The contract is therefore the **export
calling convention**, not a model's in-memory ``forward``:

    inputs  : Z ``(N,)`` int64, pos ``(N, 3)`` float, edge_index ``(E, 2)`` int64
    outputs : energy ``()`` scalar float, forces ``(N, 3)`` float

An :class:`LammpsAdapter` is the Python-side shim that maps this flat convention
onto a specific model's real input/output shapes. A molnex-native model (nested
``TensorDict`` forward) uses :class:`MolnexTensorDictAdapter`; a third-party model
that already speaks the flat convention uses :class:`FlatTensorAdapter`; anything
else gets a ~15-line subclass implementing :meth:`build_inputs` / :meth:`read_outputs`.

Adapters self-register under a string name (the same strategy-pattern + registry
used by :mod:`molix.quant`), so :func:`molix.lammps.export_for_lammps` can select
one by name and stamp it into ``meta.json`` for provenance.

Edge convention follows the repo-wide rule (see ``CLAUDE.md`` → Edge Convention):
``edge_index[:, 0]`` is the source, ``[:, 1]`` the target, and
``bond_diff = pos[target] - pos[source]``. The C++ side sorts edges by
``(source, target)`` before the call; adapters must not assume any other order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LammpsAdapter(ABC):
    """Strategy mapping the flat LAMMPS calling convention onto one model.

    Concrete subclasses set a class-level :attr:`name` (auto-registered) and
    implement :meth:`build_inputs` (flat tensors → model input) and
    :meth:`read_outputs` (model output → ``(energy_scalar, forces)``). The whole
    adapter is wrapped in :class:`LammpsForward`, which is the ``nn.Module`` that
    actually gets AOT-exported.
    """

    name: str = ""
    _registry: dict[str, type[LammpsAdapter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            LammpsAdapter._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> LammpsAdapter:
        """Instantiate the registered adapter for ``name`` (e.g. ``"molnex-tensordict"``)."""
        try:
            return cls._registry[name]()
        except KeyError:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(f"unknown adapter {name!r}; valid adapters: {valid}") from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """All registered adapter names."""
        return tuple(sorted(cls._registry))

    @abstractmethod
    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> object:
        """Convert flat LAMMPS tensors into whatever ``model`` consumes.

        Args:
            model: The wrapped potential.
            Z: Atomic numbers ``(N,)`` int64.
            pos: Positions ``(N, 3)`` float (model dtype).
            edge_index: Source-target pairs ``(E, 2)`` int64, sorted by ``(src, tgt)``.

        Returns:
            The object passed to ``model(...)`` (e.g. a nested ``TensorDict``).
        """

    @abstractmethod
    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract ``(energy_scalar (), forces (N, 3))`` from the model output."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MolnexTensorDictAdapter(LammpsAdapter):
    """Adapter for molnex-native potentials (PiNet, MACE, … via ``PiNetPotential``-style).

    Builds the post-collate nested ``TensorDict`` (``atoms`` / ``edges`` / ``graphs``
    per ``CLAUDE.md``), runs ``model(batch, compute_forces=True)``, and reads the
    ``{"energy", "forces"}`` output dict. Energy is summed to a scalar (single
    graph), matching what the C++ pair style accumulates into ``eng_vdwl``.
    """

    name = "molnex-tensordict"

    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> object:
        from tensordict import TensorDict

        n = Z.shape[0]
        src = edge_index[:, 0]
        tgt = edge_index[:, 1]
        bond_diff = pos[tgt] - pos[src]                       # pos[target] - pos[source]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-6)
        return TensorDict(
            atoms=TensorDict(
                Z=Z,
                pos=pos,
                batch=torch.zeros(n, dtype=torch.long, device=pos.device),
                batch_size=[n],
            ),
            edges=TensorDict(
                edge_index=edge_index,
                bond_diff=bond_diff,
                bond_dist=bond_dist,
                batch_size=[edge_index.shape[0]],
            ),
            graphs=TensorDict(
                num_atoms=torch.full((1,), n, dtype=torch.long, device=pos.device),
                batch_size=[1],
            ),
            batch_size=[],
        )

    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(out, dict), "molnex models return a dict; got " + type(out).__name__
        return out["energy"].sum(), out["forces"]


class FlatTensorAdapter(LammpsAdapter):
    """Adapter for third-party models already speaking the flat convention.

    The wrapped model must implement ``forward(Z, pos, edge_index)`` returning
    either ``(energy, forces)`` or a dict with ``"energy"`` / ``"forces"`` keys.
    Use this as the zero-glue path; write a bespoke :class:`LammpsAdapter`
    subclass when a model needs real input/output translation.
    """

    name = "flat"

    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> object:
        return (Z, pos, edge_index)

    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(out, dict):
            return out["energy"].sum(), out["forces"]
        energy, forces = out  # type: ignore[misc]
        return energy.sum(), forces


class LammpsForward(nn.Module):
    """Flat-convention ``nn.Module`` wrapper that is the actual AOT-export target.

    Holds a potential and an :class:`LammpsAdapter`. Its ``forward(Z, pos,
    edge_index)`` returns ``(energy_scalar, forces)`` — the exact signature the
    C++ ``pair_style molnex`` invokes through the AOTI runner.
    """

    def __init__(self, model: nn.Module, adapter: LammpsAdapter | str = "molnex-tensordict"):
        super().__init__()
        self.model = model
        self.adapter = (
            adapter if isinstance(adapter, LammpsAdapter) else LammpsAdapter.from_name(adapter)
        )

    def forward(
        self, Z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.adapter.build_inputs(self.model, Z, pos, edge_index)
        if isinstance(inputs, tuple):
            out = self.model(*inputs)
        else:
            out = self.model(inputs, compute_forces=True)
        energy, forces = self.adapter.read_outputs(out)
        return energy.reshape(()), forces
