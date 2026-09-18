"""DataLoader ``collate_fn``: compose collator then optional padder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict

from molix.data.collate import MoleculeCollator, freeze_graph_columns, graph_schema_from
from molix.data.pad import _VALENCE, MoleculePadder, _live_count


class FrameCollator:
    """``collate_fn=FrameCollator(collator=..., padder=...)``.

    ``__call__`` is the DataLoader hook (nn.Module-shaped: construct, then
    call). Collate concatenates molecules; pad inserts ghosts. Neither is
    a :class:`~molix.data.task.BatchTask`.

    Args:
        collator: :class:`~molix.data.collate.MoleculeCollator`.
        padder: :class:`~molix.data.pad.MoleculePadder`, or ``None`` for
            ragged concat (tests).
        prepare: Optional per-frame transform (bonds / units / labels)
            applied on each sample **before** concat. The DataLoader
            collate wrapper calls :meth:`prepare_samples` then concat,
            neighbor list, then pad — wrapping this object in another
            collate callable is skipped by that wrapper.
    """

    def __init__(
        self,
        *,
        collator: MoleculeCollator,
        padder: MoleculePadder | None = None,
        prepare: Any | None = None,
    ) -> None:
        self.collator = collator
        self.padder = padder
        self.prepare = prepare
        self._last: TensorDict | None = None
        self._n_real: int = 0
        self._e_real: int = 0
        self._g_real: int = 0
        self._valence_real: dict[str, int] = {}
        self._graph_schema: tuple[str, ...] | None = None

    def prepare_samples(self, samples: Sequence[Any]) -> Sequence[Any]:
        """Run :attr:`prepare` on each sample. Identity when *prepare* is None."""
        if self.prepare is None:
            return samples
        prepared: list[tuple[Any, ...]] = []
        for sample in samples:
            if hasattr(sample, "keys") and "atoms" in sample:
                frames: Sequence[Any] = (sample,)
            else:
                frames = sample
            prepared.append(tuple(self.prepare.execute(frame) for frame in frames))
        return prepared

    def __call__(self, frames: Sequence[Any]) -> TensorDict:
        batch = self.collator(self.prepare_samples(frames))
        self.record_live(batch)
        if self.padder is not None:
            batch = self.padder(batch)
        batch = self.apply_graph_schema(batch)
        self._last = batch
        return batch

    def compile_graph_schema(self, batch: TensorDict) -> tuple[str, ...]:
        """Lock ``graphs`` keys from the first batch. Idempotent.

        Later batches must carry every compiled name (else ``ValueError``);
        extra names are dropped. The order is canonical
        (:func:`~molix.data.collate.graph_schema_from`), not the first
        batch's insertion order, so spawned DataLoader workers that compile
        independently still agree.
        """
        if self._graph_schema is not None:
            return self._graph_schema
        graphs = batch.get("graphs")
        if graphs is None:
            self._graph_schema = ()
            return self._graph_schema
        backend = getattr(graphs, "_tensordict", None)
        keys = backend.keys() if isinstance(backend, dict) else graphs.keys()
        self._graph_schema = graph_schema_from(keys)
        return self._graph_schema

    def apply_graph_schema(self, batch: TensorDict) -> TensorDict:
        """Compile on the first call, then rebuild ``graphs`` to that schema."""
        schema = self.compile_graph_schema(batch)
        if not schema:
            return batch
        return freeze_graph_columns(batch, schema)

    def record_live(self, batch: TensorDict) -> None:
        """Snapshot real counts before pad (nlist may have just filled edges)."""
        self._n_real = int(batch["atoms"].batch_size[0])
        pairs = batch.get("pairs")
        self._e_real = int(pairs.batch_size[0]) if pairs is not None else 0
        graphs = batch.get("graphs")
        self._g_real = int(graphs.batch_size[0]) if graphs is not None and graphs.batch_size else 0
        live: dict[str, int] = {}
        for ns in _VALENCE:
            if ns not in batch.keys():
                continue
            live[ns] = _live_count(batch[ns])
        self._valence_real = live

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_last"] = None
        return state

    def occupancy_atoms(self) -> float:
        """``n_real / max_atoms`` of the last call; 1.0 when not padded."""
        if self.padder is None or self.padder.max_atoms == 0:
            return 1.0
        return self._n_real / self.padder.max_atoms

    def occupancy_pairs(self) -> float:
        """``e_real / max_pairs`` of the last call; 1.0 when not padded."""
        if self.padder is None or self.padder.max_pairs == 0:
            return 1.0
        return self._e_real / self.padder.max_pairs

    def occupancy_graphs(self) -> float:
        """``g_real / max_graphs`` of the last call; 1.0 when graphs are not padded."""
        if self.padder is None or not self.padder.max_graphs:
            return 1.0
        return self._g_real / self.padder.max_graphs

    def occupancy_valence(self) -> dict[str, float]:
        """``n_real / cap`` per valence family present on the last batch."""
        if self.padder is None:
            return {ns: 1.0 for ns in self._valence_real}
        out: dict[str, float] = {}
        for ns, live in self._valence_real.items():
            cap = self.padder.valence_cap(ns)
            out[ns] = 1.0 if cap == 0 else live / cap
        return out

    def sparsity(self) -> dict[str, float]:
        """``1 - occupancy`` on padded axes."""
        out = {
            "atoms": 1.0 - self.occupancy_atoms(),
            "pairs": 1.0 - self.occupancy_pairs(),
            "graphs": 1.0 - self.occupancy_graphs(),
        }
        out.update({ns: 1.0 - occ for ns, occ in self.occupancy_valence().items()})
        return out

    def nbytes(self) -> int:
        """Storage of the last TensorDict, or 0 before the first call."""
        if self._last is None:
            return 0
        return _tensordict_nbytes(self._last)

    def nbytes_payload(self) -> int:
        """Storage of real rows only (mask / live edge count)."""
        if self._last is None:
            return 0
        if self.padder is None:
            return self.nbytes()
        batch = self._last
        total = 0
        atoms = batch["atoms"]
        n = self._n_real
        for tensor in atoms.values():
            if not torch.is_tensor(tensor):
                continue
            total += tensor[:n].nbytes
        pairs = batch.get("pairs")
        if pairs is not None:
            e = self._e_real
            for tensor in pairs.values():
                if not torch.is_tensor(tensor):
                    continue
                total += tensor[:e].nbytes
        for key in batch.keys():
            name = key if isinstance(key, str) else key[0]
            if name in {"atoms", "pairs"}:
                continue
            block = batch[key]
            if isinstance(block, TensorDict) and name in self._valence_real:
                n = self._valence_real[name]
                for tensor in block.values():
                    if not torch.is_tensor(tensor):
                        continue
                    if tensor.ndim >= 2 and name == "bonds" and tensor.shape[0] <= 4:
                        total += tensor[:, :n].nbytes
                    else:
                        total += tensor[:n].nbytes
                continue
            if isinstance(block, TensorDict):
                total += _tensordict_nbytes(block)
            elif torch.is_tensor(block):
                total += block.nbytes
        return total

    def pad_overhead(self) -> dict[str, float]:
        """Extra bytes from padding, and the ratio vs payload."""
        full = float(self.nbytes())
        payload = float(self.nbytes_payload())
        extra = full - payload
        ratio = extra / payload if payload > 0 else 0.0
        return {"bytes": extra, "ratio": ratio}

    def summary(self) -> dict[str, object]:
        """One log line of last-batch pad efficiency."""
        return {
            "n_real": self._n_real,
            "e_real": self._e_real,
            "g_real": self._g_real,
            "valence_real": dict(self._valence_real),
            "occupancy_atoms": self.occupancy_atoms(),
            "occupancy_pairs": self.occupancy_pairs(),
            "occupancy_graphs": self.occupancy_graphs(),
            "occupancy_valence": self.occupancy_valence(),
            "sparsity": self.sparsity(),
            "nbytes": self.nbytes(),
            "nbytes_payload": self.nbytes_payload(),
            "pad_overhead": self.pad_overhead(),
        }


def _tensordict_nbytes(td: TensorDict) -> int:
    total = 0
    for key in td.keys(include_nested=True, leaves_only=True):
        value = td[key]
        if torch.is_tensor(value):
            total += value.nbytes
    return total
