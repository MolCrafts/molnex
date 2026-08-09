"""Unit tests for the batch-schema helpers in molpot.derivation.protocol.

Mirrors ``src/molpot/derivation/protocol.py``. Only the two helpers that own
the ``graphs`` sub-TensorDict are covered here — :func:`ensure_graphs` (its
shape) and :func:`write_energy` (the shape it asks for). The session
side-channel and :func:`absorb_model_output` are exercised elsewhere.

The contract under test is CLAUDE.md's post-collate schema: ``"graphs"`` is a
``TensorDict(batch_size=[B])`` for ``B`` graphs, the same shape the canonical
collate writes (``src/molix/data/collate.py``). Consumers already read
``batch["graphs"].batch_size[0]`` (``src/molzoo/pinet/potential.py``), so a
``batch_size=[]`` produced here is a latent ``IndexError``.

Units: energy eV. Every expected shape is a hard-coded ``torch.Size``; no
value is recomputed from the input by the assertion side.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molpot.derivation.protocol import (
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    ensure_graphs,
    write_energy,
)


def _empty_batch() -> TensorDict:
    """Root batch with no ``graphs`` namespace yet."""
    return TensorDict(batch_size=[])


def _batch_with_graphs(num_graphs: int = 2) -> TensorDict:
    """Root batch already carrying a schema-conforming ``graphs`` namespace.

    Args:
        num_graphs: ``B`` — the ``graphs`` batch size, and the length of the
            ``num_atoms`` entry that makes the namespace non-empty.

    Returns:
        ``TensorDict`` with ``graphs.batch_size == [num_graphs]`` and
        ``graphs.num_atoms == arange(num_graphs) + 1``.
    """
    return TensorDict(
        graphs=TensorDict(
            num_atoms=torch.arange(num_graphs, dtype=torch.long) + 1,
            batch_size=[num_graphs],
        ),
        batch_size=[],
    )


def _batch_with_atoms(n_atoms: int = 3) -> TensorDict:
    """Root batch with an ``atoms`` namespace, so ``atoms.energy`` is writable."""
    return TensorDict(
        atoms=TensorDict(
            pos=torch.zeros(n_atoms, 3, dtype=torch.float64),
            batch_size=[n_atoms],
        ),
        batch_size=[],
    )


class TestEnsureGraphs:
    """Target: :func:`molpot.derivation.protocol.ensure_graphs`."""

    def test_num_graphs_sets_graphs_batch_size(self):
        batch = ensure_graphs(_empty_batch(), num_graphs=3)
        assert batch["graphs"].batch_size == torch.Size([3])

    def test_omitted_num_graphs_keeps_scalar_batch_size(self):
        batch = ensure_graphs(_empty_batch())
        assert batch["graphs"].batch_size == torch.Size([])

    def test_existing_graphs_keeps_its_contents(self):
        batch = ensure_graphs(_batch_with_graphs(2), num_graphs=2)
        assert torch.equal(batch["graphs", "num_atoms"], torch.tensor([1, 2]))

    def test_existing_graphs_batch_size_is_not_rewritten(self):
        batch = ensure_graphs(_batch_with_graphs(2), num_graphs=5)
        assert batch["graphs"].batch_size == torch.Size([2])

    def test_zero_graphs_gives_empty_first_dim(self):
        batch = ensure_graphs(_empty_batch(), num_graphs=0)
        assert batch["graphs"].batch_size == torch.Size([0])

    def test_single_graph_gives_unit_first_dim(self):
        batch = ensure_graphs(_empty_batch(), num_graphs=1)
        assert batch["graphs"].batch_size == torch.Size([1])


class TestWriteEnergyGraphShape:
    """Target: :func:`molpot.derivation.protocol.write_energy`."""

    def test_vector_energy_sets_graphs_batch_size(self):
        batch = write_energy(_empty_batch(), torch.zeros(4, dtype=torch.float64))
        assert batch["graphs"].batch_size == torch.Size([4])

    def test_vector_energy_is_written_elementwise(self):
        energy = torch.tensor([-1.5, 0.0, 2.25, 7.0], dtype=torch.float64)
        batch = write_energy(_empty_batch(), energy)
        assert torch.equal(batch[ENERGY_KEY], energy)

    def test_zero_dim_energy_keeps_scalar_batch_size(self):
        batch = write_energy(_empty_batch(), torch.tensor(-3.5, dtype=torch.float64))
        assert batch["graphs"].batch_size == torch.Size([])

    def test_atomic_energy_is_written_under_atoms(self):
        atomic = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        batch = write_energy(
            _batch_with_atoms(3),
            torch.tensor([-6.0], dtype=torch.float64),
            atomic_energy=atomic,
        )
        assert torch.equal(batch[ATOMIC_ENERGY_KEY], atomic)
