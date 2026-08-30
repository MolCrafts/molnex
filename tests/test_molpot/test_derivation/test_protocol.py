"""Unit tests for the batch-schema helpers in molpot.derivation.protocol.

Mirrors ``src/molpot/derivation/protocol.py``. Three helpers are covered here:
:func:`ensure_graphs` (its shape), :func:`write_energy` (the shape it asks for)
and :func:`absorb_model_output` (which spelling of the per-atom energy it
accepts). The session side-channel is exercised elsewhere.

The contract under test is CLAUDE.md's post-collate schema: ``"graphs"`` is a
``TensorDict(batch_size=[B])`` for ``B`` graphs, the same shape the canonical
collate writes (``src/molix/batch/collate.py``). Consumers already read
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
    absorb_model_output,
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


class TestAbsorbModelOutput:
    """Target: :func:`molpot.derivation.protocol.absorb_model_output`.

    The plain-dict branch used to read the bare string ``"atomic_energy"`` with
    no constant behind it. A model returning a plain dict that spelled the key
    differently had its per-atom energy dropped silently: ``.get`` returned
    ``None`` and :func:`write_energy` skipped the write. That was a feature gap
    while the per-atom energy was optional; from spec ``lammps-iface-05-pbc`` on
    it is the *only* source of the LAMMPS partition energy, so a silent drop
    becomes a silent zero energy on every rank.

    The canonical flat spelling is therefore
    :data:`molix.schema.ATOMIC_ENERGY_FLAT_KEY` — the flat alias of the nested
    :data:`~molix.schema.ATOMIC_ENERGY_KEY`, and the only accepted one. It is
    imported inside each test so only this class is red while it is missing.

    Units: energy eV. Every expected value is a hard-coded literal.
    """

    ATOMIC = [-1.0, -2.0, -3.0]
    TOTAL = -6.0

    @classmethod
    def _atomic(cls) -> torch.Tensor:
        return torch.tensor(cls.ATOMIC, dtype=torch.float64)

    @classmethod
    def _total(cls) -> torch.Tensor:
        return torch.tensor([cls.TOTAL], dtype=torch.float64)

    def test_canonical_flat_key_is_written_to_the_nested_target(self):
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        batch = _batch_with_atoms(3)
        absorb_model_output(
            batch, {"energy": self._total(), ATOMIC_ENERGY_FLAT_KEY: self._atomic()}
        )
        assert torch.equal(batch[ATOMIC_ENERGY_KEY], self._atomic())
        assert torch.equal(batch[ENERGY_KEY], self._total())

    def test_non_canonical_spelling_is_not_absorbed(self):
        """A third spelling must not reach ``("atoms", "energy")`` — one name only."""
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        assert "node_energies" != ATOMIC_ENERGY_FLAT_KEY
        batch = _batch_with_atoms(3)
        absorb_model_output(batch, {"energy": self._total(), "node_energies": self._atomic()})
        assert "energy" not in batch["atoms"].keys()
        assert torch.equal(batch[ENERGY_KEY], self._total())

    def test_tensordict_branch_reads_the_nested_keys(self):
        """Unchanged by v3: a TensorDict output is read at the nested keys.

        Deliberately constant-free — this is the "did not regress" lock for the
        branch the flat-alias fix does not touch, so it must stay green while
        the two cases above are red.
        """
        out = TensorDict(
            atoms=TensorDict(energy=self._atomic(), batch_size=[3]),
            graphs=TensorDict(energy=self._total(), batch_size=[1]),
            batch_size=[],
        )
        batch = absorb_model_output(_batch_with_atoms(3), out)
        assert torch.equal(batch[ATOMIC_ENERGY_KEY], self._atomic())
        assert torch.equal(batch[ENERGY_KEY], self._total())
