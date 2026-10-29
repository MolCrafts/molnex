"""In-flight residual energy labels from a frozen elemental dress table.

``graphs.energy_residual`` is the collated total minus the per-graph sum of a
Z-indexed coefficient table over **real** atoms. The raw total at
``graphs.energy`` is not replaced; ``graphs.e0`` is always written. Ghost rows
are excluded by ``atoms.mask``, not by inspecting ``Z``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.dress import AtomicDress
from molix.data.task import BatchTask
from molix.F import batch_add
from molix.frame import as_tensor_frame
from molix.schema import E0_KEY, ENERGY_KEY, ENERGY_RESIDUAL_KEY


class EnergyResidual(BatchTask):
    """Write ``graphs.e0`` and ``graphs.energy_residual`` from a frozen Z table.

    Elemental atomic-reference dressing of a collated batch. For each graph
    ``g``, with structural mask ``m_i`` (``atoms.mask``, else all-True):

    ``e0(g) = sum_{i in g, m_i} c[Z_i] + b``

    with ``b`` the dress's per-graph intercept, added to every graph with at
    least one real atom and never to a dead padding slot.

    ``E_res(g) = E_total(g) - e0(g)``

    ``c[Z]``, ``e0``, ``E_total`` and ``E_res`` share the energy unit of
    ``graphs.energy`` (eV on OMol25 / OC22). The subtract is IEEE float64;
    only ``E_res`` is then stored in ``atoms.pos`` dtype. ``graphs.energy`` is
    left as the same tensor object. Ghost rows never contribute and never
    raise, even when their placeholder ``Z`` is a real element (default
    hydrogen, ``pad_atom_type=1``).

    The table is the :class:`~molix.data.AtomicDress` handed in, read as is
    (not cloned) on every batch; train and val loaders share one instance, so
    fit it on the training split only. Keep this instance on CPU: a deployed
    model takes a deep copy (:class:`~molix.deploy.DressedPotential`).

    Args:
        dress: The frozen table ``c[Z]`` and its support, in the energy unit
            of ``graphs.energy`` (eV).

    Raises:
        TypeError: If *dress* is not an :class:`~molix.data.AtomicDress`.

    Reference:
        Levine et al. "The Open Molecules 2025 (OMol25) Dataset, Evaluations,
        and Models" 2025. https://arxiv.org/abs/2505.08762

        Tran et al. "The Open Catalyst 2022 (OC22) Dataset and Challenges for
        Oxide Electrocatalysis" ACS Catal. 2023.
        https://doi.org/10.1021/acscatal.2c05426
    """

    def __init__(self, dress: AtomicDress) -> None:
        if not isinstance(dress, AtomicDress):
            raise TypeError(f"EnergyResidual takes an AtomicDress, got {type(dress).__name__}")
        self.dress = dress
        self._task_id = f"energy_residual:{dress.digest}"

    @property
    def task_id(self) -> str:
        """Cache-key identity ``energy_residual:<digest>`` of the frozen table.

        :attr:`AtomicDress.digest <molix.data.AtomicDress.digest>`, read once
        at construction: two tables never share a collate identity, and two
        algorithms that produce the same numbers do.
        """
        return self._task_id

    def execute(self, data: TensorDict | dict) -> TensorDict | dict:
        """Write ``graphs.e0`` (eV, float64) and ``graphs.energy_residual`` (eV).

        Read and written through :class:`~molix.data.TensorFrame` via
        :func:`~molix.data.as_tensor_frame`. The wrap is a view — it shares
        every leaf — so the columns land in the batch the caller still holds.
        A plain ``dict`` is refused (``TypeError``): the write would land in a
        container the caller does not hold.

        Args:
            data: Collated batch carrying ``atoms.Z``, ``atoms.batch``,
                ``atoms.pos`` (Å; fixes the residual dtype) and
                ``graphs.energy`` (eV). Optional ``atoms.mask`` (True = real
                atom); absent means every row is real.

        Returns:
            The same batch, mutated in place, with ``graphs.e0`` ``(B,)``
            float64 in eV and ``graphs.energy_residual`` ``(B,)`` in eV and
            ``atoms.pos`` dtype. ``graphs.energy`` is the same tensor object
            as on entry.

        Raises:
            TypeError: *data* is not a ``TensorDict`` / ``TensorFrame``.
            KeyError: Missing ``atoms.Z``, ``atoms.batch``, or
                ``graphs.energy``.
            ValueError: A real atom has ``Z < 0``, ``Z >= len(dress.e0)``,
                or ``support[Z] == False``. The message lists the unique
                offending ``Z`` values.
        """
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                "EnergyResidual writes graphs.e0 / graphs.energy_residual into the batch "
                "it is handed, so it needs the collated TensorDict (or a TensorFrame view "
                f"of it) — a plain {type(data).__name__} would be wrapped into a container "
                "the caller does not hold. Collate the frames with "
                "molix.data.collate_molecules first."
            )
        frame = as_tensor_frame(data)
        atoms = frame["atoms"]
        graphs = frame["graphs"]
        atomic_numbers = atoms["Z"]
        atom_batch = atoms["batch"]
        energy = frame[ENERGY_KEY]
        pos = atoms["pos"]
        # Before the padder (the loader's order) there is no mask: every row is real.
        real = atoms.get("mask", torch.ones_like(atomic_numbers, dtype=torch.bool)).to(
            dtype=torch.bool
        )

        n_table = int(self.dress.e0.shape[0])
        device = atomic_numbers.device
        table = self.dress.e0.to(device=device)
        support = self.dress.support.to(device=device)
        z_long = atomic_numbers.long()
        index = z_long.clamp(0, n_table - 1)
        in_range = (z_long >= 0) & (z_long < n_table)
        illegal = real & (~in_range | ~support[index])
        if bool(illegal.any()):
            bad_z = torch.unique(z_long[illegal]).tolist()
            raise ValueError(
                "EnergyResidual: real atoms have unsupported atomic numbers "
                f"{bad_z} (need 0 <= Z < {n_table} and support[Z])"
            )

        # Clamp then where: ``c[Z] * mask`` is NaN when a ghost gathers NaN.
        gathered = table[index]
        per_atom = torch.where(real, gathered, torch.zeros_like(gathered))
        e0 = batch_add(per_atom, atom_batch, dim_size=graphs.batch_size[0])
        # A graph with no real atom is a dead padding slot: it gets no intercept.
        live = (
            batch_add(real.to(dtype=torch.float64), atom_batch, dim_size=graphs.batch_size[0]) > 0
        )
        intercept = self.dress.intercept.to(device=device)
        e0 = e0 + torch.where(live, intercept, torch.zeros_like(intercept))
        frame[E0_KEY] = e0
        frame[ENERGY_RESIDUAL_KEY] = (energy.to(dtype=torch.float64) - e0).to(dtype=pos.dtype)
        return data
