"""In-flight residual energy labels from a frozen elemental reference table.

``graphs.energy_residual`` is the collated total minus the per-graph sum of a
Z-indexed coefficient table over **real** atoms. The raw total at
``graphs.energy`` is not replaced; ``graphs.e0`` is always written. Ghost rows
are excluded by ``atoms.mask``, not by inspecting ``Z``.
"""

from __future__ import annotations

import hashlib

import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.task import BatchTask
from molix.F import batch_add
from molix.frame import as_tensor_frame
from molix.schema import E0_KEY, ENERGY_KEY, ENERGY_RESIDUAL_KEY


class EnergyResidual(BatchTask):
    """Write ``graphs.e0`` and ``graphs.energy_residual`` from a frozen Z table.

    Elemental atomic-reference dressing of a collated batch. For each graph
    ``g``, with structural mask ``m_i`` (``atoms.mask``, else all-True):

    ``e0(g) = sum_{i in g, m_i} c[Z_i]``

    ``E_res(g) = E_total(g) - e0(g)``

    ``c[Z]``, ``e0``, ``E_total`` and ``E_res`` share the energy unit of
    ``graphs.energy`` (eV on OMol25 / OC22). The subtract is IEEE float64;
    only ``E_res`` is then stored in ``atoms.pos`` dtype. ``graphs.energy`` is
    left as the same tensor object. Ghost rows never contribute and never
    raise, even when their placeholder ``Z`` is a real element (default
    hydrogen, ``pad_atom_type=1``).

    Args:
        coefficients: 1-D Z-indexed reference table, energy unit of
            ``graphs.energy`` (eV). Cloned and stored as float64.
        support: 1-D boolean mask of the same length. ``True`` means that
            atomic number is defined. A zero coefficient with
            ``support[Z] == False`` is not support.

    Reference:
        Levine et al. "The Open Molecules 2025 (OMol25) Dataset, Evaluations,
        and Models" 2025. https://arxiv.org/abs/2505.08762

        Tran et al. "The Open Catalyst 2022 (OC22) Dataset and Challenges for
        Oxide Electrocatalysis" ACS Catal. 2023.
        https://doi.org/10.1021/acscatal.2c05426
    """

    def __init__(self, *, coefficients: torch.Tensor, support: torch.Tensor) -> None:
        if coefficients.ndim != 1:
            raise ValueError(
                "EnergyResidual coefficients must be 1-D "
                f"(got ndim={coefficients.ndim}, shape={tuple(coefficients.shape)})"
            )
        if support.ndim != 1:
            raise ValueError(
                "EnergyResidual support must be 1-D "
                f"(got ndim={support.ndim}, shape={tuple(support.shape)})"
            )
        if coefficients.shape[0] != support.shape[0]:
            raise ValueError(
                "EnergyResidual coefficients and support must have the same length "
                f"(got {coefficients.shape[0]} and {support.shape[0]})"
            )
        if support.dtype != torch.bool:
            raise ValueError(
                "EnergyResidual support.dtype must be torch.bool "
                f"(got {support.dtype}; uint8 is not bool)"
            )
        self.coefficients = coefficients.detach().clone().to(dtype=torch.float64)
        self.support = support.detach().clone().to(dtype=torch.bool)
        coeff_bytes = (
            self.coefficients.cpu().contiguous().numpy().astype("<f8", copy=False).tobytes()
        )
        support_bytes = self.support.cpu().contiguous().to(dtype=torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(coeff_bytes + support_bytes).hexdigest()[:16]
        self._task_id = f"energy_residual:{digest}"

    @property
    def task_id(self) -> str:
        """Cache-key identity ``energy_residual:<digest>`` of the frozen table.

        The digest is a hex prefix of SHA-256 over little-endian float64
        coefficient bytes then uint8 support bytes, so two tables never share
        a collate identity.
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
            ValueError: A real atom has ``Z < 0``, ``Z >= len(coefficients)``,
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
        if "mask" in atoms:
            real = atoms["mask"].to(dtype=torch.bool)
        else:
            real = torch.ones_like(atomic_numbers, dtype=torch.bool)

        n_table = int(self.coefficients.shape[0])
        device = atomic_numbers.device
        table = self.coefficients.to(device=device)
        support = self.support.to(device=device)
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
        frame[E0_KEY] = e0
        frame[ENERGY_RESIDUAL_KEY] = (energy.to(dtype=torch.float64) - e0).to(dtype=pos.dtype)
        return data
