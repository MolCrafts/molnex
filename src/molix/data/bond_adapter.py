"""molpy bonds-block -> canonical bond_index [2, N] boundary helper.

molpy's ``Atomistic.to_frame()`` emits a ``bonds`` block with columns
``atomi`` / ``atomj`` (the covalent connectivity) plus a bond-type column.
This helper stacks those columns into the canonical COO ``bond_index``
``[2, num_bonds]`` (PyG layout) paired with ``bond_types`` ``[num_bonds]``
consumed by :class:`~molpot.potentials.bonds.BondHarmonic` and the
``bonds`` collate namespace. See spec graph-connectivity-alignment-03-bonds.

Deliberately minimal — it is only the column->COO stack, not a general molpy
structure importer.
"""

from __future__ import annotations

import torch


def bond_index_from_columns(
    atomi: torch.Tensor,
    atomj: torch.Tensor,
    bond_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack molpy ``atomi`` / ``atomj`` columns into canonical ``bond_index``.

    Args:
        atomi: Source atom index per bond ``(num_bonds,)``.
        atomj: Target atom index per bond ``(num_bonds,)``.
        bond_types: Integer bond-type id per bond ``(num_bonds,)``.

    Returns:
        ``(bond_index, bond_types)`` where ``bond_index`` is COO
        ``[2, num_bonds]`` (row 0 = source, row 1 = target), both long.

    Raises:
        ValueError: the three columns do not share a single length.
    """
    atomi = torch.as_tensor(atomi).long().reshape(-1)
    atomj = torch.as_tensor(atomj).long().reshape(-1)
    bond_types = torch.as_tensor(bond_types).long().reshape(-1)
    if not (atomi.shape == atomj.shape == bond_types.shape):
        raise ValueError(
            f"atomi/atomj/bond_types must share length; got {atomi.shape}, "
            f"{atomj.shape}, {bond_types.shape}."
        )
    return torch.stack([atomi, atomj], dim=0), bond_types
