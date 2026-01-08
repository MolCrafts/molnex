"""Collate functions for batching molecular data.

Provides collate_fn implementations for batching AtomicTD objects.
Uses padded tensors with masks for all atom-level fields.
"""

import torch
from tensordict import TensorDict

from molix.data.atomic_td import AtomicTD


def collate_atomic_tds(atomic_tds: list[AtomicTD]) -> AtomicTD:
    """Collate a list of AtomicTD into a single batched AtomicTD.
    
    Uses padding + attention mask for all atom-level fields to handle 
    variable-length molecules.
    
    Args:
        atomic_tds: List of AtomicTD objects (each representing a single molecule)
        
    Returns:
        Batched AtomicTD with:
            - atoms.Z: Padded tensor [B, max_atoms] with padding_idx=0
            - atoms.mask: Boolean mask [B, max_atoms] (True = valid atom)
            - atoms.xyz: Padded tensor [B, max_atoms, 3]
            - target.*: Stacked per-molecule targets
            - graph.batch: Batch indices for all atoms
            - graph.num_atoms: Number of atoms per molecule [B]
    
    Example:
        >>> td1 = AtomicTD.create(Z=torch.tensor([6, 1, 1]), xyz=torch.randn(3, 3), batch=torch.zeros(3, dtype=torch.long))
        >>> td2 = AtomicTD.create(Z=torch.tensor([8, 1]), xyz=torch.randn(2, 3), batch=torch.zeros(2, dtype=torch.long))
        >>> batched = collate_atomic_tds([td1, td2])
        >>> batched["atoms", "Z"].shape
        torch.Size([2, 3])  # [batch_size, max_atoms]
        >>> batched["atoms", "mask"].shape
        torch.Size([2, 3])  # [batch_size, max_atoms]
        >>> batched["atoms", "xyz"].shape
        torch.Size([2, 3, 3])  # [batch_size, max_atoms, 3]
    """
    assert len(atomic_tds) > 0, "Cannot collate empty list"
    
    # Collect per-atom fields
    Z_list = [td["atoms", "Z"] for td in atomic_tds]
    xyz_list = [td["atoms", "xyz"] for td in atomic_tds]
    
    # Pad atomic numbers with 0 (padding_idx)
    max_atoms = max(len(Z) for Z in Z_list)
    batch_size = len(atomic_tds)
    
    Z_padded = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    xyz_padded = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    
    for i, (Z, xyz) in enumerate(zip(Z_list, xyz_list)):
        n_atoms = len(Z)
        Z_padded[i, :n_atoms] = Z
        mask[i, :n_atoms] = True
        xyz_padded[i, :n_atoms] = xyz
    
    data = {
        ("atoms", "Z"): Z_padded,
        ("atoms", "mask"): mask,
        ("atoms", "xyz"): xyz_padded,
    }
    
    # Update batch indices
    num_atoms_per_mol = [len(Z) for Z in Z_list]
    batch_indices = []
    for mol_idx, num_atoms in enumerate(num_atoms_per_mol):
        batch_indices.append(torch.full((num_atoms,), mol_idx, dtype=torch.int64))
    data[("graph", "batch")] = torch.cat(batch_indices)
    data[("graph", "num_atoms")] = torch.tensor(num_atoms_per_mol, dtype=torch.int64)
    
    # Optional per-atom fields
    optional_atom_fields = [("v", torch.float32, 3), ("f", torch.float32, 3), 
                           ("q", torch.float32, 1), ("type", torch.int64, 1)]
    for field, dtype, dim in optional_atom_fields:
        if ("atoms", field) in atomic_tds[0].keys(include_nested=True):
            field_list = [td["atoms", field] for td in atomic_tds]
            
            # Create padded tensor based on field dimension
            if dim == 1:
                field_padded = torch.zeros(batch_size, max_atoms, dtype=dtype)
                for i, field_val in enumerate(field_list):
                    n_atoms = len(field_val)
                    field_padded[i, :n_atoms] = field_val
            else:  # dim == 3
                field_padded = torch.zeros(batch_size, max_atoms, dim, dtype=dtype)
                for i, field_val in enumerate(field_list):
                    n_atoms = len(field_val)
                    field_padded[i, :n_atoms] = field_val
            
            data[("atoms", field)] = field_padded
    
    # Collect all target keys
    target_keys = set()
    for td in atomic_tds:
        for key in td.keys(include_nested=True):
            if key[0] == "target":
                target_keys.add(key[1])
    
    # Stack per-molecule targets
    for target_key in target_keys:
        target_values = []
        for td in atomic_tds:
            if ("target", target_key) in td.keys(include_nested=True):
                target_values.append(td["target", target_key])
        
        if target_values:
            # Stack along batch dimension
            stacked = torch.cat(target_values, dim=0)
            data[("target", target_key)] = stacked
    
    # Optional topology fields (bonds, angles, dihedrals)
    # These need special handling for batch offsets
    if ("bonds", "i") in atomic_tds[0].keys(include_nested=True):
        bond_i_list = []
        bond_j_list = []
        offset = 0
        for td in atomic_tds:
            num_atoms = len(td["atoms", "Z"])
            bond_i_list.append(td["bonds", "i"] + offset)
            bond_j_list.append(td["bonds", "j"] + offset)
            offset += num_atoms
        
        data[("bonds", "i")] = torch.cat(bond_i_list)
        data[("bonds", "j")] = torch.cat(bond_j_list)
        
        # Bond vectors and distances
        if ("bonds", "diff") in atomic_tds[0].keys(include_nested=True):
            data[("bonds", "diff")] = torch.cat([td["bonds", "diff"] for td in atomic_tds])
        if ("bonds", "dist") in atomic_tds[0].keys(include_nested=True):
            data[("bonds", "dist")] = torch.cat([td["bonds", "dist"] for td in atomic_tds])
    
    # Handle pairs.* fields (for neighbor lists used by EquivariantPotentialNet)
    if ("pairs", "i") in atomic_tds[0].keys(include_nested=True):
        pair_i_list = []
        pair_j_list = []
        offset = 0
        for td in atomic_tds:
            num_atoms = len(td["atoms", "Z"])
            pair_i_list.append(td["pairs", "i"] + offset)
            pair_j_list.append(td["pairs", "j"] + offset)
            offset += num_atoms
        
        data[("pairs", "i")] = torch.cat(pair_i_list)
        data[("pairs", "j")] = torch.cat(pair_j_list)
        
        # Pair vectors and distances
        if ("pairs", "diff") in atomic_tds[0].keys(include_nested=True):
            data[("pairs", "diff")] = torch.cat([td["pairs", "diff"] for td in atomic_tds])
        if ("pairs", "dist") in atomic_tds[0].keys(include_nested=True):
            data[("pairs", "dist")] = torch.cat([td["pairs", "dist"] for td in atomic_tds])
    
    return AtomicTD(data, batch_size=[])