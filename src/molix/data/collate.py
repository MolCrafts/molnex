from __future__ import annotations
import torch
from molix.data.atom_td import AtomTD


"""Collate functions for batching molecular data.

Provides collate_fn implementations for batching AtomTD objects by concatenation.
"""

def collate_atomic_tds(atomic_tds: list[AtomTD]) -> AtomTD:
    """Collate a list of AtomTD into a single batched AtomTD by concatenation.
    
    Concatenates all atom-level and bond-level fields along the first dimension.
    Handles bond index offsets based on cumulative atom counts.
    
    Args:
        atomic_tds: List of AtomTD objects (each representing a single molecule)
        
    Returns:
        Batched AtomTD with concatenated fields and proper batch indices.
    """
    assert len(atomic_tds) > 0, "Cannot collate empty list"
    
    # 1. Handle atom-level fields
    Z = torch.cat([td.Z for td in atomic_tds], dim=0)
    xyz = torch.cat([td.xyz for td in atomic_tds], dim=0)
    
    # Calculate cumulative offsets for indices
    num_atoms_list = [len(td.Z) for td in atomic_tds]
    offsets = torch.zeros(len(atomic_tds), dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(num_atoms_list[:-1]), dim=0)
    
    # 2. Handle batch indices
    batch_list = []
    for i, n in enumerate(num_atoms_list):
        batch_list.append(torch.full((n,), i, dtype=torch.long))
    batch = torch.cat(batch_list, dim=0)
    
    # 3. Handle optional atom fields
    v = torch.cat([td.v for td in atomic_tds], dim=0) if atomic_tds[0].v is not None else None
    f = torch.cat([td.f for td in atomic_tds], dim=0) if atomic_tds[0].f is not None else None
    q = torch.cat([td.q for td in atomic_tds], dim=0) if atomic_tds[0].q is not None else None
    atom_type = torch.cat([td.atom_type for td in atomic_tds], dim=0) if atomic_tds[0].atom_type is not None else None
    
    # 4. Handle bond-level fields with offsets
    bond_i = None
    bond_j = None
    bond_vec = None
    bond_dist = None
    bond_type = None
    
    if atomic_tds[0].bond_i is not None:
        bond_i_list = []
        bond_j_list = []
        for i, td in enumerate(atomic_tds):
            bond_i_list.append(td.bond_i + offsets[i])
            bond_j_list.append(td.bond_j + offsets[i])
        bond_i = torch.cat(bond_i_list, dim=0)
        bond_j = torch.cat(bond_j_list, dim=0)
        
        if atomic_tds[0].bond_vec is not None:
            bond_vec = torch.cat([td.bond_vec for td in atomic_tds], dim=0)
        if atomic_tds[0].bond_dist is not None:
            bond_dist = torch.cat([td.bond_dist for td in atomic_tds], dim=0)
        if atomic_tds[0].bond_type is not None:
            bond_type = torch.cat([td.bond_type for td in atomic_tds], dim=0)
            
    # 5. Handle molecule-level fields (stack them)
    energy = None
    if atomic_tds[0].energy is not None:
        energy = torch.cat([td.energy for td in atomic_tds], dim=0)
    
    dipole = None
    if atomic_tds[0].dipole is not None:
        dipole = torch.cat([td.dipole for td in atomic_tds], dim=0)
        
    stress = None
    if atomic_tds[0].stress is not None:
        stress = torch.cat([td.stress for td in atomic_tds], dim=0)

    num_atoms = torch.tensor(num_atoms_list, dtype=torch.long)
    
    # 6. Handle angle and dihedral fields (omitted for brevity but same logic with offsets)
    # They can be added as needed.
    
    return AtomTD.create(
        Z=Z,
        xyz=xyz,
        batch=batch,
        v=v,
        f=f,
        q=q,
        atom_type=atom_type,
        bond_i=bond_i,
        bond_j=bond_j,
        bond_vec=bond_vec,
        bond_dist=bond_dist,
        bond_type=bond_type,
        num_atoms=num_atoms,
        energy=energy,
        dipole=dipole,
        stress=stress
    )