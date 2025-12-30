"""AtomicTD: TensorDict-based atomistic data structure.

Protocol-level data container for molecular systems.
Moved to molix as it's infrastructure, not potential-specific.

Schema conventions:
- Separate index fields: bonds.i/j, angles.i/j/k, dihedrals.i/j/k/l
- target.* namespace for molecular targets (no y_ prefix)
- Hierarchical keys: ("namespace", "field")
- Configurable dtype via Config
"""

from typing import Optional
from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class Config:
    """Global configuration for TensorDict."""
    dtype: torch.dtype = torch.float32


class AtomicTD(TensorDict):
    """Atomistic TensorDict for molecular systems.
    
    Field naming convention (hierarchical keys):
    
    Atomic properties:
    - ("atoms", "z"): Atomic numbers [N] (int64)
    - ("atoms", "x"): Atomic positions [N, 3] (dtype)
    - ("atoms", "v"): Atomic velocities [N, 3] (dtype, optional)
    - ("atoms", "f"): Atomic forces [N, 3] (dtype, optional, target)
    - ("atoms", "q"): Atomic charges [N] (dtype, optional)
    - ("atoms", "type"): Atom types [N] (int64, optional)
    
    Bond topology (separate i/j indices):
    - ("bonds", "i"): Bond source indices [E] (int64, optional)
    - ("bonds", "j"): Bond target indices [E] (int64, optional)
    - ("bonds", "vec"): Bond vectors [E, 3] (dtype, optional)
    - ("bonds", "dist"): Bond distances [E] (dtype, optional)
    - ("bonds", "type"): Bond types [E] (int64, optional)
    
    Angle topology (separate i/j/k indices):
    - ("angles", "i"): Angle first atom [A] (int64, optional)
    - ("angles", "j"): Angle center atom [A] (int64, optional)
    - ("angles", "k"): Angle third atom [A] (int64, optional)
    - ("angles", "theta"): Angle values [A] (dtype, optional)
    - ("angles", "type"): Angle types [A] (int64, optional)
    
    Dihedral topology (separate i/j/k/l indices):
    - ("dihedrals", "i"): Dihedral first atom [D] (int64, optional)
    - ("dihedrals", "j"): Dihedral second atom [D] (int64, optional)
    - ("dihedrals", "k"): Dihedral third atom [D] (int64, optional)
    - ("dihedrals", "l"): Dihedral fourth atom [D] (int64, optional)
    - ("dihedrals", "phi"): Dihedral values [D] (dtype, optional)
    - ("dihedrals", "type"): Dihedral types [D] (int64, optional)
    
    Improper topology (separate i/j/k/l indices):
    - ("impropers", "i"): Improper first atom [I] (int64, optional)
    - ("impropers", "j"): Improper second atom [I] (int64, optional)
    - ("impropers", "k"): Improper third atom [I] (int64, optional)
    - ("impropers", "l"): Improper fourth atom [I] (int64, optional)
    - ("impropers", "type"): Improper types [I] (int64, optional)
    
    Graph metadata:
    - ("graph", "batch"): Molecule indices [N] (int64)
    - ("graph", "num_atoms"): Atoms per molecule [B] (int64, optional)
    
    Molecular targets (target.* namespace, no y_ prefix):
    - ("target", "energy"): Molecular energy [B] (dtype, optional)
    - ("target", "dipole"): Dipole moment [B, 3] (dtype, optional)
    - ("target", "stress"): Stress tensor [B, 3, 3] (dtype, optional)
    
    Example:
        >>> config = Config(dtype=torch.float32)
        >>> atomic_td = AtomicTD.create(
        ...     z=torch.tensor([8, 1, 1]),
        ...     x=torch.randn(3, 3),
        ...     batch=torch.tensor([0, 0, 0]),
        ...     bond_i=torch.tensor([0, 1, 0]),
        ...     bond_j=torch.tensor([1, 2, 2]),
        ...     energy=torch.tensor([0.5]),
        ...     config=config
        ... )
        >>> atomic_td["target", "energy"]
        tensor([0.5000])
    """
    
    @classmethod
    def create(
        cls,
        z: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        config: Optional[Config] = None,
        # Atomic properties
        v: Optional[torch.Tensor] = None,
        f: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        atom_type: Optional[torch.Tensor] = None,
        # Bond topology (separate i/j indices)
        bond_i: Optional[torch.Tensor] = None,
        bond_j: Optional[torch.Tensor] = None,
        bond_vec: Optional[torch.Tensor] = None,
        bond_dist: Optional[torch.Tensor] = None,
        bond_type: Optional[torch.Tensor] = None,
        # Angle topology (separate i/j/k indices)
        angle_i: Optional[torch.Tensor] = None,
        angle_j: Optional[torch.Tensor] = None,
        angle_k: Optional[torch.Tensor] = None,
        angle_theta: Optional[torch.Tensor] = None,
        angle_type: Optional[torch.Tensor] = None,
        # Dihedral topology (separate i/j/k/l indices)
        dihedral_i: Optional[torch.Tensor] = None,
        dihedral_j: Optional[torch.Tensor] = None,
        dihedral_k: Optional[torch.Tensor] = None,
        dihedral_l: Optional[torch.Tensor] = None,
        dihedral_phi: Optional[torch.Tensor] = None,
        dihedral_type: Optional[torch.Tensor] = None,
        # Improper topology (separate i/j/k/l indices)
        improper_i: Optional[torch.Tensor] = None,
        improper_j: Optional[torch.Tensor] = None,
        improper_k: Optional[torch.Tensor] = None,
        improper_l: Optional[torch.Tensor] = None,
        improper_type: Optional[torch.Tensor] = None,
        # Graph metadata
        num_atoms: Optional[torch.Tensor] = None,
        # Molecular targets (target.* namespace, no y_ prefix)
        energy: Optional[torch.Tensor] = None,
        dipole: Optional[torch.Tensor] = None,
        stress: Optional[torch.Tensor] = None,
        **kwargs
    ) -> "AtomicTD":
        """Create AtomicTD with explicit field names.
        
        Args:
            z: Atomic numbers [N]
            x: Atomic positions [N, 3]
            batch: Molecule indices [N]
            config: Configuration (dtype)
            v: Atomic velocities [N, 3] (optional)
            f: Atomic forces [N, 3] (optional, target)
            q: Atomic charges [N] (optional)
            atom_type: Atom types [N] (optional)
            bond_i: Bond source indices [E] (optional)
            bond_j: Bond target indices [E] (optional)
            bond_vec: Bond vectors [E, 3] (optional)
            bond_dist: Bond distances [E] (optional)
            bond_type: Bond types [E] (optional)
            angle_i/j/k: Angle atom indices [A] (optional)
            angle_theta: Angle values [A] (optional)
            angle_type: Angle types [A] (optional)
            dihedral_i/j/k/l: Dihedral atom indices [D] (optional)
            dihedral_phi: Dihedral values [D] (optional)
            dihedral_type: Dihedral types [D] (optional)
            improper_i/j/k/l: Improper atom indices [I] (optional)
            improper_type: Improper types [I] (optional)
            num_atoms: Atoms per molecule [B] (optional)
            energy: Molecular energy [B] (optional, target)
            dipole: Dipole moment [B, 3] (optional, target)
            stress: Stress tensor [B, 3, 3] (optional, target)
            **kwargs: Additional fields
            
        Returns:
            AtomicTD instance
        """
        if config is None:
            config = Config()
        
        dtype = config.dtype
        
        # Required fields
        data = {
            ("atoms", "z"): z.long(),
            ("atoms", "x"): x.to(dtype),
            ("graph", "batch"): batch.long(),
        }
        
        # Optional atomic properties
        if v is not None:
            data["atoms", "v"] = v.to(dtype)
        if f is not None:
            data["atoms", "f"] = f.to(dtype)
        if q is not None:
            data["atoms", "q"] = q.to(dtype)
        if atom_type is not None:
            data["atoms", "type"] = atom_type.long()
        
        # Bond topology (separate i/j indices)
        if bond_i is not None:
            data["bonds", "i"] = bond_i.long()
        if bond_j is not None:
            data["bonds", "j"] = bond_j.long()
        if bond_vec is not None:
            data["bonds", "vec"] = bond_vec.to(dtype)
        if bond_dist is not None:
            data["bonds", "dist"] = bond_dist.to(dtype)
        if bond_type is not None:
            data["bonds", "type"] = bond_type.long()
        
        # Angle topology (separate i/j/k indices)
        if angle_i is not None:
            data["angles", "i"] = angle_i.long()
        if angle_j is not None:
            data["angles", "j"] = angle_j.long()
        if angle_k is not None:
            data["angles", "k"] = angle_k.long()
        if angle_theta is not None:
            data["angles", "theta"] = angle_theta.to(dtype)
        if angle_type is not None:
            data["angles", "type"] = angle_type.long()
        
        # Dihedral topology (separate i/j/k/l indices)
        if dihedral_i is not None:
            data["dihedrals", "i"] = dihedral_i.long()
        if dihedral_j is not None:
            data["dihedrals", "j"] = dihedral_j.long()
        if dihedral_k is not None:
            data["dihedrals", "k"] = dihedral_k.long()
        if dihedral_l is not None:
            data["dihedrals", "l"] = dihedral_l.long()
        if dihedral_phi is not None:
            data["dihedrals", "phi"] = dihedral_phi.to(dtype)
        if dihedral_type is not None:
            data["dihedrals", "type"] = dihedral_type.long()
        
        # Improper topology (separate i/j/k/l indices)
        if improper_i is not None:
            data["impropers", "i"] = improper_i.long()
        if improper_j is not None:
            data["impropers", "j"] = improper_j.long()
        if improper_k is not None:
            data["impropers", "k"] = improper_k.long()
        if improper_l is not None:
            data["impropers", "l"] = improper_l.long()
        if improper_type is not None:
            data["impropers", "type"] = improper_type.long()
        
        # Graph metadata
        if num_atoms is not None:
            data["graph", "num_atoms"] = num_atoms.long()
        
        # Molecular targets (target.* namespace, no y_ prefix)
        if energy is not None:
            data["target", "energy"] = energy.to(dtype)
        if dipole is not None:
            data["target", "dipole"] = dipole.to(dtype)
        if stress is not None:
            data["target", "stress"] = stress.to(dtype)
        
        # Additional fields
        data.update(kwargs)
        
        return cls(data, batch_size=[])
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms."""
        return len(self["atoms", "z"])
    
    @property
    def num_molecules(self) -> int:
        """Number of molecules."""
        return int(self["graph", "batch"].max()) + 1
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds."""
        try:
            return len(self["bonds", "i"])
        except KeyError:
            return 0
    
    @property
    def num_angles(self) -> int:
        """Number of angles."""
        try:
            return len(self["angles", "i"])
        except KeyError:
            return 0
    
    @property
    def num_dihedrals(self) -> int:
        """Number of dihedrals."""
        try:
            return len(self["dihedrals", "i"])
        except KeyError:
            return 0
