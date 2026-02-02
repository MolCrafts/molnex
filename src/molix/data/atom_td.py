from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class AtomTD:
    """Atomistic data container for molecular systems.
    
    Attributes:
        Z: Atomic numbers [N] (int64)
        xyz: Atomic positions [N, 3] (float32)
        batch: Molecule indices [N] (int64)
        v: Atomic velocities [N, 3] (float32, optional)
        f: Atomic forces [N, 3] (float32, optional, target)
        q: Atomic charges [N] (float32, optional)
        atom_type: Atom types [N] (int64, optional)
        
        bond_i: Bond source indices [E] (int64, optional)
        bond_j: Bond target indices [E] (int64, optional)
        bond_vec: Bond vectors [E, 3] (float32, optional)
        bond_dist: Bond distances [E] (float32, optional)
        bond_type: Bond types [E] (int64, optional)
        
        angle_i: Angle first atom [A] (int64, optional)
        angle_j: Angle center atom [A] (int64, optional)
        angle_k: Angle third atom [A] (int64, optional)
        angle_theta: Angle values [A] (float32, optional)
        angle_type: Angle types [A] (int64, optional)
        
        dihedral_i: Dihedral first atom [D] (int64, optional)
        dihedral_j: Dihedral second atom [D] (int64, optional)
        dihedral_k: Dihedral third atom [D] (int64, optional)
        dihedral_l: Dihedral fourth atom [D] (int64, optional)
        dihedral_phi: Dihedral values [D] (float32, optional)
        dihedral_type: Dihedral types [D] (int64, optional)
        
        num_atoms: Atoms per molecule [B] (int64, optional)
        energy: Molecular energy [B] (float32, optional)
        dipole: Dipole moment [B, 3] (float32, optional)
        stress: Stress tensor [B, 3, 3] (float32, optional)
        
        extra: dict[str, torch.Tensor] = None
    """
    Z: torch.Tensor
    xyz: torch.Tensor
    batch: torch.Tensor
    v: torch.Tensor | None = None
    f: torch.Tensor | None = None
    q: torch.Tensor | None = None
    atom_type: torch.Tensor | None = None
    bond_i: torch.Tensor | None = None
    bond_j: torch.Tensor | None = None
    bond_vec: torch.Tensor | None = None
    bond_dist: torch.Tensor | None = None
    bond_type: torch.Tensor | None = None
    angle_i: torch.Tensor | None = None
    angle_j: torch.Tensor | None = None
    angle_k: torch.Tensor | None = None
    angle_theta: torch.Tensor | None = None
    angle_type: torch.Tensor | None = None
    dihedral_i: torch.Tensor | None = None
    dihedral_j: torch.Tensor | None = None
    dihedral_k: torch.Tensor | None = None
    dihedral_l: torch.Tensor | None = None
    dihedral_phi: torch.Tensor | None = None
    dihedral_type: torch.Tensor | None = None
    num_atoms: torch.Tensor | None = None
    energy: torch.Tensor | None = None
    dipole: torch.Tensor | None = None
    stress: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] | None = None

    def to_model_kwargs(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Convert to keyword arguments for MACE/ScaleShiftMACE models."""
        edge_index = None
        if self.bond_i is not None and self.bond_j is not None:
            edge_index = torch.stack([self.bond_i, self.bond_j], dim=0)

        kwargs = {
            "node_attrs": {"Z": self.Z},
            "pos": self.xyz,
            "bond_dist": self.bond_dist,
            "bond_diff": self.bond_vec,
            "edge_index": edge_index,
            "batch": self.batch,
        }
        
        # Filter out None values
        return {k: v for k, v in kwargs.items() if v is not None}

    @classmethod
    def create(
        cls,
        Z: torch.Tensor,
        xyz: torch.Tensor,
        batch: torch.Tensor,
        v: torch.Tensor | None = None,
        f: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
        atom_type: torch.Tensor | None = None,
        bond_i: torch.Tensor | None = None,
        bond_j: torch.Tensor | None = None,
        bond_vec: torch.Tensor | None = None,
        bond_dist: torch.Tensor | None = None,
        bond_type: torch.Tensor | None = None,
        angle_i: torch.Tensor | None = None,
        angle_j: torch.Tensor | None = None,
        angle_k: torch.Tensor | None = None,
        angle_theta: torch.Tensor | None = None,
        angle_type: torch.Tensor | None = None,
        dihedral_i: torch.Tensor | None = None,
        dihedral_j: torch.Tensor | None = None,
        dihedral_k: torch.Tensor | None = None,
        dihedral_l: torch.Tensor | None = None,
        dihedral_phi: torch.Tensor | None = None,
        dihedral_type: torch.Tensor | None = None,
        num_atoms: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        dipole: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        **kwargs
    ) -> "AtomTD":
        return cls(
            Z=Z.long(),
            xyz=xyz.float(),
            batch=batch.long(),
            v=v.float() if v is not None else None,
            f=f.float() if f is not None else None,
            q=q.float() if q is not None else None,
            atom_type=atom_type.long() if atom_type is not None else None,
            bond_i=bond_i.long() if bond_i is not None else None,
            bond_j=bond_j.long() if bond_j is not None else None,
            bond_vec=bond_vec.float() if bond_vec is not None else None,
            bond_dist=bond_dist.float() if bond_dist is not None else None,
            bond_type=bond_type.long() if bond_type is not None else None,
            angle_i=angle_i.long() if angle_i is not None else None,
            angle_j=angle_j.long() if angle_j is not None else None,
            angle_k=angle_k.long() if angle_k is not None else None,
            angle_theta=angle_theta.float() if angle_theta is not None else None,
            angle_type=angle_type.long() if angle_type is not None else None,
            dihedral_i=dihedral_i.long() if dihedral_i is not None else None,
            dihedral_j=dihedral_j.long() if dihedral_j is not None else None,
            dihedral_k=dihedral_k.long() if dihedral_k is not None else None,
            dihedral_l=dihedral_l.long() if dihedral_l is not None else None,
            dihedral_phi=dihedral_phi.float() if dihedral_phi is not None else None,
            dihedral_type=dihedral_type.long() if dihedral_type is not None else None,
            num_atoms=num_atoms.long() if num_atoms is not None else None,
            energy=energy.float() if energy is not None else None,
            dipole=dipole.float() if dipole is not None else None,
            stress=stress.float() if stress is not None else None,
            extra=kwargs if kwargs else None
        )
