"""AtomicTD: TensorDict-based atomistic data structure.

AtomicTD inherits from TensorDict and uses molpy-aligned field names:
- ("atoms", "z"): Atomic numbers
- ("atoms", "x"): Atomic positions
- ("atoms", "y"): Atomic targets (optional)
- ("bonds", "i"): Bond/edge indices
- ("bonds", "vec"): Bond/edge vectors
- ("graph", "batch"): Molecule/batch indices
"""

from typing import Optional, List, Dict, Any

import torch
from tensordict import TensorDict


class AtomicTD(TensorDict):
    """Atomistic TensorDict with molpy-aligned field names.
    
    Field naming convention (hierarchical):
    - ("atoms", "z"): Atomic numbers [N] (LongTensor)
    - ("atoms", "x"): Atomic positions [N, 3] (FloatTensor)
    - ("atoms", "y"): Atomic targets [N] (FloatTensor, optional)
    - ("atoms", "type"): Atom types for potentials [N] (LongTensor, optional)
    - ("graph", "batch"): Molecule/batch indices [N] (LongTensor)
    - ("bonds", "i"): Edge indices [2, E] (LongTensor, optional)
    - ("bonds", "vec"): Edge vectors [E, 3] (FloatTensor, optional)
    - ("bonds", "type"): Bond types for potentials [E] (LongTensor, optional)
    - ("angles", "i"): Angle indices [3, A] (LongTensor, optional)
    - ("angles", "type"): Angle types for potentials [A] (LongTensor, optional)
    - ("dihedrals", "i"): Dihedral indices [4, D] (LongTensor, optional)
    - ("dihedrals", "type"): Dihedral types for potentials [D] (LongTensor, optional)
    - ("mol", "y_energy"): Molecular energy targets [B] (FloatTensor, optional)
    
    TensorDict natively supports nested access:
    >>> atomic_td["atoms"]["x"]  # Works out of the box!
    >>> atomic_td["atoms", "x"]  # Also works
    
    Example:
        >>> atomic_td = AtomicTD.create(
        ...     z=torch.tensor([1, 1, 8]),
        ...     x=torch.randn(3, 3),
        ...     batch=torch.tensor([0, 0, 0]),
        ...     y_energy=torch.tensor([0.5])
        ... )
    """
    
    @classmethod
    def create(
        cls,
        z: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
        y_energy: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_vec: Optional[torch.Tensor] = None,
        atom_type: Optional[torch.Tensor] = None,
        bond_type: Optional[torch.Tensor] = None,
        angle_index: Optional[torch.Tensor] = None,
        angle_type: Optional[torch.Tensor] = None,
        dihedral_index: Optional[torch.Tensor] = None,
        dihedral_type: Optional[torch.Tensor] = None,
        **kwargs
    ) -> "AtomicTD":
        """Create AtomicTD with molpy-aligned field names.
        
        Args:
            z: Atomic numbers [N]
            x: Atomic positions [N, 3]
            batch: Molecule/batch indices [N]
            y_energy: Molecular energy targets [B] (optional)
            edge_index: Edge indices [2, E] (optional)
            edge_vec: Edge vectors [E, 3] (optional)
            atom_type: Atom types for potentials [N] (optional)
            bond_type: Bond types for potentials [E] (optional)
            angle_index: Angle indices [3, A] (optional)
            angle_type: Angle types for potentials [A] (optional)
            dihedral_index: Dihedral indices [4, D] (optional)
            dihedral_type: Dihedral types for potentials [D] (optional)
            **kwargs: Additional fields
            
        Returns:
            AtomicTD instance
        """
        data = {
            ("atoms", "z"): z,
            ("atoms", "x"): x,
            ("graph", "batch"): batch,
        }
        
        # Add optional molecular targets
        data["mol", "y_energy"] = y_energy or torch.tensor([])
        
        # Add optional edge data
        data["bonds", "i"] = edge_index or torch.zeros((2, 0), dtype=torch.long)
        data["bonds", "vec"] = edge_vec or torch.zeros((0, 3), dtype=x.dtype)
        
        # Add optional topology type data for classic potentials
        if atom_type is not None:
            data["atoms", "type"] = atom_type
        if bond_type is not None:
            data["bonds", "type"] = bond_type
        if angle_index is not None:
            data["angles", "i"] = angle_index
        if angle_type is not None:
            data["angles", "type"] = angle_type
        if dihedral_index is not None:
            data["dihedrals", "i"] = dihedral_index
        if dihedral_type is not None:
            data["dihedrals", "type"] = dihedral_type
        
        # Add any additional fields
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
    def num_edges(self) -> int:
        """Number of edges."""
        return self["bonds", "i"].size(1)


class AtomicCollator:
    """Collator for batching AtomicTD instances.
    
    Concatenates atomic properties and creates batch indices.
    """
    
    def __call__(self, batch_list: List[Dict[str, torch.Tensor]]) -> AtomicTD:
        """Collate list of atomic data dictionaries.
        
        Args:
            batch_list: List of dicts with keys: z, x, y_energy, etc.
            
        Returns:
            Single AtomicTD with batched data
        """
        # Concatenate atomic properties
        z_list = []
        x_list = []
        batch_indices = []
        
        # Stack molecular properties
        y_energy_list = []
        
        for mol_idx, mol_data in enumerate(batch_list):
            num_atoms = len(mol_data["z"])
            
            z_list.append(mol_data["z"])
            x_list.append(mol_data["x"])
            batch_indices.append(torch.full((num_atoms,), mol_idx, dtype=torch.long))
            
            y_energy = mol_data.get("y_energy")
            y_energy_list.append(y_energy or torch.tensor([0.0]))
        
        # Concatenate
        z = torch.cat(z_list, dim=0)
        x = torch.cat(x_list, dim=0)
        batch = torch.cat(batch_indices, dim=0)
        y_energy = torch.stack(y_energy_list, dim=0)
        
        return AtomicTD.create(z=z, x=x, batch=batch, y_energy=y_energy)
