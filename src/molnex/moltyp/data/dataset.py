from typing import List, Optional, Any
import torch
import numpy as np
from molpy.core.element import Element
from .data import Data

class MoleculeDataset:
    """
    In-memory dataset for small-to-medium datasets.
    """
    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    @classmethod
    def from_molpy(cls, systems: List[Any], labeler=None) -> 'MoleculeDataset':
        """
        Create dataset from list of molpy.core.Atomistic objects.
        
        Args:
            systems: List of Atomistic objects (duck-typed)
            labeler: Optional callable to assign labels to atoms (e.g. ProxyLabeler)
        """
        data_list = []
        for system in systems:
            # 1. Atomic Numbers
            # Assuming system.atoms gives entities with 'symbol'
            atoms = list(system.atoms)
            z_vals = []
            for a in atoms:
                symbol = a.get("symbol", "C")
                try:
                    z = Element.get_atomic_number(symbol)
                except KeyError:
                     # Fallback or unknown
                    z = 0
                z_vals.append(z)
                
            z = torch.tensor(z_vals, dtype=torch.long)
            
            # 2. Coordinates
            # system.xyz should return numpy array or list of lists
            coords = system.xyz
            pos = torch.tensor(coords, dtype=torch.float32)
            
            # 3. Topology (if available)
            edge_index = None
            if hasattr(system, "bonds"):
                bonds = list(system.bonds)
                if bonds:
                    edge_list = []
                    # bond.itom and bond.jtom are atom objects
                    # Need to map them to indices. 
                    # Assuming stable list order from system.atoms
                    atom_to_idx = {id(a): i for i, a in enumerate(atoms)}
                    
                    for bond in bonds:
                        idx_i = atom_to_idx.get(id(bond.itom))
                        idx_j = atom_to_idx.get(id(bond.jtom))
                        
                        if idx_i is not None and idx_j is not None:
                            edge_list.append([idx_i, idx_j])
                            edge_list.append([idx_j, idx_i]) # Undirected
                    
                    if edge_list:
                        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # 4. Labels (if labeler provided)
            y = None
            if labeler is not None:
                labels = labeler(system) # Should return list of ints or tensor
                if isinstance(labels, list):
                    y = torch.tensor(labels, dtype=torch.long)
                else:
                    y = labels
            
            data_list.append(Data(z=z, pos=pos, edge_index=edge_index, y=y))
            
        return cls(data_list)
        
    def __getitem__(self, idx) -> Data:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)
