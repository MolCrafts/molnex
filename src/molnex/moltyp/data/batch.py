from dataclasses import dataclass
import torch
from typing import List, Optional
from .data import Data

@dataclass
class Batch(Data):
    """
    Batch object representing a batch of molecules.
    """
    batch: Optional[torch.Tensor] = None # (N_total,) batch index mapping each atom to its molecule index
    ptr: Optional[torch.Tensor] = None   # (B+1,) pointers to start of each molecule in the batch

    @classmethod
    def from_data_list(cls, data_list: List[Data]) -> 'Batch':
        if len(data_list) == 0:
            raise ValueError("Cannot verify batch from empty data list")

        # computes sizes
        num_nodes_list = [d.num_nodes for d in data_list]
        total_nodes = sum(num_nodes_list)
        batch_size = len(data_list)

        # Concatenate node features
        z = torch.cat([d.z for d in data_list], dim=0)
        pos = torch.cat([d.pos for d in data_list], dim=0)
        
        # Handle labels if present
        y = None
        if data_list[0].y is not None:
            # Assumes all have y if the first one does
            y = torch.cat([d.y for d in data_list], dim=0) # type: ignore

        # Handle edges if present
        edge_index = None
        edge_attr = None
        
        if data_list[0].edge_index is not None:
             # Need to increment indices
            edge_indices = []
            edge_attrs = []
            cum_nodes = 0
            for d in data_list:
                if d.edge_index is not None:
                    edge_indices.append(d.edge_index + cum_nodes)
                    if d.edge_attr is not None:
                        edge_attrs.append(d.edge_attr)
                cum_nodes += d.num_nodes
            
            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
            
            if edge_attrs:
                edge_attr = torch.cat(edge_attrs, dim=0)

        # Create batch vector
        batch = torch.repeat_interleave(
            torch.tensor(range(batch_size), dtype=torch.long),
            torch.tensor(num_nodes_list, dtype=torch.long)
        )
        
        # Create ptr vector
        ptr = torch.tensor([0] + num_nodes_list, dtype=torch.long).cumsum(0)

        return cls(
            z=z, 
            pos=pos, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y,
            batch=batch, 
            ptr=ptr
        )

    def to(self, device: str) -> 'Batch':
        self.z = self.z.to(device)
        self.pos = self.pos.to(device)
        self.batch = self.batch.to(device)
        self.ptr = self.ptr.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self
