"""TensorDictModule wrappers for neighbor list computation."""

from typing import Optional, Tuple

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ..binding.locality.neighbors import get_neighbor_pairs as _get_neighbor_pairs


class GetNeighborPairs(TensorDictModuleBase):
    """TensorDictModule wrapper for get_neighbor_pairs operator.
    
    Computes neighbor pairs within a cutoff distance and stores results in TensorDict.
    
    Args:
        cutoff: Maximum distance between atom pairs
        max_num_pairs: Maximum number of pairs. If -1 (default), all pairs are returned
        check_errors: Whether to check for errors (incompatible with CUDA graphs)
        in_keys: Input keys for positions and optional box vectors
        out_keys: Output keys for neighbors, deltas, distances, and num_found
        
    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from op.modules import GetNeighborPairs
        >>> 
        >>> # Create module
        >>> neighbor_fn = GetNeighborPairs(cutoff=3.0)
        >>> 
        >>> # Create input
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> td = TensorDict({("atoms", "x"): positions}, batch_size=[])
        >>> 
        >>> # Compute neighbors
        >>> td = neighbor_fn(td)
        >>> td[("bonds", "i")].shape
        torch.Size([2, 3])
    """
    
    def __init__(
        self,
        cutoff: float,
        max_num_pairs: int = -1,
        check_errors: bool = False,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.check_errors = check_errors
        
        # Default keys following AtomicTD convention
        if in_keys is None:
            in_keys = [("atoms", "x"), ("meta", "box")]
        if out_keys is None:
            out_keys = [
                ("bonds", "neighbors"),
                ("bonds", "vec"),
                ("bonds", "dist"),
                ("bonds", "num_found"),
            ]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Compute neighbor pairs from atomic positions.
        
        Args:
            td: TensorDict with:
                - ("atoms", "x"): Positions [N, 3]
                - ("meta", "box"): Optional box vectors [3, 3]
                
        Returns:
            TensorDict with:
                - ("bonds", "neighbors"): Neighbor indices [2, num_pairs]
                - ("bonds", "vec"): Displacement vectors [num_pairs, 3]
                - ("bonds", "dist"): Distances [num_pairs]
                - ("bonds", "num_found"): Number of pairs found [1]
        """
        positions = td[self.in_keys[0]]
        
        # Get box vectors if available
        box_vectors = None
        if len(self.in_keys) > 1 and self.in_keys[1] in td.keys(include_nested=True):
            box_vectors = td[self.in_keys[1]]
        
        # Compute neighbors
        neighbors, deltas, distances, num_found = _get_neighbor_pairs(
            positions=positions,
            cutoff=self.cutoff,
            max_num_pairs=self.max_num_pairs,
            box_vectors=box_vectors,
            check_errors=self.check_errors,
        )
        
        # Store results
        td[self.out_keys[0]] = neighbors
        td[self.out_keys[1]] = deltas
        td[self.out_keys[2]] = distances
        td[self.out_keys[3]] = num_found
        
        return td
    
    def __repr__(self) -> str:
        return (
            f"GetNeighborPairs(cutoff={self.cutoff}, "
            f"max_num_pairs={self.max_num_pairs}, "
            f"check_errors={self.check_errors})"
        )
