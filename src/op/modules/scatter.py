"""TensorDictModule wrappers for scatter operations."""

from typing import Optional, Tuple

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ..binding.scatter.scatter_fn import (
    scatter_sum as _scatter_sum,
    scatter_mean as _scatter_mean,
    scatter_max as _scatter_max,
    scatter_min as _scatter_min,
)


class ScatterSum(TensorDictModuleBase):
    """TensorDictModule wrapper for scatter_sum operation.
    
    Sums values from src into out at indices specified in index.
    
    Args:
        dim: Dimension along which to scatter (default: -1)
        dim_size: Output size along scatter dimension (default: None, auto-inferred)
        in_keys: Input keys for src and index
        out_keys: Output key for result
        
    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from op.modules import ScatterSum
        >>> 
        >>> scatter = ScatterSum(in_keys=[("data", "src"), ("data", "index")],
        ...                      out_keys=[("data", "out")])
        >>> src = torch.randn(10, 64)
        >>> index = torch.tensor([0, 1, 0, 1, 2, 1, 0, 2, 1, 0])
        >>> td = TensorDict({("data", "src"): src, ("data", "index"): index}, batch_size=[])
        >>> td = scatter(td)
        >>> td[("data", "out")].shape
        torch.Size([3, 64])
    """
    
    def __init__(
        self,
        dim: int = -1,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply scatter_sum operation.
        
        Args:
            td: TensorDict with src and index tensors
            
        Returns:
            TensorDict with scattered result
        """
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out = _scatter_sum(src, index, dim=self.dim, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        
        return td


class ScatterMean(TensorDictModuleBase):
    """TensorDictModule wrapper for scatter_mean operation.
    
    Computes mean of values from src into out at indices specified in index.
    
    Args:
        dim: Dimension along which to scatter (default: -1)
        dim_size: Output size along scatter dimension (default: None, auto-inferred)
        in_keys: Input keys for src and index
        out_keys: Output key for result
    """
    
    def __init__(
        self,
        dim: int = -1,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply scatter_mean operation."""
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out = _scatter_mean(src, index, dim=self.dim, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        
        return td


class ScatterMax(TensorDictModuleBase):
    """TensorDictModule wrapper for scatter_max operation.
    
    Computes max of values from src into out at indices specified in index.
    
    Args:
        dim: Dimension along which to scatter (default: -1)
        dim_size: Output size along scatter dimension (default: None, auto-inferred)
        in_keys: Input keys for src and index
        out_keys: Output keys for result and argmax
    """
    
    def __init__(
        self,
        dim: int = -1,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out"), ("data", "argmax")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply scatter_max operation."""
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out, argmax = _scatter_max(src, index, dim=self.dim, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        if len(self.out_keys) > 1:
            td[self.out_keys[1]] = argmax
        
        return td


class ScatterMin(TensorDictModuleBase):
    """TensorDictModule wrapper for scatter_min operation.
    
    Computes min of values from src into out at indices specified in index.
    
    Args:
        dim: Dimension along which to scatter (default: -1)
        dim_size: Output size along scatter dimension (default: None, auto-inferred)
        in_keys: Input keys for src and index
        out_keys: Output keys for result and argmin
    """
    
    def __init__(
        self,
        dim: int = -1,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out"), ("data", "argmin")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply scatter_min operation."""
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out, argmin = _scatter_min(src, index, dim=self.dim, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        if len(self.out_keys) > 1:
            td[self.out_keys[1]] = argmin
        
        return td
