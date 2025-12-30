"""TensorDictModule wrappers for segment operations."""

from typing import Optional, Tuple

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from ..binding.scatter.segment_coo import (
    segment_sum_coo as _segment_sum_coo,
    segment_mean_coo as _segment_mean_coo,
    segment_max_coo as _segment_max_coo,
    segment_min_coo as _segment_min_coo,
)
from ..binding.scatter.segment_csr import (
    segment_sum_csr as _segment_sum_csr,
    segment_mean_csr as _segment_mean_csr,
    segment_max_csr as _segment_max_csr,
    segment_min_csr as _segment_min_csr,
)


class SegmentSumCOO(TensorDictModuleBase):
    """TensorDictModule wrapper for segment_sum_coo operation.
    
    Sums values from src into segments specified by sorted index (COO format).
    
    Args:
        dim_size: Output size (default: None, auto-inferred)
        in_keys: Input keys for src and index
        out_keys: Output key for result
        
    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from op.modules import SegmentSumCOO
        >>> 
        >>> segment = SegmentSumCOO(in_keys=[("data", "src"), ("data", "index")],
        ...                         out_keys=[("data", "out")])
        >>> src = torch.randn(10, 64)
        >>> index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # Sorted
        >>> td = TensorDict({("data", "src"): src, ("data", "index"): index}, batch_size=[])
        >>> td = segment(td)
        >>> td[("data", "out")].shape
        torch.Size([3, 64])
    """
    
    def __init__(
        self,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply segment_sum_coo operation."""
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out = _segment_sum_coo(src, index, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        
        return td


class SegmentMeanCOO(TensorDictModuleBase):
    """TensorDictModule wrapper for segment_mean_coo operation."""
    
    def __init__(
        self,
        dim_size: Optional[int] = None,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        self.dim_size = dim_size
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "index")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply segment_mean_coo operation."""
        src = td[self.in_keys[0]]
        index = td[self.in_keys[1]]
        
        out = _segment_mean_coo(src, index, dim_size=self.dim_size)
        td[self.out_keys[0]] = out
        
        return td


class SegmentSumCSR(TensorDictModuleBase):
    """TensorDictModule wrapper for segment_sum_csr operation.
    
    Sums values from src into segments specified by indptr (CSR format).
    
    Args:
        in_keys: Input keys for src and indptr
        out_keys: Output key for result
        
    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from op.modules import SegmentSumCSR
        >>> 
        >>> segment = SegmentSumCSR(in_keys=[("data", "src"), ("data", "indptr")],
        ...                         out_keys=[("data", "out")])
        >>> src = torch.randn(10, 64)
        >>> indptr = torch.tensor([0, 2, 5, 10])  # 3 segments
        >>> td = TensorDict({("data", "src"): src, ("data", "indptr"): indptr}, batch_size=[])
        >>> td = segment(td)
        >>> td[("data", "out")].shape
        torch.Size([3, 64])
    """
    
    def __init__(
        self,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "indptr")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply segment_sum_csr operation."""
        src = td[self.in_keys[0]]
        indptr = td[self.in_keys[1]]
        
        out = _segment_sum_csr(src, indptr)
        td[self.out_keys[0]] = out
        
        return td


class SegmentMeanCSR(TensorDictModuleBase):
    """TensorDictModule wrapper for segment_mean_csr operation."""
    
    def __init__(
        self,
        in_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
        out_keys: Optional[Tuple[Tuple[str, str], ...]] = None,
    ):
        super().__init__()
        
        if in_keys is None:
            in_keys = [("data", "src"), ("data", "indptr")]
        if out_keys is None:
            out_keys = [("data", "out")]
        
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Apply segment_mean_csr operation."""
        src = td[self.in_keys[0]]
        indptr = td[self.in_keys[1]]
        
        out = _segment_mean_csr(src, indptr)
        td[self.out_keys[0]] = out
        
        return td
