"""Transformer block with NestedTensor support.

Follows PyTorch tutorial: PreNorm + MHA + FFN + residuals.
Uses F.scaled_dot_product_attention directly (no wrapper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase


class TransformerBlock(TensorDictModuleBase):
    """Single Transformer block with NestedTensor support.
    
    Architecture (PreNorm):
        x → LayerNorm → MultiHeadAttention → Residual
          → LayerNorm → FFN → Residual
    
    Uses F.scaled_dot_product_attention directly for attention computation.
    Supports NestedTensor inputs for variable-length sequences.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        
    Example:
        >>> block = TransformerBlock(d_model=128, nhead=4)
        >>> h_nt = torch.nested.nested_tensor([
        ...     torch.randn(3, 128),
        ...     torch.randn(5, 128),
        ... ])
        >>> td = TensorDict({("atoms", "h"): h_nt}, batch_size=[])
        >>> td = block(td)
        >>> td["atoms", "h"].is_nested
        True
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        # Initialize TensorDictModuleBase FIRST
        super().__init__()
        
        # Set in_keys and out_keys as attributes
        self.in_keys = [("atoms", "h")]
        self.out_keys = [("atoms", "h")]
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_head = d_model // nhead
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.dropout = dropout
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through Transformer block.
        
        Args:
            td: TensorDict with ("atoms", "h") NestedTensor [B, (Li), d_model]
            
        Returns:
            TensorDict with updated ("atoms", "h")
        """
        h_nt = td["atoms", "h"]  # NestedTensor [B, (Li), d_model]
        
        # Self-attention with PreNorm
        h_norm = self.norm1(h_nt)
        h_nt = h_nt + self._self_attention(h_norm)
        
        # FFN with PreNorm
        h_norm = self.norm2(h_nt)
        h_nt = h_nt + self.ffn(h_norm)
        
        td["atoms", "h"] = h_nt
        return td
    
    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention using F.scaled_dot_product_attention.
        
        Args:
            x: NestedTensor [B, (Li), d_model]
            
        Returns:
            Attention output [B, (Li), d_model]
        """
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, (Li), d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: [B, (Li), d_model] → [B, (Li), nhead, d_head]
        # Then transpose to [B, nhead, (Li), d_head]
        B = x.size(0)
        
        q = q.reshape(B, -1, self.nhead, self.d_head).transpose(1, 2).contiguous()
        k = k.reshape(B, -1, self.nhead, self.d_head).transpose(1, 2).contiguous()
        v = v.reshape(B, -1, self.nhead, self.d_head).transpose(1, 2).contiguous()
        
        # Scaled dot-product attention (supports NestedTensor directly)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # NestedTensor handles variable lengths automatically
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, nhead, (Li), d_head]
        
        # Reshape back: [B, nhead, (Li), d_head] → [B, (Li), d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, -1, self.d_model)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        return attn_out
    
    def __repr__(self) -> str:
        return (
            f"TransformerBlock(d_model={self.d_model}, nhead={self.nhead}, "
            f"dim_feedforward={self.dim_feedforward}, dropout={self.dropout})"
        )


class TransformerEncoder(TensorDictModuleBase):
    """Stack of N TransformerBlocks.
    
    Args:
        num_layers: Number of Transformer blocks
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        
    Example:
        >>> encoder = TransformerEncoder(num_layers=6, d_model=128, nhead=4)
        >>> h_nt = torch.nested.nested_tensor([
        ...     torch.randn(3, 128),
        ...     torch.randn(5, 128),
        ... ])
        >>> td = TensorDict({("atoms", "h"): h_nt}, batch_size=[])
        >>> td = encoder(td)
        >>> td["rep", "h"].is_nested
        True
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        # Initialize TensorDictModuleBase FIRST
        super().__init__()
        
        # Set in_keys and out_keys as attributes
        self.in_keys = [("atoms", "h")]
        self.out_keys = [("rep", "h")]
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Create transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through all Transformer blocks.
        
        Args:
            td: TensorDict with ("atoms", "h")
            
        Returns:
            TensorDict with ("rep", "h")
        """
        for layer in self.layers:
            td = layer(td)
        
        # Move to rep namespace
        td["rep", "h"] = td["atoms", "h"]
        return td
    
    def __repr__(self) -> str:
        return (
            f"TransformerEncoder(num_layers={self.num_layers}, "
            f"d_model={self.d_model})"
        )
