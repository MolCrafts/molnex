"""Atom embedding module for NestedTensor inputs."""

import torch
import torch.nn as nn
from tensordict import TensorDict


class AtomEmbedding(nn.Module):
    """Embed atomic numbers and positions to d_model using NestedTensor.
    
    Converts atomic numbers and 3D positions (NestedTensor) to learned embeddings
    suitable for Transformer processing. Supports variable-length molecules via
    NestedTensor input/output.
    
    Args:
        num_types: Number of atom types (vocabulary size)
        d_model: Embedding dimension
        use_positions: Whether to include positional encoding (default: True)
        
    Example:
        >>> embedding = AtomEmbedding(num_types=100, d_model=128)
        >>> z_nt = torch.nested.nested_tensor([
        ...     torch.tensor([6, 1, 1]),
        ...     torch.tensor([6, 1, 1, 1, 1]),
        ... ])
        >>> x_nt = torch.nested.nested_tensor([
        ...     torch.randn(3, 3),
        ...     torch.randn(5, 3),
        ... ])
        >>> td = TensorDict({
        ...     ("atoms", "z"): z_nt,
        ...     ("atoms", "x"): x_nt,
        ... }, batch_size=[])
        >>> td = embedding(td)
        >>> td["atoms", "h"].is_nested
        True
        >>> td["atoms", "h"].size(0)  # Batch size
        2
    """
    
    def __init__(
        self,
        num_types: int = 100,
        d_model: int = 128,
        use_positions: bool = True,
    ):
        super().__init__()
        
        self.num_types = num_types
        self.d_model = d_model
        self.use_positions = use_positions
        
        # Atomic number embedding
        self.z_embedding = nn.Embedding(num_types, d_model)
        
        # Positional encoding (if enabled)
        if use_positions:
            self.pos_encode = nn.Sequential(
                nn.Linear(3, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            self.norm = nn.LayerNorm(d_model)
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Embed atomic numbers and positions to hidden states.
        
        Args:
            td: TensorDict with:
                - ("atoms", "z"): NestedTensor [B, (Li)] atomic numbers
                - ("atoms", "x"): NestedTensor [B, (Li), 3] positions (optional)
            
        Returns:
            TensorDict with ("atoms", "h") NestedTensor [B, (Li), d_model]
        """
        z_nt = td["atoms", "z"]  # NestedTensor [B, (Li)]
        
        # Embed atomic numbers
        h_nt = self.z_embedding(z_nt)  # [B, (Li), d_model]
        
        # Add positional encoding if enabled
        if self.use_positions:
            x_nt = td["atoms", "x"]  # NestedTensor [B, (Li), 3]
            x_features = self.pos_encode(x_nt)  # [B, (Li), d_model]
            h_nt = h_nt + x_features
            h_nt = self.norm(h_nt)
        
        td["atoms", "h"] = h_nt
        return td
    
    def __repr__(self) -> str:
        return (
            f"AtomEmbedding(num_types={self.num_types}, "
            f"d_model={self.d_model}, use_positions={self.use_positions})"
        )
