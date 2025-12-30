"""Collate functions for batching molecular data.

Provides collate_fn implementations for converting list[molpy.Frame] to batched
AtomicTD with NestedTensor support for variable-length molecules.
"""

import torch
from tensordict import TensorDict

from molix.data.atomic_td import AtomicTD, Config


def collate_frames(frames: list, config: Config | None = None) -> AtomicTD:
    """Universal collate function: list[molpy.Frame] -> AtomicTD with NestedTensors.
    
    This is the canonical collate function for batching molecular data. It converts
    a list of molpy.Frame objects into an AtomicTD with NestedTensor fields for
    efficient variable-length batching without padding.
    
    All frames must have the same structure:
        - frame["atoms"]["x"]: positions [Ni, 3]
        - frame["atoms"]["z"]: atomic numbers [Ni]
        - frame["atoms"]["v"]: velocities [Ni, 3] (optional)
    
    Args:
        frames: List of molpy.Frame objects
        config: Optional Config for dtype control
        
    Returns:
        AtomicTD with NestedTensor fields:
            - ("atoms", "x"): NestedTensor [B, (Li), 3]
            - ("atoms", "z"): NestedTensor [B, (Li)]
            - ("atoms", "v"): NestedTensor [B, (Li), 3] (if present)
            - ("graph", "num_atoms"): Tensor [B]
            - ("graph", "batch_size"): int scalar
            
    Example:
        >>> from molpy import Frame
        >>> import numpy as np
        >>> 
        >>> frame1 = Frame(blocks={"atoms": {"x": np.random.randn(3, 3), "z": np.array([6, 1, 1])}})
        >>> frame2 = Frame(blocks={"atoms": {"x": np.random.randn(5, 3), "z": np.array([6, 1, 1, 1, 1])}})
        >>> 
        >>> batch = collate_frames([frame1, frame2])
        >>> batch["atoms", "z"].is_nested
        True
        >>> batch["graph", "num_atoms"].tolist()
        [3, 5]
    """
    if config is None:
        config = Config()
    
    assert len(frames) > 0, "Cannot collate empty list of frames"
    
    # Extract per-molecule data
    x_list = []
    z_list = []
    v_list = []
    has_velocities = False
    
    for frame in frames:
        atoms_block = frame["atoms"]
        
        # Positions (required) - ensure shape is (N, 3)
        x = torch.tensor(atoms_block["x"], dtype=config.dtype)
        if x.ndim == 1:
            # If 1D, reshape to (N//3, 3)
            x = x.reshape(-1, 3)
        x_list.append(x)
        
        # Atomic numbers (required)
        z = torch.tensor(atoms_block["z"], dtype=torch.long)
        z_list.append(z)
        
        # Velocities (optional)
        if "v" in atoms_block.keys():
            v = torch.tensor(atoms_block["v"], dtype=config.dtype)
            if v.ndim == 1:
                v = v.reshape(-1, 3)
            v_list.append(v)
            has_velocities = True
    
    # Create NestedTensors
    x_nt = torch.nested.nested_tensor(x_list, dtype=config.dtype)
    z_nt = torch.nested.nested_tensor(z_list, dtype=torch.long)
    
    # Metadata
    num_atoms = torch.tensor([len(z) for z in z_list], dtype=torch.long)
    batch_size = len(frames)
    
    # Build data dict
    data = {
        ("atoms", "x"): x_nt,
        ("atoms", "z"): z_nt,
        ("graph", "num_atoms"): num_atoms,
        ("graph", "batch_size"): torch.tensor(batch_size, dtype=torch.long),
    }
    
    # Add velocities if present
    if has_velocities:
        v_nt = torch.nested.nested_tensor(v_list, dtype=config.dtype)
        data[("atoms", "v")] = v_nt
    
    # Extract metadata (box, pbc, etc.) if present
    boxes = []
    pbcs = []
    for frame in frames:
        if hasattr(frame, 'metadata'):
            if "box" in frame.metadata:
                boxes.append(frame.metadata["box"])
            if "pbc" in frame.metadata:
                pbcs.append(frame.metadata["pbc"])
    
    if boxes:
        data[("meta", "box")] = boxes  # Keep as list (not tensorized)
    if pbcs:
        data[("meta", "pbc")] = pbcs  # Keep as list
    
    # Extract targets from metadata["target"] if present
    # Collect all unique target keys across frames
    target_keys = set()
    for frame in frames:
        if hasattr(frame, 'metadata') and "target" in frame.metadata:
            target_keys.update(frame.metadata["target"].keys())
    
    # For each target key, extract values
    for key in target_keys:
        values = []
        is_per_atom = False
        
        for frame in frames:
            if hasattr(frame, 'metadata') and "target" in frame.metadata:
                value = frame.metadata["target"].get(key)
                if value is not None:
                    # Convert to tensor
                    value_tensor = torch.tensor(value, dtype=config.dtype if not isinstance(value, (int, str)) else torch.float32)
                    
                    # Check if per-atom (shape matches number of atoms)
                    if value_tensor.ndim >= 1 and len(value_tensor) == len(z_list[len(values)]):
                        is_per_atom = True
                    
                    values.append(value_tensor)
        
        if values:
            # Per-atom targets use NestedTensor, scalar targets use regular tensor
            if is_per_atom and any(v.ndim >= 1 for v in values):
                data[("target", key)] = torch.nested.nested_tensor(values, dtype=config.dtype)
            else:
                # Stack scalar values
                data[("target", key)] = torch.stack(values) if values[0].ndim == 0 else torch.stack(values)
    
    return AtomicTD(data, batch_size=[])


# Alias for backward compatibility
nested_collate_fn = collate_frames
