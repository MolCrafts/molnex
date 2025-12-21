"""Test TensorDict native nested access for Frame protocol."""

import torch
import pytest
from molpot.data.atomic_td import AtomicTD


def test_tensordict_nested_access():
    """Test that TensorDict natively supports nested access."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
    )
    
    # TensorDict native nested access
    x_nested = atomic_td["atoms"]["x"]
    assert x_nested.shape == (3, 3)
    assert torch.allclose(x_nested[0], torch.tensor([0.0, 0.0, 0.0]))
    
    # Tuple access still works
    x_tuple = atomic_td["atoms", "x"]
    assert torch.equal(x_nested, x_tuple)


def test_tensordict_nested_with_types():
    """Test nested access with type fields."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        atom_type=torch.tensor([0, 1]),
    )
    
    # Access types via nested dict
    types_nested = atomic_td["atoms"]["type"]
    assert torch.equal(types_nested, torch.tensor([0, 1]))
    
    # Tuple access
    types_tuple = atomic_td["atoms", "type"]
    assert torch.equal(types_nested, types_tuple)


def test_tensordict_bonds_nested():
    """Test nested access for bonds."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1, 1]),
        x=torch.randn(3, 3),
        batch=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        bond_type=torch.tensor([0, 0]),
    )
    
    # Nested access to bonds
    bond_indices = atomic_td["bonds"]["i"]
    assert bond_indices.shape == (2, 2)
    
    bond_types = atomic_td["bonds"]["type"]
    assert torch.equal(bond_types, torch.tensor([0, 0]))


def test_tensordict_graph_nested():
    """Test nested access for graph fields."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1]),
        x=torch.randn(2, 3),
        batch=torch.tensor([0, 0]),
    )
    
    # Access batch via nested dict
    batch = atomic_td["graph"]["batch"]
    assert torch.equal(batch, torch.tensor([0, 0]))


def test_tensordict_set_nested():
    """Test setting values via nested access."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6]),
        x=torch.randn(1, 3),
        batch=torch.tensor([0]),
    )
    
    # Set via nested access
    atomic_td["atoms"]["mass"] = torch.tensor([12.01])
    
    # Verify via both access methods
    assert torch.equal(atomic_td["atoms"]["mass"], torch.tensor([12.01]))
    assert torch.equal(atomic_td["atoms", "mass"], torch.tensor([12.01]))


def test_tensordict_missing_field():
    """Test error handling for missing fields."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6]),
        x=torch.randn(1, 3),
        batch=torch.tensor([0]),
    )
    
    # Should raise KeyError for missing field
    with pytest.raises(KeyError):
        _ = atomic_td["atoms"]["nonexistent"]
