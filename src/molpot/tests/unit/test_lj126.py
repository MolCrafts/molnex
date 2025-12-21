"""Unit tests for LJ126 potential."""

import torch
import pytest
from molpot.potentials import LJ126
from molpot.data.atomic_td import AtomicTD


def test_lj126_initialization():
    """Test LJ126 initialization with valid parameters."""
    epsilon = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    sigma = torch.tensor([[3.0, 3.5], [3.5, 4.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    assert lj.epsilon.shape == (2, 2)
    assert lj.sigma.shape == (2, 2)
    assert lj.cutoff == 10.0


def test_lj126_shape_validation():
    """Test that LJ126 validates parameter shapes."""
    epsilon = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    sigma = torch.tensor([3.0, 4.0])  # Wrong shape
    
    with pytest.raises(ValueError, match="must have same shape"):
        LJ126(epsilon=epsilon, sigma=sigma)


def test_lj126_single_type():
    """Test LJ126 with single atom type."""
    # Single type system
    epsilon = torch.tensor([[1.0]])
    sigma = torch.tensor([[3.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    # Two atoms at distance 3.0 (sigma)
    # At r = sigma, LJ potential should be 0
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        atom_type=torch.tensor([0, 0]),
    )
    
    energy = lj(atomic_td)
    
    # At r = sigma, E = 4*epsilon*[(sigma/sigma)^12 - (sigma/sigma)^6] = 4*1*[1 - 1] = 0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_lj126_multi_type():
    """Test LJ126 with multiple atom types."""
    # Two types with different parameters
    epsilon = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
    sigma = torch.tensor([[3.0, 3.5], [3.5, 4.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    # Three atoms: type 0, type 1, type 1
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 8]),
        x=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        atom_type=torch.tensor([0, 1, 1]),
    )
    
    energy = lj(atomic_td)
    
    # Should compute energy without error
    assert energy.ndim == 0  # Scalar
    assert not torch.isnan(energy)


def test_lj126_gradient():
    """Test that LJ126 energy is differentiable (for force computation)."""
    epsilon = torch.tensor([[1.0]])
    sigma = torch.tensor([[3.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    # Create positions that require gradients
    x = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], requires_grad=True)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=x,
        batch=torch.tensor([0, 0]),
        atom_type=torch.tensor([0, 0]),
    )
    
    energy = lj(atomic_td)
    
    # Compute gradient (forces = -dE/dx)
    energy.backward()
    
    assert x.grad is not None
    assert x.grad.shape == (2, 3)


def test_lj126_empty_system():
    """Test LJ126 with no atoms."""
    epsilon = torch.tensor([[1.0]])
    sigma = torch.tensor([[3.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([]),
        x=torch.zeros((0, 3)),
        batch=torch.tensor([]),
        atom_type=torch.tensor([]),
    )
    
    energy = lj(atomic_td)
    
    # Empty system should have zero energy
    assert torch.isclose(energy, torch.tensor(0.0))


def test_lj126_cutoff():
    """Test that LJ126 respects cutoff radius."""
    epsilon = torch.tensor([[1.0]])
    sigma = torch.tensor([[3.0]])
    
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=5.0)
    
    # Two atoms beyond cutoff
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        atom_type=torch.tensor([0, 0]),
    )
    
    energy = lj(atomic_td)
    
    # Beyond cutoff, no interaction
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)
