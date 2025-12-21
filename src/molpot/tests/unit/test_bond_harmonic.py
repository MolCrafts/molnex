"""Unit tests for BondHarmonic potential."""

import torch
import pytest
from molpot.potentials import BondHarmonic
from molpot.data.atomic_td import AtomicTD


def test_bond_harmonic_initialization():
    """Test BondHarmonic initialization with valid parameters."""
    k = torch.tensor([100.0, 200.0])
    r0 = torch.tensor([1.5, 1.2])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    assert bond.k.shape == (2,)
    assert bond.r0.shape == (2,)


def test_bond_harmonic_shape_validation():
    """Test that BondHarmonic validates parameter shapes."""
    k = torch.tensor([100.0, 200.0])
    r0 = torch.tensor([1.5])  # Wrong shape
    
    with pytest.raises(ValueError, match="must have same shape"):
        BondHarmonic(k=k, r0=r0)


def test_bond_harmonic_single_bond():
    """Test BondHarmonic with single bond at equilibrium."""
    k = torch.tensor([100.0])
    r0 = torch.tensor([1.5])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    # Two atoms at equilibrium distance
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0], [1]]),
        bond_type=torch.tensor([0]),
    )
    
    energy = bond(atomic_td)
    
    # At equilibrium, energy should be 0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_bond_harmonic_stretched():
    """Test BondHarmonic with stretched bond."""
    k = torch.tensor([100.0])
    r0 = torch.tensor([1.5])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    # Two atoms stretched by 0.5
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0], [1]]),
        bond_type=torch.tensor([0]),
    )
    
    energy = bond(atomic_td)
    
    # E = 0.5 * k * (r - r0)^2 = 0.5 * 100 * (2.0 - 1.5)^2 = 0.5 * 100 * 0.25 = 12.5
    expected = 0.5 * 100.0 * (2.0 - 1.5) ** 2
    assert torch.isclose(energy, torch.tensor(expected), atol=1e-5)


def test_bond_harmonic_multiple_bonds():
    """Test BondHarmonic with multiple bonds of different types."""
    k = torch.tensor([100.0, 200.0])
    r0 = torch.tensor([1.5, 1.2])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    # Three atoms with two bonds
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.7, 0.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        bond_type=torch.tensor([0, 1]),
    )
    
    energy = bond(atomic_td)
    
    # Bond 0: r=1.5, r0=1.5, E=0
    # Bond 1: r=1.2, r0=1.2, E=0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_bond_harmonic_gradient():
    """Test that BondHarmonic energy is differentiable."""
    k = torch.tensor([100.0])
    r0 = torch.tensor([1.5])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=x,
        batch=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0], [1]]),
        bond_type=torch.tensor([0]),
    )
    
    energy = bond(atomic_td)
    energy.backward()
    
    assert x.grad is not None
    assert x.grad.shape == (2, 3)


def test_bond_harmonic_empty():
    """Test BondHarmonic with no bonds."""
    k = torch.tensor([100.0])
    r0 = torch.tensor([1.5])
    
    bond = BondHarmonic(k=k, r0=r0)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        batch=torch.tensor([0, 0]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        bond_type=torch.zeros(0, dtype=torch.long),
    )
    
    energy = bond(atomic_td)
    
    # No bonds, zero energy
    assert torch.isclose(energy, torch.tensor(0.0))
