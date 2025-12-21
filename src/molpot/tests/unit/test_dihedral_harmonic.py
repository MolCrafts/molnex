"""Unit tests for DihedralHarmonic potential."""

import torch
import pytest
import math
from molpot.potentials import DihedralHarmonic
from molpot.data.atomic_td import AtomicTD


def test_dihedral_harmonic_initialization():
    """Test DihedralHarmonic initialization with valid parameters."""
    k = torch.tensor([10.0, 20.0])
    phi0 = torch.tensor([0.0, math.pi])
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    assert dihedral.k.shape == (2,)
    assert dihedral.phi0.shape == (2,)


def test_dihedral_harmonic_shape_validation():
    """Test that DihedralHarmonic validates parameter shapes."""
    k = torch.tensor([10.0, 20.0])
    phi0 = torch.tensor([0.0])  # Wrong shape
    
    with pytest.raises(ValueError, match="must have same shape"):
        DihedralHarmonic(k=k, phi0=phi0)


def test_dihedral_harmonic_planar():
    """Test DihedralHarmonic with planar (0 degree) dihedral."""
    k = torch.tensor([10.0])
    phi0 = torch.tensor([0.0])
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    # Four atoms in plane (dihedral = 0)
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 1]),
        x=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
        batch=torch.tensor([0, 0, 0, 0]),
        dihedral_index=torch.tensor([[0], [1], [2], [3]]),
        dihedral_type=torch.tensor([0]),
    )
    
    energy = dihedral(atomic_td)
    
    # At equilibrium (phi = 0), energy should be 0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_dihedral_harmonic_trans():
    """Test DihedralHarmonic with trans (180 degree) configuration."""
    k = torch.tensor([10.0])
    phi0 = torch.tensor([math.pi])  # 180 degrees
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    # Four atoms in trans configuration
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 1]),
        x=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],  # Out of plane
        ]),
        batch=torch.tensor([0, 0, 0, 0]),
        dihedral_index=torch.tensor([[0], [1], [2], [3]]),
        dihedral_type=torch.tensor([0]),
    )
    
    energy = dihedral(atomic_td)
    
    # Should compute energy without error
    assert energy.ndim == 0
    assert not torch.isnan(energy)


def test_dihedral_harmonic_perpendicular():
    """Test DihedralHarmonic with perpendicular configuration."""
    k = torch.tensor([10.0])
    phi0 = torch.tensor([math.pi / 2])  # 90 degrees
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    # Four atoms with 90 degree dihedral
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 1]),
        x=torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]),
        batch=torch.tensor([0, 0, 0, 0]),
        dihedral_index=torch.tensor([[0], [1], [2], [3]]),
        dihedral_type=torch.tensor([0]),
    )
    
    energy = dihedral(atomic_td)
    
    # At equilibrium (phi = pi/2), energy should be close to 0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-4)


def test_dihedral_harmonic_multiple():
    """Test DihedralHarmonic with multiple dihedrals."""
    k = torch.tensor([10.0, 20.0])
    phi0 = torch.tensor([0.0, math.pi])
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    # Five atoms with two dihedrals
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 6, 1]),
        x=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]),
        batch=torch.tensor([0, 0, 0, 0, 0]),
        dihedral_index=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
        dihedral_type=torch.tensor([0, 0]),
    )
    
    energy = dihedral(atomic_td)
    
    # Both dihedrals are planar (phi = 0)
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_dihedral_harmonic_gradient():
    """Test that DihedralHarmonic energy is differentiable."""
    k = torch.tensor([10.0])
    phi0 = torch.tensor([0.0])
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], requires_grad=True)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 1]),
        x=x,
        batch=torch.tensor([0, 0, 0, 0]),
        dihedral_index=torch.tensor([[0], [1], [2], [3]]),
        dihedral_type=torch.tensor([0]),
    )
    
    energy = dihedral(atomic_td)
    energy.backward()
    
    assert x.grad is not None
    assert x.grad.shape == (4, 3)


def test_dihedral_harmonic_empty():
    """Test DihedralHarmonic with no dihedrals."""
    k = torch.tensor([10.0])
    phi0 = torch.tensor([0.0])
    
    dihedral = DihedralHarmonic(k=k, phi0=phi0)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 6, 6, 1]),
        x=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
        batch=torch.tensor([0, 0, 0, 0]),
        dihedral_index=torch.zeros((4, 0), dtype=torch.long),
        dihedral_type=torch.zeros(0, dtype=torch.long),
    )
    
    energy = dihedral(atomic_td)
    
    # No dihedrals, zero energy
    assert torch.isclose(energy, torch.tensor(0.0))
