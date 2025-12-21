"""Unit tests for AngleHarmonic potential."""

import torch
import pytest
import math
from molpot.potentials import AngleHarmonic
from molpot.data.atomic_td import AtomicTD


def test_angle_harmonic_initialization():
    """Test AngleHarmonic initialization with valid parameters."""
    k = torch.tensor([50.0, 75.0])
    theta0 = torch.tensor([math.pi / 2, math.pi / 3])
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    assert angle.k.shape == (2,)
    assert angle.theta0.shape == (2,)


def test_angle_harmonic_shape_validation():
    """Test that AngleHarmonic validates parameter shapes."""
    k = torch.tensor([50.0, 75.0])
    theta0 = torch.tensor([math.pi / 2])  # Wrong shape
    
    with pytest.raises(ValueError, match="must have same shape"):
        AngleHarmonic(k=k, theta0=theta0)


def test_angle_harmonic_equilibrium():
    """Test AngleHarmonic with angle at equilibrium."""
    k = torch.tensor([50.0])
    theta0 = torch.tensor([math.pi / 2])  # 90 degrees
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    # Three atoms forming 90 degree angle
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        angle_index=torch.tensor([[0], [1], [2]]),
        angle_type=torch.tensor([0]),
    )
    
    energy = angle(atomic_td)
    
    # At equilibrium, energy should be 0
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_angle_harmonic_bent():
    """Test AngleHarmonic with bent angle."""
    k = torch.tensor([50.0])
    theta0 = torch.tensor([math.pi])  # 180 degrees (linear)
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    # Three atoms forming 90 degree angle (bent from 180)
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        angle_index=torch.tensor([[0], [1], [2]]),
        angle_type=torch.tensor([0]),
    )
    
    energy = angle(atomic_td)
    
    # E = 0.5 * k * (theta - theta0)^2
    # theta = pi/2, theta0 = pi
    # E = 0.5 * 50 * (pi/2 - pi)^2 = 0.5 * 50 * (pi/2)^2
    expected = 0.5 * 50.0 * (math.pi / 2) ** 2
    assert torch.isclose(energy, torch.tensor(expected), atol=1e-4)


def test_angle_harmonic_multiple_angles():
    """Test AngleHarmonic with multiple angles of different types."""
    k = torch.tensor([50.0, 75.0])
    theta0 = torch.tensor([math.pi / 2, math.pi / 3])
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    # Four atoms with two angles
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1, 1]),
        x=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]),
        batch=torch.tensor([0, 0, 0, 0]),
        angle_index=torch.tensor([[0, 2], [1, 1], [2, 3]]),
        angle_type=torch.tensor([0, 0]),
    )
    
    energy = angle(atomic_td)
    
    # Both angles at 90 degrees (equilibrium for type 0)
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-5)


def test_angle_harmonic_gradient():
    """Test that AngleHarmonic energy is differentiable."""
    k = torch.tensor([50.0])
    theta0 = torch.tensor([math.pi / 2])
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], requires_grad=True)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1]),
        x=x,
        batch=torch.tensor([0, 0, 0]),
        angle_index=torch.tensor([[0], [1], [2]]),
        angle_type=torch.tensor([0]),
    )
    
    energy = angle(atomic_td)
    energy.backward()
    
    assert x.grad is not None
    assert x.grad.shape == (3, 3)


def test_angle_harmonic_empty():
    """Test AngleHarmonic with no angles."""
    k = torch.tensor([50.0])
    theta0 = torch.tensor([math.pi / 2])
    
    angle = AngleHarmonic(k=k, theta0=theta0)
    
    atomic_td = AtomicTD.create(
        z=torch.tensor([1, 8, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        angle_index=torch.zeros((3, 0), dtype=torch.long),
        angle_type=torch.zeros(0, dtype=torch.long),
    )
    
    energy = angle(atomic_td)
    
    # No angles, zero energy
    assert torch.isclose(energy, torch.tensor(0.0))
