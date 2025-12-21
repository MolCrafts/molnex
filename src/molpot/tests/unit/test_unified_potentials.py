"""Test unified potential system with BasePotential."""

import torch
import pytest
from molpot.data.atomic_td import AtomicTD
from molpot.potentials import LJ126, BondHarmonic, AngleHarmonic, DihedralHarmonic
from molpot.models.pinet2 import PiNet


def test_lj126_with_base_potential():
    """Test LJ126 using new BasePotential."""
    # Create LJ126 potential
    epsilon = torch.tensor([[0.066, 0.048], [0.048, 0.030]])  # C-C, C-H, H-H
    sigma = torch.tensor([[3.5, 3.0], [3.0, 2.5]])
    lj = LJ126(epsilon=epsilon, sigma=sigma, cutoff=10.0)
    
    # Create test data
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        atom_type=torch.tensor([0, 1, 1]),  # C, H, H
    )
    
    # Test calc_energy (protocol method)
    energy = lj.calc_energy(atomic_td)
    assert isinstance(energy, float)
    assert energy < 0  # Attractive at this distance
    
    # Test calc_forces (protocol method)
    forces = lj.calc_forces(atomic_td)
    assert forces.shape == (3, 3)
    
    # Test forward (PyTorch method)
    energy_tensor = lj(atomic_td)
    assert isinstance(energy_tensor, torch.Tensor)
    assert energy_tensor.ndim == 0  # Scalar


def test_bond_harmonic_with_base_potential():
    """Test BondHarmonic using new BasePotential."""
    # Create BondHarmonic potential
    k = torch.tensor([340.0, 300.0])  # C-H, H-H
    r0 = torch.tensor([1.09, 0.74])
    bond = BondHarmonic(k=k, r0=r0)
    
    # Create test data
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1, 1]),
        x=torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [0.0, 1.09, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),  # C-H bonds
        bond_type=torch.tensor([0, 0]),
    )
    
    # Test calc_energy
    energy = bond.calc_energy(atomic_td)
    assert isinstance(energy, float)
    assert abs(energy) < 1e-3  # Should be near zero at equilibrium
    
    # Test calc_forces
    forces = bond.calc_forces(atomic_td)
    assert forces.shape == (3, 3)


def test_pinet_with_base_potential():
    """Test PiNet ML potential using new BasePotential."""
    # Create PiNet model
    pinet = PiNet(
        hidden_dim=32,
        num_layers=2,
        cutoff=5.0,
        num_rbf=20,
        max_z=10,
    )
    
    # Create test data (no type needed for ML potential!)
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1, 1]),  # Only atomic numbers needed
        x=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        batch=torch.tensor([0, 0, 0]),
        # No atom_type field needed!
    )
    
    # Test calc_energy
    energy = pinet.calc_energy(atomic_td)
    assert isinstance(energy, float)
    
    # Test calc_forces
    forces = pinet.calc_forces(atomic_td)
    assert forces.shape == (3, 3)
    
    # Test forward
    energy_tensor = pinet(atomic_td)
    assert isinstance(energy_tensor, torch.Tensor)
    
    # Test that it's trainable
    assert any(p.requires_grad for p in pinet.parameters())


def test_nested_access_compatibility():
    """Test that nested access works for both tuple and dict style."""
    atomic_td = AtomicTD.create(
        z=torch.tensor([6, 1]),
        x=torch.randn(2, 3),
        batch=torch.tensor([0, 0]),
    )
    
    # Both access styles should work
    x_tuple = atomic_td["atoms", "x"]
    x_nested = atomic_td["atoms"]["x"]
    
    assert torch.equal(x_tuple, x_nested)


def test_protocol_compliance():
    """Test that all potentials implement the protocol."""
    potentials = [
        LJ126(epsilon=torch.ones(2, 2), sigma=torch.ones(2, 2)),
        BondHarmonic(k=torch.ones(2), r0=torch.ones(2)),
        AngleHarmonic(k=torch.ones(2), theta0=torch.ones(2)),
        DihedralHarmonic(k=torch.ones(2), phi0=torch.ones(2)),
        PiNet(hidden_dim=16, num_layers=1),
    ]
    
    for pot in potentials:
        # Check attributes
        assert hasattr(pot, "name")
        assert hasattr(pot, "type")
        
        # Check methods
        assert hasattr(pot, "calc_energy")
        assert hasattr(pot, "calc_forces")
        assert hasattr(pot, "forward")
        
        # Check inheritance
        assert isinstance(pot, torch.nn.Module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
