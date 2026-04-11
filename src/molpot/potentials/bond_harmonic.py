import torch
from typing import Any
from molpot.potentials.base import BasePotential


class BondHarmonic(BasePotential):
    """Harmonic bond stretching potential.
    
    Energy formula:
        E = 0.5 * k * (r - r0)^2
    
    Parameters are stored as type-indexed vectors:
        k[bond_type]: Force constant
        r0[bond_type]: Equilibrium bond length
    
    Attributes:
        k: Force constants [num_bond_types]
        r0: Equilibrium bond lengths [num_bond_types]
    """
    
    name = "bond_harmonic_torch"
    type = "bond"
    
    def __init__(
        self,
        k: torch.Tensor,
        r0: torch.Tensor,
    ):
        """Initialize BondHarmonic potential.
        
        Args:
            k: Force constant vector [num_bond_types]
            r0: Equilibrium bond length vector [num_bond_types]
        """
        super().__init__()
        
        # Validate shapes
        if k.shape != r0.shape:
            raise ValueError(
                f"k and r0 must have same shape, got "
                f"k: {k.shape}, r0: {r0.shape}"
            )
        
        if k.ndim != 1:
            raise ValueError(
                f"k must be 1D vector [num_bond_types], got shape {k.shape}"
            )
        
        # Register as buffers
        self.register_buffer("k", k)
        self.register_buffer("r0", r0)
    
    def forward(self, data: dict[str, Any] | None = None, **kwargs) -> torch.Tensor:
        """Compute harmonic bond energy.
        
        Args:
            data: Optional dictionary with molecular fields
            **kwargs: Alternate way to pass explicit tensors:
                - pos: Positions [N, 3]
                - bond_index: Bond indices [2, num_bonds]
                - bond_types: Bond types [num_bonds]
                
        Returns:
            Total bond energy (scalar)
        """
        # Extract data
        pos = kwargs.get("pos")
        bond_index = kwargs.get("bond_index")
        bond_types = kwargs.get("bond_types")
        
        if pos is None and data is not None:
            if isinstance(data, dict):
                pos = data.get("pos")
                if bond_index is None:
                    if "edge_index" in data:
                        bond_index = data["edge_index"]
                    elif "bond_index" in data:
                        bond_index = data["bond_index"]
                if bond_types is None:
                    bond_types = data.get("bond_types")
                if pos is None and "atoms" in data:
                    pos = data["atoms"]["x"]
                if bond_index is None and "bonds" in data:
                    bond_index = data["bonds"].get("i")
                if bond_types is None and "bonds" in data:
                    bond_types = data["bonds"].get("type")
        
        if pos is None or bond_index is None or bond_types is None:
            raise ValueError("BondHarmonic requires pos, bond_index, and bond_types.")
            
        # Convert numpy to torch if needed
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            bond_index = torch.from_numpy(bond_index).long()
            bond_types = torch.from_numpy(bond_types).long()
        
        # Handle empty bonds
        if bond_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        
        # Get positions for each bond
        pos_i = pos[bond_index[0]]  # [num_bonds, 3]
        pos_j = pos[bond_index[1]]  # [num_bonds, 3]
        
        # Compute bond vectors and lengths
        bond_vec = pos_j - pos_i  # [num_bonds, 3]
        bond_lengths = torch.norm(bond_vec, dim=-1)  # [num_bonds]
        
        # Look up parameters for each bond
        k_bonds = self.k[bond_types]  # [num_bonds]
        r0_bonds = self.r0[bond_types]  # [num_bonds]
        
        # Compute harmonic energy: E = 0.5 * k * (r - r0)^2
        displacement = bond_lengths - r0_bonds
        energy_per_bond = 0.5 * k_bonds * displacement ** 2
        
        # Sum over all bonds
        total_energy = energy_per_bond.sum()
        
        return total_energy
    
    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"BondHarmonic(num_bond_types={num_types})"
