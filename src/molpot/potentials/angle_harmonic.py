import torch
from typing import Any
from molpot.potentials.base import BasePotential


class AngleHarmonic(BasePotential):
    """Harmonic angle bending potential.
    
    Energy formula:
        E = 0.5 * k * (theta - theta0)^2
    
    Parameters are stored as type-indexed vectors:
        k[angle_type]: Force constant
        theta0[angle_type]: Equilibrium angle (in radians)
    
    Attributes:
        k: Force constants [num_angle_types]
        theta0: Equilibrium angles in radians [num_angle_types]
    """
    
    name = "angle_harmonic_torch"
    type = "angle"
    
    def __init__(
        self,
        k: torch.Tensor,
        theta0: torch.Tensor,
    ):
        """Initialize AngleHarmonic potential.
        
        Args:
            k: Force constant vector [num_angle_types]
            theta0: Equilibrium angle vector in radians [num_angle_types]
        """
        super().__init__()
        
        # Validate shapes
        if k.shape != theta0.shape:
            raise ValueError(
                f"k and theta0 must have same shape, got "
                f"k: {k.shape}, theta0: {theta0.shape}"
            )
        
        if k.ndim != 1:
            raise ValueError(
                f"k must be 1D vector [num_angle_types], got shape {k.shape}"
            )
        
        # Register as buffers
        self.register_buffer("k", k)
        self.register_buffer("theta0", theta0)
    
    def forward(self, data: dict[str, Any] | None = None, **kwargs) -> torch.Tensor:
        """Compute harmonic angle energy.
        
        Args:
            data: Optional dictionary with molecular fields
            **kwargs: Alternate way to pass explicit tensors:
                - pos: Positions [N, 3]
                - angle_index: Angle indices [3, num_angles] (i-j-k triplets)
                - angle_types: Angle types [num_angles]
                
        Returns:
            Total angle energy (scalar)
        """
        # Extract data
        pos = kwargs.get("pos")
        angle_index = kwargs.get("angle_index")
        angle_types = kwargs.get("angle_types")
        
        if pos is None and data is not None:
            if isinstance(data, dict):
                pos = data.get("pos")
                angle_index = data.get("angle_index") if angle_index is None else angle_index
                angle_types = data.get("angle_types") if angle_types is None else angle_types
                if pos is None and "atoms" in data:
                    pos = data["atoms"]["x"]
        
        if pos is None or angle_index is None or angle_types is None:
            raise ValueError("AngleHarmonic requires pos, angle_index, and angle_types.")
            
        # Convert numpy to torch if needed
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float()
            angle_index = torch.from_numpy(angle_index).long()
            angle_types = torch.from_numpy(angle_types).long()
        
        # Handle empty angles
        if angle_index.size(1) == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        
        # Get positions for each angle (i-j-k where j is the central atom)
        pos_i = pos[angle_index[0]]  # [num_angles, 3]
        pos_j = pos[angle_index[1]]  # [num_angles, 3] (central atom)
        pos_k = pos[angle_index[2]]  # [num_angles, 3]
        
        # Compute vectors from central atom
        vec_ji = pos_i - pos_j  # [num_angles, 3]
        vec_jk = pos_k - pos_j  # [num_angles, 3]
        
        # Normalize vectors
        vec_ji_norm = vec_ji / (torch.norm(vec_ji, dim=-1, keepdim=True) + 1e-8)
        vec_jk_norm = vec_jk / (torch.norm(vec_jk, dim=-1, keepdim=True) + 1e-8)
        
        # Compute angle using dot product
        cos_theta = torch.sum(vec_ji_norm * vec_jk_norm, dim=-1)  # [num_angles]
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Avoid numerical issues
        theta = torch.acos(cos_theta)  # [num_angles]
        
        # Look up parameters for each angle
        k_angles = self.k[angle_types]  # [num_angles]
        theta0_angles = self.theta0[angle_types]  # [num_angles]
        
        # Compute harmonic energy: E = 0.5 * k * (theta - theta0)^2
        displacement = theta - theta0_angles
        energy_per_angle = 0.5 * k_angles * displacement ** 2
        
        # Sum over all angles
        total_energy = energy_per_angle.sum()
        
        return total_energy
    
    def __repr__(self) -> str:
        num_types = len(self.k)
        return f"AngleHarmonic(num_angle_types={num_types})"
