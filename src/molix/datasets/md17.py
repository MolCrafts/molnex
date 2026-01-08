"""MD17 dataset loader."""

import os
import ssl
import urllib.request

import molpy
import numpy as np
import torch
from torch.utils.data import Dataset

# Allow unverified context for downloading from some scientific sites if needed
ssl._create_default_https_context = ssl._create_unverified_context


class MD17Dataset(Dataset):
    """MD17 dataset for molecular dynamics.
    
    Provides access to MD17 molecular dynamics trajectories with energies
    and forces calculated at the PBE+vdW-TS level of theory.
    
    Source: http://www.quantum-machine.org/gdml/data/npz/md17
    
    Args:
        root: Directory to cache downloaded files
        molecule: Molecule name (e.g., "aspirin", "benzene", "ethanol")
        download: If True, download the dataset if not already cached
        
    Example:
        >>> dataset = MD17Dataset(root="./data/md17", molecule="aspirin")
        >>> frame = dataset[0]
        >>> print(frame.metadata['target']['energy'])  # Energy in kcal/mol
    """
    
    BASE_URL = "http://www.quantum-machine.org/gdml/data/npz/md17"
    
    def __init__(self, root: str, molecule: str = "aspirin", download: bool = True):
        self.root = root
        self.molecule = molecule
        self.filename = f"md17_{molecule}.npz"
        self.filepath = os.path.join(root, self.filename)
        
        os.makedirs(root, exist_ok=True)
        
        if download and not os.path.exists(self.filepath):
            self._download()
            
        assert os.path.exists(self.filepath), (
            f"MD17 file not found at {self.filepath}. "
            "Please set download=True or manually place the file."
        )

        # Load NPZ file
        self.data = np.load(self.filepath)

        # Extract arrays (MD17 format: 'z', 'R', 'E', 'F')
        assert 'z' in self.data, "MD17 file missing 'z' (atomic numbers)"
        assert 'R' in self.data, "MD17 file missing 'R' (positions)"
        assert 'E' in self.data, "MD17 file missing 'E' (energies)"
        assert 'F' in self.data, "MD17 file missing 'F' (forces)"
        
        self.z = self.data['z']  # Atomic numbers [N_atoms]
        self.R = self.data['R']  # Positions [N_frames, N_atoms, 3]
        self.E = self.data['E']  # Energies [N_frames]
        self.F = self.data['F']  # Forces [N_frames, N_atoms, 3]

    def _download(self):
        """Download MD17 dataset from quantum-machine.org."""
        url = f"{self.BASE_URL}/{self.filename}"
        print(f"Downloading {url} to {self.filepath}...")
        urllib.request.urlretrieve(url, self.filepath)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx: int) -> molpy.Frame:
        """Get a frame as a molpy.Frame object.
        
        Args:
            idx: Frame index
            
        Returns:
            molpy.Frame with molecular structure, energy, and forces in metadata["target"]
        """
        # Create Frame with atomic structure
        frame = molpy.Frame(
            blocks={
                "atoms": {
                    "x": self.R[idx],  # Positions for this frame [N_atoms, 3]
                    "z": self.z,       # Atomic numbers (same for all frames) [N_atoms]
                }
            }
        )
        
        # Store energy and forces in metadata["target"]
        frame.metadata["target"] = {
            "energy": float(self.E[idx]),
            "forces": self.F[idx]  # [N_atoms, 3]
        }
        
        return frame


def md17_collate_fn(frames: list[molpy.Frame]):
    """Collate function for MD17 dataset: list[molpy.Frame] -> (AtomicTD, targets).
    
    Converts a batch of molpy.Frame objects into a batched AtomicTD with
    NestedTensor fields and extracts energy/force targets.
    
    Args:
        frames: List of molpy.Frame objects with MD17 energy/forces in metadata["target"]
        
    Returns:
        Tuple of (batch_td, targets) where:
        Tuple of (batch_td, targets) where:
            - batch_td: AtomicTD with padded tensor fields for molecular structures
            - targets: Dict with "energy" [B] and "forces" padded tensor [B, max_atoms, 3]
            
    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MD17Dataset(root="./data/md17", molecule="aspirin")
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=md17_collate_fn)
        >>> batch_td, targets = next(iter(loader))
        >>> batch_td["atoms", "x"].shape
        torch.Size([32, 21, 3])  # [B, max_atoms, 3]
        >>> targets["energy"].shape
        torch.Size([32])
        >>> targets["forces"].shape
        torch.Size([32, 21, 3])
    """
    from molix.data.collate import collate_frames
    
    # Extract targets from frame.metadata["target"]
    energies = torch.tensor(
        [frame.metadata["target"]["energy"] for frame in frames],
        dtype=torch.float32
    )
    
    # Forces are per-atom, so use padded tensor
    forces_list = [
        torch.tensor(frame.metadata["target"]["forces"], dtype=torch.float32)
        for frame in frames
    ]
    
    # Pad forces
    max_atoms = max(len(f) for f in forces_list)
    batch_size = len(frames)
    forces_padded = torch.zeros(batch_size, max_atoms, 3, dtype=torch.float32)
    
    for i, f in enumerate(forces_list):
        n_atoms = len(f)
        forces_padded[i, :n_atoms] = f
        
    # Collate frames into batched AtomicTD (uses padded tensors internally)
    batch_td = collate_frames(frames)
    
    targets = {
        "energy": energies,
        "forces": forces_padded
    }
    
    return batch_td, targets
