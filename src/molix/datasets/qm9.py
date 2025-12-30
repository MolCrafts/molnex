"""QM9 dataset loader using molpy.read_xyz."""

import io
import os
import tarfile
import urllib.request
from typing import Literal

import molpy
import torch
from torch.utils.data import Dataset


class QM9Dataset(Dataset):
    """QM9 dataset for molecular property prediction.
    
    Downloads and caches the QM9 dataset tarball, applies the standard exclusion
    list, and provides access to molecular structures as molpy.Frame objects.
    Frames are cached in memory after first access to avoid re-reading the tarball.
    
    The dataset contains ~130k small organic molecules with quantum mechanical
    properties calculated at the B3LYP/6-31G(2df,p) level of theory.
    
    Args:
        root: Directory to cache downloaded files
        split: Dataset split - 'train', 'val', or 'test'
        download: If True, download the dataset if not already cached
        
    Example:
        >>> dataset = QM9Dataset(root="./data/qm9", split="train", download=True)
        >>> frame = dataset[0]
        >>> print(frame.metadata['target']['U0'])  # Internal energy at 0K
        >>> 
        >>> # Use with DataLoader and collate_frames
        >>> from torch.utils.data import DataLoader
        >>> from molix.data.collate import collate_frames
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_frames)
        >>> batch_td = next(iter(loader))
    """
    
    URL = "https://ndownloader.figshare.com/files/3195389"
    EXCLUDE_URL = "https://figshare.com/ndownloader/files/3195404"
    
    # Property names from QM9 XYZ file header (line 2)
    PROPERTY_NAMES = [
        'tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo',
        'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    def __init__(
        self,
        root: str = "./data/qm9",
        split: Literal["train", "val", "test"] = "train",
        download: bool = True,
    ):
        self.root = root
        self.split = split
        self.tarball_path = os.path.join(root, "qm9.tar.bz2")
        self.exclude_path = os.path.join(root, "qm9_exclude.txt")
        
        os.makedirs(root, exist_ok=True)
        
        if download:
            if not os.path.exists(self.tarball_path):
                self._download_tarball()
            if not os.path.exists(self.exclude_path):
                self._download_exclusion_list()
        
        if not os.path.exists(self.tarball_path):
            raise RuntimeError(
                f"QM9 tarball not found at {self.tarball_path}. "
                "Set download=True to download automatically."
            )
        
        # Load exclusion list
        self.excluded_indices = self._load_exclusion_list()
        
        # Get list of XYZ files from tarball, filtered by exclusion list
        self.xyz_members = self._get_xyz_members()
        
        # Split dataset
        self.indices = self._get_split_indices()
        
        # Cache for loaded frames to avoid re-reading tarball
        self._frames: dict[int, molpy.Frame] = {}
    
    def _download_tarball(self):
        """Download the QM9 tarball from figshare."""
        print(f"Downloading QM9 dataset from {self.URL}...")
        print("This may take a few minutes (~350 MB)...")
        
        # Add User-Agent header to avoid 403 errors
        req = urllib.request.Request(
            self.URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            with open(self.tarball_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Downloaded to {self.tarball_path}")
    
    def _download_exclusion_list(self):
        """Download the exclusion list from figshare."""
        print(f"Downloading QM9 exclusion list from {self.EXCLUDE_URL}...")
        
        # Add User-Agent header to avoid 403 errors
        req = urllib.request.Request(
            self.EXCLUDE_URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            with open(self.exclude_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Downloaded to {self.exclude_path}")
    
    def _load_exclusion_list(self) -> set[int]:
        """Load the list of excluded molecule indices."""
        if not os.path.exists(self.exclude_path):
            return set()
        
        excluded = set()
        with open(self.exclude_path, 'r') as f:
            lines = f.readlines()
            # Skip header (first 9 lines) and footer (last line)
            for line in lines[9:-1]:
                parts = line.split()
                if parts:
                    excluded.add(int(parts[0]))
        
        print(f"Loaded {len(excluded)} excluded molecules")
        return excluded
    
    def _get_xyz_members(self) -> list[str]:
        """Get list of XYZ file members from tarball, excluding problematic molecules."""
        with tarfile.open(self.tarball_path, 'r:bz2') as tar:
            members = tar.getnames()
        
        # Filter for XYZ files and apply exclusion list
        xyz_files = []
        for name in members:
            if not name.endswith('.xyz'):
                continue
            
            # Extract molecule index from filename (e.g., dsgdb9nsd_000001.xyz -> 1)
            mol_idx = int(name[-10:-4])
            if mol_idx not in self.excluded_indices:
                xyz_files.append(name)
        
        xyz_files.sort()
        print(f"Found {len(xyz_files)} valid molecules in QM9 dataset")
        return xyz_files
    
    def _get_split_indices(self) -> list[int]:
        """Get indices for the requested split."""
        n_total = len(self.xyz_members)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if self.split == "train":
            return list(range(n_train))
        elif self.split == "val":
            return list(range(n_train, n_train + n_val))
        elif self.split == "test":
            return list(range(n_train + n_val, n_total))
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> molpy.Frame:
        """Get a molecule as a molpy.Frame object.
        
        Args:
            idx: Index in the current split
            
        Returns:
            molpy.Frame with molecular structure and properties in metadata["target"]
        """
        # Check cache first
        if idx in self._frames:
            return self._frames[idx]
        
        # Map split index to global index
        global_idx = self.indices[idx]
        member_name = self.xyz_members[global_idx]
        
        # Extract XYZ file from tarball
        with tarfile.open(self.tarball_path, 'r:bz2') as tar:
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            assert f is not None, f"Failed to extract {member_name}"
            
            # Read XYZ content
            content = f.read().decode('utf-8')
        
        # Parse property line (line 2 of XYZ file) for QM9 properties
        lines = content.splitlines()
        assert len(lines) >= 2, f"Invalid XYZ file: {member_name}"
        
        property_line = lines[1].split()
        properties = {}
        
        # Parse tag and index separately (strings/ints)
        if len(property_line) > 0:
            properties['tag'] = property_line[0]
        if len(property_line) > 1:
            properties['index'] = int(property_line[1])
        
        # Parse numeric properties
        for i, prop_name in enumerate(self.PROPERTY_NAMES[2:], start=2):
            if i < len(property_line):
                # Handle scientific notation with *^ format
                value_str = property_line[i].replace('*^', 'E')
                properties[prop_name] = float(value_str)
        
        # Create temporary file-like object for molpy.read_xyz
        xyz_bytes = content.encode('utf-8')
        xyz_file = io.BytesIO(xyz_bytes)
        
        # Use molpy.read_xyz to parse the XYZ structure
        frame = molpy.read_xyz(xyz_file)
        
        # Store QM9 properties in metadata["target"]
        frame.metadata["target"] = properties
        
        # Cache the frame
        self._frames[idx] = frame
        
        return frame



