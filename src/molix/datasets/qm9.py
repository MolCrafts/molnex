"""QM9 dataset loader using molpy.read_xyz."""

import io
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Literal, Sequence, TYPE_CHECKING

import numpy as np
import molpy as mp
from molpy.io.data.xyz import XYZReader
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from molix.data.dataset import Dataset as BaseDataset
from molix.data.preprocess import DatasetPreprocessor

from molix.data import AtomicTD


class QM9Dataset(BaseDataset):
    """QM9 dataset for molecular property prediction.
    
    Downloads and caches the QM9 dataset tarball, applies the standard exclusion
    list, and provides access to molecular structures as mp.Frame objects.
    
    Args:
        root: Directory to cache downloaded files
        source: Source for QM9 tarball - can be:
            - URL (str): Download from URL (default: figshare)
            - Local path (str/Path): Use local tarball file
            - None: Use default figshare URL
        preprocessors: Optional sequence of preprocessors to apply to each frame
        extract: If True, extract the tarball for faster access (default: False)
        total: If set, randomly sample this many molecules (default: None = use all)
        
    Example:
        >>> # Use on-the-fly access (default)
        >>> dataset = QM9Dataset(root="./data/qm9", extract=False)
        >>> 
        >>> # Extract for maximum performance
        >>> dataset = QM9Dataset(root="./data/qm9", extract=True)
        >>> 
        >>> 
        >>> # Use subset for quick testing
        >>> dataset = QM9Dataset(root="./data/qm9", total=1000)
        >>> # Use with preprocessors
        >>> from molix.data.preprocess import AtomicDressPreprocessor, NeighborListPreprocessor
        >>> atomic_dress = AtomicDressPreprocessor(elements=[1, 6, 7, 8, 9])
        >>> neighbor_list = NeighborListPreprocessor(cutoff=5.0)
        >>> dataset = QM9Dataset(root="./data/qm9", preprocessors=[atomic_dress, neighbor_list])
        >>> dataset.prepare()  # Load and preprocess all frames
        >>> train_dataset, val_dataset = dataset.split()  # Split using Dataset.split()
    """
    
    DEFAULT_URL = "https://ndownloader.figshare.com/files/3195389"
    EXCLUDE_URL = "https://figshare.com/ndownloader/files/3195404"
    
    # Property names from QM9 XYZ file header (line 2)
    PROPERTY_NAMES = [
        'tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo',
        'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    def __init__(
        self,
        root: str | Path = "./data/qm9",
        source: str | Path | None = None,
        preprocessors: Sequence[DatasetPreprocessor] | None = None,
        extract: bool = False,
        total: int | None = None,
    ):
        # Initialize base Dataset with preprocessors
        super().__init__(preprocessors=preprocessors)
        
        self.root = Path(root)
        self.tarball_path = self.root / "qm9.tar.bz2"
        self.extracted_path = self.root / "qm9_extracted"
        self.exclude_path = self.root / "qm9_exclude.txt"
        self.extract = extract
        self.total = total
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Handle source parameter
        if source is None:
            source = self.DEFAULT_URL
        
        source_is_url = isinstance(source, str) and (source.startswith("http://") or source.startswith("https://"))
        source_path = Path(source)
        
        if not source_is_url and source_path.exists():
            if source_path.resolve() != self.tarball_path.resolve():
                print(f"Using local QM9 tarball: {source}")
                shutil.copy2(source, self.tarball_path)
        elif source_is_url:
            if not self.tarball_path.exists():
                self._download_tarball(str(source))
        else:
            # Local path specified but does not exist
            if not self.tarball_path.exists():
                print(f"Warning: Local QM9 source not found: {source}")
        
        if not self.exclude_path.exists():
            self._download_exclusion_list()
        
        if not self.tarball_path.exists():
            raise RuntimeError(f"QM9 tarball not found at {self.tarball_path}")
        
        # Load exclusion list
        self.excluded_indices = self._load_exclusion_list()
        
        # Cache for TarInfo objects to allow O(1) archive access
        self._tar_info_cache: dict[str, tarfile.TarInfo] = {}
        
        # Handle extraction or indexing
        if self.extract:
            if not self.extracted_path.exists() or not any(self.extracted_path.iterdir()):
                self._extract_tarball()
            self.xyz_members = self._index_files()
        else:
            self.xyz_members = self._index_archive()
        
        # Random sampling if total is specified
        if self.total is not None and self.total < len(self.xyz_members):
            import random
            random.seed(42)  # For reproducibility
            self.xyz_members = random.sample(self.xyz_members, self.total)
            self.xyz_members.sort()  # Keep sorted for consistency
            print(f"Randomly sampled {self.total} molecules from QM9 dataset")
        
        # Memory cache for frames
        self._frames: list["AtomicTD"] = []
    
    def _download_tarball(self, url: str):
        """Download the QM9 tarball from URL."""
        print(f"Downloading QM9 dataset from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            self.tarball_path.write_bytes(response.read())
        print(f"Downloaded to {self.tarball_path}")
    
    def _download_exclusion_list(self):
        """Download the exclusion list from figshare."""
        print(f"Downloading QM9 exclusion list from {self.EXCLUDE_URL}...")
        req = urllib.request.Request(self.EXCLUDE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            self.exclude_path.write_bytes(response.read())
        print(f"Downloaded to {self.exclude_path}")

    def _extract_tarball(self):
        """Extract the QM9 tarball for efficient direct access."""
        print(f"Extracting QM9 tarball to {self.extracted_path}...")
        self.extracted_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self.tarball_path, 'r:bz2') as tar:
            tar.extractall(path=self.extracted_path)
        print("Extraction complete.")

    def _index_files(self) -> list[str]:
        """Index extracted XYZ files, applying exclusion list."""
        members = [f.name for f in self.extracted_path.glob("*.xyz")]
        xyz_files = []
        for name in members:
            try:
                mol_idx = int(name[-10:-4])
                if mol_idx not in self.excluded_indices:
                    xyz_files.append(name)
            except ValueError:
                continue
        xyz_files.sort()
        print(f"Found {len(xyz_files)} valid molecules in QM9 dataset")
        return xyz_files

    def _index_archive(self) -> list[str]:
        """Index XYZ files in the archive and cache TarInfo objects for fast access."""
        print("Indexing QM9 archive, this will take a moment...")
        xyz_files = []
        with tarfile.open(self.tarball_path, 'r:bz2') as tar:
            for member in tqdm(tar, desc="Indexing archive"):
                mol_idx = int(member.name[-10:-4])
                if mol_idx not in self.excluded_indices:
                    xyz_files.append(member.name)
                    self._tar_info_cache[member.name] = member
        xyz_files.sort()
        print(f"Found {len(xyz_files)} valid molecules in QM9 archive")
        return xyz_files

    def _load_exclusion_list(self) -> set[int]:
        """Load the set of excluded molecule indices."""
        excluded = set()
        if self.exclude_path.exists():
            lines = self.exclude_path.read_text().splitlines()
            for line in lines[9:-1]:
                parts = line.split()
                if parts:
                    excluded.add(int(parts[0]))
        print(f"Loaded {len(excluded)} excluded molecules")
        return excluded
    
    def __len__(self) -> int:
        if self._frames:
            return len(self._frames)
        return len(self.xyz_members)
    
    def _load_all_frames(self) -> list["AtomicTD"]:
        """Load all raw frames from the dataset and convert to AtomicTD.
        
        Returns:
            List of AtomicTD instances with proper schema
        """
        
        atomic_tds = []
        
        # Open tarfile once if not extracted
        tar = None
        if not self.extract:
            tar = tarfile.open(self.tarball_path, 'r:bz2')
        
        try:
            for idx in tqdm(range(len(self.xyz_members)), desc="Loading frames"):
                member_name = self.xyz_members[idx]
            
                if self.extract:
                    file_path = self.extracted_path / member_name
                    content = file_path.read_text(encoding='utf-8')
                else:
                    tar_info = self._tar_info_cache[member_name]
                    f = tar.extractfile(tar_info)
                    if f is None:
                        raise RuntimeError(f"Failed to extract file {member_name} from archive")
                    content = f.read().decode('utf-8')
            
                # Sanitize Mathematica notation
                content = content.replace("*^", "E")
                
                # Read XYZ
                reader = XYZReader(io.StringIO(content), header=self.PROPERTY_NAMES)
                frame = reader.read()

                # Extract data from frame
                Z = torch.from_numpy(frame["atoms"]["number"]).long()
                xyz = torch.from_numpy(
                    np.stack(
                        (
                            frame["atoms"]["x"],
                            frame["atoms"]["y"],
                            frame["atoms"]["z"],
                        ),
                        axis=1,   # (N, 3)
                        )
                ).float()
                batch = torch.zeros(len(Z), dtype=torch.long)
                
                # Build data dict directly
                data = {
                    ("atoms", "Z"): Z,
                    ("atoms", "xyz"): xyz,
                    ("graph", "batch"): batch,
                }
                
                # Add targets (manual, no tag/index)
                meta = frame.metadata
                data[("target", "A")] = torch.tensor([float(meta['A'])], dtype=torch.float32)
                data[("target", "B")] = torch.tensor([float(meta['B'])], dtype=torch.float32)
                data[("target", "C")] = torch.tensor([float(meta['C'])], dtype=torch.float32)
                data[("target", "mu")] = torch.tensor([float(meta['mu'])], dtype=torch.float32)
                data[("target", "alpha")] = torch.tensor([float(meta['alpha'])], dtype=torch.float32)
                data[("target", "homo")] = torch.tensor([float(meta['homo'])], dtype=torch.float32)
                data[("target", "lumo")] = torch.tensor([float(meta['lumo'])], dtype=torch.float32)
                data[("target", "gap")] = torch.tensor([float(meta['gap'])], dtype=torch.float32)
                data[("target", "r2")] = torch.tensor([float(meta['r2'])], dtype=torch.float32)
                data[("target", "zpve")] = torch.tensor([float(meta['zpve'])], dtype=torch.float32)
                data[("target", "U0")] = torch.tensor([float(meta['U0'])], dtype=torch.float32)
                data[("target", "U")] = torch.tensor([float(meta['U'])], dtype=torch.float32)
                data[("target", "H")] = torch.tensor([float(meta['H'])], dtype=torch.float32)
                data[("target", "G")] = torch.tensor([float(meta['G'])], dtype=torch.float32)
                data[("target", "Cv")] = torch.tensor([float(meta['Cv'])], dtype=torch.float32)
                
                # Create AtomicTD directly from data dict
                atomic_td = AtomicTD(data, batch_size=[])
                atomic_tds.append(atomic_td)

        except Exception as e:
            raise e
        
        finally:
            # Close tarfile if we opened it
            if tar is not None:
                tar.close()
        
        return atomic_tds
    
    def prepare(self) -> None:
        """Load all frames and apply preprocessors.
        
        This method should be called before accessing frames if preprocessors
        are set. It loads all raw frames, applies preprocessors, and caches
        the processed results.
        """
        if self._preprocessed:
            return
        
        if not self.preprocessors:
            self._preprocessed = True
            return
        
        raw_frames = self._load_all_frames()
        
        # Apply preprocessors to all frames
        processed_frames = self._apply_preprocess(raw_frames)
        
        # Cache all processed frames
        self._frames = list(processed_frames)
        
        self._preprocessed = True
    
    def __getitem__(self, idx: int) -> mp.Frame:
        """Get a molecule as a mp.Frame with properties in metadata['target']."""
        return self._frames[idx]