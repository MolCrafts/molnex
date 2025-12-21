from typing import Any, List
import torch

class ProxyLabeler:
    """
    Simple proxy labeler using atomic number mapping.
    Real OPLS would be complex graph matching.
    Here we map Z=1 -> 0 (H), Z=6 -> 1 (C), Z=7 -> 2 (N), Z=8 -> 3 (O), others -> 4
    """
    def __call__(self, system: Any) -> torch.Tensor:
        # Assuming system has atoms with symbol
        labels = []
        for atom in system.atoms:
            sym = atom.get("symbol", "C")
            if sym == "H":
                labels.append(0)
            elif sym == "C":
                labels.append(1)
            elif sym == "N":
                labels.append(2)
            elif sym == "O":
                labels.append(3)
            else:
                labels.append(4)
        return torch.tensor(labels, dtype=torch.long)
