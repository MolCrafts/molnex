from typing import Dict, List, Any
import json
import torch

class Exporter:
    """
    Exporter for predicted types.
    """
    def __init__(self, type_map: Dict[int, str]):
        """
        Args:
            type_map: Dictionary mapping type index to type string.
        """
        self.type_map = type_map

    def export_type_map(self, atom_ids: List[int], predictions: torch.Tensor, output_path: str):
        """
        Export mapping from atom ID to predicted type.
        
        Args:
            atom_ids: List of unique atom IDs (or indices).
            predictions: Tensor of class indices (N,)
            output_path: Path to write JSON file.
        """
        preds = predictions.cpu().tolist()
        if len(atom_ids) != len(preds):
            raise ValueError("Length mismatch between atom_ids and predictions")
            
        mapping = {}
        for aid, pred_idx in zip(atom_ids, preds):
            type_str = self.type_map.get(pred_idx, str(pred_idx))
            mapping[aid] = type_str
            
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)

    def suggest_parameters(self, bond_types: List[tuple[str, str]]) -> List[str]:
        """
        Suggest parameter keys for bonded terms.
        """
        suggestions = []
        for t1, t2 in bond_types:
            suggestions.append(f"{t1}-{t2}")
        return sorted(list(set(suggestions)))
