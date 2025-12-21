import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Any
from molnex.moltyp.encoder import TopologyEncoder, GeometricEncoder
from molnex.moltyp.head import TypeClassifier
from molnex.moltyp.data import MoleculeDataset, Batch

class MolTypTask:
    """
    Training task for molecular typing.
    """
    def __init__(self, encoder_type: str = "topology", hidden_dim: int = 64, num_classes: int = 5, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Build model
        if encoder_type == "topology":
            self.encoder = TopologyEncoder(hidden_dim=hidden_dim)
        elif encoder_type == "geometric":
            self.encoder = GeometricEncoder(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        self.head = TypeClassifier(hidden_dim=hidden_dim, num_types=num_classes)
        
        self.encoder.to(self.device)
        self.head.to(self.device)
        
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.head.parameters()), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataset: MoleculeDataset, batch_size: int = 2):
        self.encoder.train()
        self.head.train()
        
        total_loss = 0.0
        # Simple collation loop
        # In prod, use torch.utils.data.DataLoader with collate_fn=Batch.from_data_list
        
        indices = torch.randperm(len(dataset))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            data_list = [dataset[idx] for idx in batch_indices]
            batch = Batch.from_data_list(data_list).to(self.device)
            
            self.optimizer.zero_grad()
            
            features = self.encoder(batch)
            logits = self.head(features)
            
            # Loss computation
            if batch.y is None:
                continue
                
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / (len(dataset) / batch_size)

    def evaluate(self, dataset: MoleculeDataset):
        self.encoder.eval()
        self.head.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            full_batch = Batch.from_data_list(dataset.data_list).to(self.device)
            features = self.encoder(full_batch)
            logits = self.head(features)
            preds = logits.argmax(dim=-1)
            
            if full_batch.y is not None:
                correct = (preds == full_batch.y).sum().item()
                total = full_batch.y.size(0)
                
        acc = correct / total if total > 0 else 0.0
        return {"accuracy": acc}
