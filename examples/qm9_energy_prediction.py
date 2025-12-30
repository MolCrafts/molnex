"""QM9 Energy Prediction with Transformer.

This example demonstrates training a transformer model on QM9 using molnex infrastructure.

Architecture:
    AtomEmbedding (NestedTensor) -> TransformerEncoder -> ScalarHead -> Energy prediction

Dataset:
    QM9 - ~130k molecules with quantum properties (predicting U0 energy)

Usage:
    python examples/qm9_energy_prediction.py
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
import torchmetrics

from molix import Trainer
from molix.core.hooks import CheckpointHook, ProgressBarHook, BaseHook
from molix.datasets.qm9 import QM9Dataset
from molix.data.collate import collate_frames
from molix.data.pipeline import DataPipeline

from molrep import AtomEmbedding, TransformerEncoder, ScalarHead


# ============================================================================
# Model
# ============================================================================

class QM9EnergyModel(nn.Module):
    """Transformer model for QM9 energy prediction.
    
    Uses NestedTensor throughout for variable-length molecules.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        pooling: Pooling method ('mean', 'sum', 'max')
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        
        # Embedding: atomic numbers + positions -> hidden states (NestedTensor)
        self.embedding = AtomEmbedding(
            num_types=100,
            d_model=d_model,
            use_positions=True,
        )
        
        # Transformer encoder: self-attention (NestedTensor)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Prediction head: pool + MLP -> scalar
        self.head = ScalarHead(
            d_model=d_model,
            hidden_dim=d_model,
            pooling=pooling,
        )
    
    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass.
        
        Args:
            td: TensorDict with ("atoms", "x") and ("atoms", "z") NestedTensors
            
        Returns:
            TensorDict with ("pred", "scalar") [B]
        """
        # Embed atoms
        td = self.embedding(td)  # -> ("atoms", "h")
        
        # Encode with transformer
        td = self.encoder(td)  # -> ("rep", "h")
        
        # Predict energy
        td = self.head(td)  # -> ("pred", "scalar")
        
        return td


# ============================================================================
# Hooks
# ============================================================================

class MetricsHook(BaseHook):
    """Track training and validation metrics."""
    
    def __init__(self, device):
        self.device = device
        self.train_mae = torchmetrics.MeanAbsoluteError().to(device)
        self.train_mse = torchmetrics.MeanSquaredError().to(device)
        self.val_mae = torchmetrics.MeanAbsoluteError().to(device)
        self.val_mse = torchmetrics.MeanSquaredError().to(device)
        self.train_losses = []
        self.val_losses = []
    
    def on_epoch_start(self, trainer, state):
        """Reset metrics."""
        self.train_mae.reset()
        self.train_mse.reset()
        self.val_mae.reset()
        self.val_mse.reset()
        self.train_losses = []
        self.val_losses = []
    
    def on_train_batch_end(self, trainer, state, batch, outputs):
        """Update training metrics."""
        pred = outputs["predictions"]["pred", "scalar"]
        target = batch["target", "U0"]
        loss = outputs["loss"].item()
        
        self.train_mae.update(pred, target)
        self.train_mse.update(pred, target)
        self.train_losses.append(loss)
    
    def on_eval_batch_end(self, trainer, state, batch, outputs):
        """Update validation metrics."""
        pred = outputs["predictions"]["pred", "scalar"]
        target = batch["target", "U0"]
        loss = outputs["loss"].item()
        
        self.val_mae.update(pred, target)
        self.val_mse.update(pred, target)
        self.val_losses.append(loss)
    
    def on_epoch_end(self, trainer, state):
        """Print metrics."""
        train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0
        val_loss = sum(self.val_losses) / len(self.val_losses) if self.val_losses else 0
        
        print(f"\nEpoch {state.epoch + 1}:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {self.train_mae.compute():.4f} eV, "
              f"RMSE: {torch.sqrt(self.train_mse.compute()):.4f} eV")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {self.val_mae.compute():.4f} eV, "
              f"RMSE: {torch.sqrt(self.val_mse.compute()):.4f} eV")


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 512
    dropout = 0.1
    pooling = "mean"
    
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    
    data_root = "./data/qm9"
    checkpoint_dir = "./checkpoints/qm9"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Data
    print("=" * 60)
    print("Loading QM9 Dataset")
    print("=" * 60)
    
    train_dataset = QM9Dataset(root=data_root, split="train", download=True)
    val_dataset = QM9Dataset(root=data_root, split="val", download=False)
    
    train_pipeline = DataPipeline(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_frames,
        num_workers=0,
    )
    
    val_pipeline = DataPipeline(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_frames,
        num_workers=0,
    )
    
    # Model
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)
    
    model = QM9EnergyModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pooling=pooling,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    
    # Loss function
    def loss_fn(pred_td, batch):
        """MSE loss."""
        pred = pred_td["pred", "scalar"]
        target = batch["target", "U0"].to(device)
        return nn.functional.mse_loss(pred, target)
    
    # Hooks
    hooks = [
        MetricsHook(device=device),
        CheckpointHook(checkpoint_dir=checkpoint_dir, save_every_n_epochs=5),
        ProgressBarHook(desc="QM9 Training"),
    ]
    
    # Trainer
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Simple wrapper to move batch to device
    class DataModule:
        def train_dataloader(self):
            for batch in train_pipeline:
                yield batch.to(device)
        
        def val_dataloader(self):
            for batch in val_pipeline:
                yield batch.to(device)
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=lambda params: optimizer,
        hooks=hooks,
    )
    
    trainer.train(
        datamodule=DataModule(),
        max_epochs=num_epochs,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
