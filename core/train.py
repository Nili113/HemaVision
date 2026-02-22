"""
HemaVision Training Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete training loop with:
  - Weighted BCE loss (class imbalance handling)
  - ReduceLROnPlateau scheduler
  - Early stopping
  - Model checkpointing (best validation AUC)
  - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)

Training Flow:
┌────────────┐     ┌────────────┐      ┌────────────┐
│  Setup     │ ──▶ │  Training  │ ──▶ │  Evaluate  │
│  optimizer,│     │  epoch     │      │  on test   │
│  loss, etc.│     │  loop      │      │  set       │
└────────────┘     └─────┬──────┘      └────────────┘
                         │
                    ┌────▼─────┐
                    │ Validate │
                    │ + check  │──▶ Save best model
                    │ patience │
                    └──────────┘

Author: Firoj
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from core.model import DualStreamFusionModel
from utils.config import HemaVisionConfig, get_config

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute all training metrics."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_auc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "learning_rate": [],
        }

    def update(self, epoch_metrics: Dict):
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def get_best_epoch(self, metric: str = "val_auc") -> int:
        values = self.history.get(metric, [])
        if not values:
            return 0
        return int(np.argmax(values))

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Training history saved to {path}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation AUC and stops training if no improvement
    is seen for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best: {self.best_score:.4f}"
                )
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class AMLTrainer:
    """
    Complete training pipeline for the DualStreamFusionModel.

    Handles:
    - Optimizer, loss, scheduler setup
    - Training loop with gradient accumulation support
    - Validation with comprehensive metrics
    - Model checkpointing (best validation AUC)
    - Early stopping
    - Learning rate scheduling

    Example:
        >>> trainer = AMLTrainer(model, train_loader, val_loader, config)
        >>> history = trainer.train()
        >>> test_metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: DualStreamFusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[HemaVisionConfig] = None,
        pos_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.config = config or get_config()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device is not None:
            self.device = device
        elif self.config.inference.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.inference.device)

        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Loss function — weighted BCE for class imbalance
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        logger.info(f"Weighted BCE loss with pos_weight={pos_weight:.2f}")

        # Optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Maximize AUC
            patience=self.config.training.scheduler_patience,
            factor=self.config.training.scheduler_factor,
            min_lr=self.config.training.scheduler_min_lr,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience
        )

        # Metrics tracker
        self.metrics = MetricsTracker()

        # Best model state
        self.best_val_auc = 0.0
        self.best_model_state = None

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            Dict with training history and final metrics.
        """
        num_epochs = self.config.training.epochs
        logger.info(f"Starting training for {num_epochs} epochs...")

        total, trainable = self.model.count_parameters()
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_loss, train_acc = self._train_one_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update metrics
            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc_roc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "learning_rate": current_lr,
            }
            self.metrics.update(epoch_metrics)

            # Scheduler step (monitoring AUC)
            self.scheduler.step(val_metrics["auc_roc"])

            # Model checkpointing
            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._save_checkpoint(epoch, val_metrics)
                marker = " ★ NEW BEST"
            else:
                marker = ""

            # Epoch summary
            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | "
                f"Acc: {train_acc:.3f}/{val_metrics['accuracy']:.3f} | "
                f"AUC: {val_metrics['auc_roc']:.4f} | "
                f"F1: {val_metrics['f1']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"{elapsed:.1f}s{marker}"
            )

            # Early stopping check
            if self.early_stopping(val_metrics["auc_roc"]):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training complete in {total_time / 60:.1f} minutes. "
            f"Best validation AUC: {self.best_val_auc:.4f}"
        )

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights.")

        # Save training history
        history_path = self.config.paths.results_dir / "training_history.json"
        self.metrics.save(history_path)

        return self.metrics.history

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (images, tabular, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, tabular)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        """
        Validate the model on the validation set.

        Returns:
            Dict with all validation metrics.
        """
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        for images, tabular, labels in self.val_loader:
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images, tabular)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = running_loss / max(len(self.val_loader), 1)

        # Compute metrics
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        all_probs_np = np.array(all_probs)

        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc = 0.5  # If only one class present

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(all_labels_np, all_preds_np),
            "precision": precision_score(all_labels_np, all_preds_np, zero_division=0),
            "recall": recall_score(all_labels_np, all_preds_np, zero_division=0),
            "f1": f1_score(all_labels_np, all_preds_np, zero_division=0),
            "auc_roc": auc,
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on the test set with comprehensive metrics.

        Returns:
            Dict with all test metrics including confusion matrix.
        """
        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        for images, tabular, labels in test_loader:
            images = images.to(self.device)
            tabular = tabular.to(self.device)

            logits = self.model(images, tabular)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        all_probs_np = np.array(all_probs)

        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except ValueError:
            auc = 0.5

        cm = confusion_matrix(all_labels_np, all_preds_np)
        report = classification_report(
            all_labels_np, all_preds_np,
            target_names=["Normal", "AML Blast"],
            output_dict=True,
            zero_division=0,
        )

        results = {
            "accuracy": accuracy_score(all_labels_np, all_preds_np),
            "precision": precision_score(all_labels_np, all_preds_np, zero_division=0),
            "recall": recall_score(all_labels_np, all_preds_np, zero_division=0),
            "f1": f1_score(all_labels_np, all_preds_np, zero_division=0),
            "auc_roc": auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(all_labels_np),
            "positive_rate": float(all_labels_np.mean()),
        }

        # Pretty print results
        logger.info(
            f"\n{'━' * 50}\n"
            f"  TEST SET RESULTS\n"
            f"{'━' * 50}\n"
            f"  Accuracy:  {results['accuracy']:.4f}\n"
            f"  Precision: {results['precision']:.4f}\n"
            f"  Recall:    {results['recall']:.4f}\n"
            f"  F1 Score:  {results['f1']:.4f}\n"
            f"  AUC-ROC:   {results['auc_roc']:.4f}\n"
            f"  Confusion Matrix:\n"
            f"    {cm}\n"
            f"{'━' * 50}"
        )

        # Save results
        results_path = self.config.paths.results_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {results_path}")

        return results

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.config.paths.checkpoints_dir / "best_model.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auc": self.best_val_auc,
            "metrics": metrics,
            "config": {
                "backbone": self.config.model.backbone,
                "num_tabular_features": self.model.num_tabular_features,
                "fusion_dim": self.model.fusion_dim,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Checkpoint saved: epoch={epoch}, "
            f"val_auc={metrics['auc_roc']:.4f}"
        )

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        num_tabular_features: int,
        device: Optional[torch.device] = None,
    ) -> DualStreamFusionModel:
        """
        Load a model from checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file
            num_tabular_features: Number of tabular input features
            device: Device to load the model onto

        Returns:
            Loaded DualStreamFusionModel
        """
        device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = DualStreamFusionModel(
            num_tabular_features=num_tabular_features
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        logger.info(
            f"Model loaded from {checkpoint_path} "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"val_auc={checkpoint.get('best_val_auc', '?'):.4f})"
        )

        return model


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """
    Plot training curves: loss, accuracy, AUC, and learning rate.

    Generates a 2x2 grid of plots saved to disk.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not available. Skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HemaVision Training History", fontsize=16, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    ax.set_title("Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    ax.set_title("Accuracy", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    # AUC-ROC
    ax = axes[1, 0]
    ax.plot(epochs, history["val_auc"], "g-", label="Val AUC", linewidth=2)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Target (0.95)")
    ax.set_title("Validation AUC-ROC", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.legend()

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history["learning_rate"], "m-", linewidth=2)
    ax.set_title("Learning Rate", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")
    plt.close()
