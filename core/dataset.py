"""
HemaVision Dataset & DataLoader
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom PyTorch Dataset for the multimodal AML pipeline.

Each sample yields a triplet:
  (image_tensor, tabular_tensor, label)

Includes separate augmentation pipelines for training and evaluation.

Author: Firoj
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from utils.config import AugmentationConfig, HemaVisionConfig, get_config

logger = logging.getLogger(__name__)


def get_train_transforms(config: AugmentationConfig = None) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Applies stochastic augmentations to improve generalization:
    - Random rotation, flips for orientation invariance
    - Color jitter for staining variation robustness
    - ImageNet normalization for transfer learning
    """
    config = config or AugmentationConfig()
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.RandomHorizontalFlip(config.horizontal_flip_prob),
        transforms.RandomVerticalFlip(config.vertical_flip_prob),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(config.normalize_mean),
            std=list(config.normalize_std),
        ),
    ])


def get_eval_transforms(config: AugmentationConfig = None) -> transforms.Compose:
    """
    Validation/Test transform pipeline.

    Minimal transforms — just resize and normalize.
    No stochastic augmentations to ensure deterministic evaluation.
    """
    config = config or AugmentationConfig()
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(config.normalize_mean),
            std=list(config.normalize_std),
        ),
    ])


class AMLDataset(Dataset):
    """
    PyTorch Dataset for the AML multimodal pipeline.

    Each __getitem__ returns:
        image:   (3, 224, 224) — augmented cell microscopy image
        tabular: (num_features,) — normalized clinical features
        label:   scalar — 0 (normal) or 1 (AML blast)

    Args:
        dataframe:       DataFrame with columns [image_path, label, + tabular features]
        tabular_columns: List of column names for tabular features
        transform:       Image transform pipeline
        image_col:       Column name for image paths
        label_col:       Column name for labels
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tabular_columns: List[str],
        transform: Optional[Callable] = None,
        image_col: str = "image_path",
        label_col: str = "label",
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tabular_columns = tabular_columns
        self.transform = transform or get_eval_transforms()
        self.image_col = image_col
        self.label_col = label_col

        # Pre-extract tabular features as numpy array for speed
        self.tabular_data = self.dataframe[tabular_columns].values.astype(np.float32)
        self.labels = self.dataframe[label_col].values.astype(np.float32)
        self.image_paths = self.dataframe[image_col].values

        logger.info(
            f"AMLDataset created: {len(self)} samples, "
            f"{len(tabular_columns)} tabular features, "
            f"positive rate: {self.labels.mean():.3f}"
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a blank image if file is missing (graceful degradation)
            logger.warning(f"Image not found: {img_path}, using blank image.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Apply transforms
        image = self.transform(image)

        # Tabular features
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        return image, tabular, label

    def get_labels(self) -> np.ndarray:
        """Return all labels (for computing class weights / sampler)."""
        return self.labels


def create_weighted_sampler(dataset: AMLDataset) -> WeightedRandomSampler:
    """
    Create a weighted random sampler to handle class imbalance.

    Over-samples the minority class so each batch has balanced representation.
    """
    labels = dataset.get_labels()
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tabular_columns: List[str],
    config: Optional[HemaVisionConfig] = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_df:           Training DataFrame
        val_df:             Validation DataFrame
        test_df:            Test DataFrame
        tabular_columns:    List of tabular feature column names
        config:             Configuration object
        use_weighted_sampler: Whether to use weighted sampling for training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = config or get_config()
    aug_config = config.augmentation
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers

    # Create datasets
    train_dataset = AMLDataset(
        dataframe=train_df,
        tabular_columns=tabular_columns,
        transform=get_train_transforms(aug_config),
    )

    val_dataset = AMLDataset(
        dataframe=val_df,
        tabular_columns=tabular_columns,
        transform=get_eval_transforms(aug_config),
    )

    test_dataset = AMLDataset(
        dataframe=test_df,
        tabular_columns=tabular_columns,
        transform=get_eval_transforms(aug_config),
    )

    # Create sampler for training (handles class imbalance)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        train_shuffle = False  # Sampler and shuffle are mutually exclusive

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created:\n"
        f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches\n"
        f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches\n"
        f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
