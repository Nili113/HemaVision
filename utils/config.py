"""
HemaVision Configuration Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralized configuration for the AML detection pipeline.
All hyperparameters, paths, and settings in one place.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PathConfig:
    """All path-related configuration."""
    # Root directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_root: Path = field(default_factory=lambda: Path("AML-Cytomorphology_LMU"))

    # Data paths
    images_dir: Path = field(init=False)
    patient_csv: Path = field(init=False)

    # Output paths
    output_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    gradcam_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Known cell-type folder names from the AML-Cytomorphology dataset
    _KNOWN_CELL_TYPES = {
        'BLA', 'MYO', 'NGS', 'LYT', 'MON', 'EOS', 'BAS', 'EBO',
        'NGB', 'KSC', 'LYA', 'MMZ', 'MOB', 'MYB', 'NIF', 'OTH',
        'PEB', 'PLM', 'PMB', 'PMO', 'ART',
    }

    def __post_init__(self):
        # Auto-detect the actual images directory.
        # Kaggle datasets have varying nesting depths, e.g.:
        #   data_root/images/BLA/...
        #   data_root/PKG - AML-Cytomorphology/AML-Cytomorphology/MYO/...
        #   data_root/BLA/...
        self.images_dir = self._find_images_dir()
        self.patient_csv = self.data_root / "patient_data.csv"
        self.output_dir = self.project_root / "outputs"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.gradcam_dir = self.output_dir / "gradcam_results"
        self.logs_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"

    def _find_images_dir(self) -> Path:
        """Walk data_root to find the directory containing cell-type folders."""
        # 1. Check data_root/images (standard layout)
        candidate = self.data_root / "images"
        if candidate.exists() and self._has_cell_type_dirs(candidate):
            return candidate

        # 2. Check data_root itself
        if self._has_cell_type_dirs(self.data_root):
            return self.data_root

        # 3. Walk the tree (handles arbitrary nesting like PKG - .../AML-...)
        if self.data_root.exists():
            for root, dirs, _files in os.walk(self.data_root):
                root_path = Path(root)
                matching = {d for d in dirs if d.upper() in self._KNOWN_CELL_TYPES}
                if len(matching) >= 3:  # At least 3 known cell types
                    return root_path

        # 4. Fallback to data_root (will fail later with a clear error)
        return self.data_root

    def _has_cell_type_dirs(self, path: Path) -> bool:
        """Check if a directory directly contains known cell-type subdirs."""
        if not path.is_dir():
            return False
        children = {d.name.upper() for d in path.iterdir() if d.is_dir()}
        return len(children & self._KNOWN_CELL_TYPES) >= 3

    def create_directories(self):
        """Create all necessary output directories."""
        for path in [self.output_dir, self.checkpoints_dir,
                     self.gradcam_dir, self.logs_dir, self.results_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Visual stream
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    visual_feature_dim: int = 2048

    # Tabular stream
    tabular_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    tabular_dropout: float = 0.3

    # Fusion & classifier
    fusion_hidden_dim: int = 256
    classifier_dropout: float = 0.5
    num_classes: int = 1  # Binary classification (sigmoid output)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Core
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7

    # Early stopping
    early_stopping_patience: int = 10

    # Data split ratios (patient-level)
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    # Reproducibility
    random_seed: int = 42


@dataclass
class AugmentationConfig:
    """Image augmentation configuration."""
    image_size: Tuple[int, int] = (224, 224)
    rotation_degrees: int = 20
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.05

    # ImageNet normalization
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class InferenceConfig:
    """Inference and deployment configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    gradcam_target_layer: str = "layer4"
    gradcam_num_samples: int = 20
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860


@dataclass
class HemaVisionConfig:
    """Master configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        self.paths.create_directories()

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        return (
            f"\n{'━' * 60}\n"
            f"  HemaVision Configuration\n"
            f"{'━' * 60}\n"
            f"  Model:     {self.model.backbone} (pretrained={self.model.pretrained})\n"
            f"  Freeze:    {self.model.freeze_backbone}\n"
            f"  Epochs:    {self.training.epochs}\n"
            f"  Batch:     {self.training.batch_size}\n"
            f"  LR:        {self.training.learning_rate}\n"
            f"  Split:     {self.training.train_ratio}/{self.training.val_ratio}/{self.training.test_ratio}\n"
            f"  Seed:      {self.training.random_seed}\n"
            f"  Image:     {self.augmentation.image_size}\n"
            f"  Data Root: {self.paths.data_root}\n"
            f"  Output:    {self.paths.output_dir}\n"
            f"{'━' * 60}\n"
        )


# Global singleton config
def get_config(**overrides) -> HemaVisionConfig:
    """
    Get the global configuration instance.

    Args:
        **overrides: Override specific config values.
                     Example: get_config(data_root=Path("/new/data"))

    Returns:
        HemaVisionConfig: The configuration object.
    """
    config = HemaVisionConfig()

    # Apply overrides
    if "data_root" in overrides:
        config.paths.data_root = Path(overrides["data_root"])
        config.paths.__post_init__()
        config.paths.create_directories()

    if "output_dir" in overrides:
        config.paths.output_dir = Path(overrides["output_dir"])
        config.paths.__post_init__()
        config.paths.create_directories()

    return config
