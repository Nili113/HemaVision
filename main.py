"""
HemaVision Main Orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
End-to-end pipeline for AML detection:
  1. Data loading & preprocessing
  2. Model initialization
  3. Training with validation
  4. Test set evaluation
  5. Grad-CAM explainability
  6. Model export
  7. Results summary

Usage:
    python main.py                                          # Full pipeline
    python main.py --data-root /path/to/data                # Custom data path
    python main.py --epochs 20 --batch-size 16              # Override hyperparams
    python main.py --eval-only --checkpoint best_model.pt   # Eval only

Author: Firoj
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from core.data_loader import AMLDataPreprocessor
from core.dataset import create_dataloaders
from core.model import DualStreamFusionModel
from core.train import AMLTrainer, plot_training_history
from core.gradcam import GradCAM
from utils.config import get_config, HemaVisionConfig

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("HemaVision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HemaVision — Multimodal AML Detection Pipeline"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to AML-Cytomorphology_LMU dataset"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gradcam-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> HemaVisionConfig:
    """Build configuration from arguments."""
    config = get_config()

    if args.data_root:
        config.paths.data_root = Path(args.data_root)
        config.paths.__post_init__()

    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.unfreeze_backbone:
        config.model.freeze_backbone = False

    config.training.random_seed = args.seed
    config.inference.gradcam_num_samples = args.gradcam_samples
    config.paths.create_directories()

    return config


def run_pipeline(config: HemaVisionConfig, eval_only: bool = False,
                 checkpoint_path: str = None):
    """
    Execute the full HemaVision pipeline.

    Pipeline:
    ┌─────────┐   ┌─────────┐    ┌─────────┐     ┌──────────┐   ┌─────────┐
    │  Load   │──▶│  Build  │──▶│  Train  │──▶ │ Evaluate │──▶│ GradCAM │
    │  Data   │   │  Model  │    │         │     │  Test    │   │         │
    └─────────┘   └─────────┘    └─────────┘     └──────────┘   └─────────┘
    """
    start_time = time.time()

    logger.info(config.summary())

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Data Loading & Preprocessing
    # ──────────────────────────────────────────────────────────
    logger.info("Phase 1: Loading and preprocessing data...")

    preprocessor = AMLDataPreprocessor(config)
    unified_df = preprocessor.create_unified_dataframe()
    feature_names, num_features = preprocessor.prepare_features()
    train_df, val_df, test_df = preprocessor.split_data()
    pos_weight = preprocessor.get_class_weights()

    logger.info(f"Data summary:\n{json.dumps(preprocessor.get_split_summary(), indent=2, default=str)}")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Create DataLoaders
    # ──────────────────────────────────────────────────────────
    logger.info("Phase 2: Creating DataLoaders...")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tabular_columns=feature_names,
        config=config,
    )

    # ──────────────────────────────────────────────────────────
    # PHASE 3: Model Initialization
    # ──────────────────────────────────────────────────────────
    logger.info("Phase 3: Initializing model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and Path(checkpoint_path).exists():
        model = AMLTrainer.load_checkpoint(checkpoint_path, num_features, device)
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        model = DualStreamFusionModel(
            num_tabular_features=num_features,
            config=config.model,
        )
        logger.info(model.summary())

    # ──────────────────────────────────────────────────────────
    # PHASE 4: Training
    # ──────────────────────────────────────────────────────────
    if not eval_only:
        logger.info("Phase 4: Training...")

        trainer = AMLTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            pos_weight=pos_weight,
            device=device,
        )
        history = trainer.train()

        # Find optimal classification threshold on validation set
        optimal_threshold = trainer.find_optimal_threshold(val_loader)

        # Plot training curves
        plot_path = config.paths.results_dir / "training_history.png"
        plot_training_history(history, save_path=plot_path)

        # ── PHASE 5: Test Evaluation ─────────────────────────
        logger.info("Phase 5: Evaluating on test set...")
        test_results = trainer.evaluate(test_loader, threshold=optimal_threshold)
    else:
        logger.info("Skipping training (eval-only mode).")
        trainer = AMLTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            pos_weight=pos_weight,
            device=device,
        )
        # Try to recover optimal threshold from checkpoint
        optimal_threshold = 0.5
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            optimal_threshold = ckpt.get("optimal_threshold", 0.5)
            logger.info(f"Using threshold from checkpoint: {optimal_threshold:.4f}")
        else:
            optimal_threshold = trainer.find_optimal_threshold(val_loader)
        test_results = trainer.evaluate(test_loader, threshold=optimal_threshold)

    # ──────────────────────────────────────────────────────────
    # PHASE 6: Grad-CAM Explainability
    # ──────────────────────────────────────────────────────────
    logger.info("Phase 6: Generating Grad-CAM visualizations...")

    try:
        gradcam = GradCAM(model, target_layers=config.inference.gradcam_target_layers)
        # Use optimal threshold so Grad-CAM labels match test evaluation
        threshold = optimal_threshold if 'optimal_threshold' in dir() else 0.5
        gradcam.batch_visualize(
            dataloader=test_loader,
            num_samples=config.inference.gradcam_num_samples,
            save_dir=str(config.paths.gradcam_dir),
            threshold=threshold,
        )
        logger.info(
            f"Saved {config.inference.gradcam_num_samples} Grad-CAM "
            f"visualizations to {config.paths.gradcam_dir}"
        )
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    # ──────────────────────────────────────────────────────────
    # PHASE 7: Export & Summary
    # ──────────────────────────────────────────────────────────
    logger.info("Phase 7: Exporting model and generating summary...")

    # Save final model (.pt) — include optimal threshold for inference
    model_path = config.paths.checkpoints_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimal_threshold": optimal_threshold if 'optimal_threshold' in dir() else 0.5,
        "config": {
            "backbone": config.model.backbone,
            "num_tabular_features": num_features,
            "tabular_feature_names": feature_names,
            "feature_type": "morphological",
        },
    }, model_path)
    logger.info(f"Final model saved to {model_path}")

    # Export ONNX model
    onnx_path = config.paths.checkpoints_dir / "final_model.onnx"
    try:
        import importlib
        if importlib.util.find_spec("onnx") is None:
            logger.warning(
                "ONNX package not installed. Run: pip install onnx onnxruntime\n"
                "  Skipping ONNX export."
            )
            raise ImportError("onnx not installed")

        model.eval()
        model_cpu = model.cpu()
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_tabular = torch.randn(1, num_features)
        torch.onnx.export(
            model_cpu,
            (dummy_image, dummy_tabular),
            str(onnx_path),
            input_names=["image", "tabular"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "tabular": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
        )
        # Move model back to original device
        model.to(device)
        logger.info(f"ONNX model exported to {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        model.to(device)

    # Generate results summary
    elapsed = time.time() - start_time
    summary = {
        "project": "HemaVision — AML Detection System",
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": round(elapsed / 60, 2),
        "dataset": {
            "total_images": len(unified_df),
            "total_patients": unified_df["patient_id"].nunique(),
            "splits": preprocessor.get_split_summary(),
        },
        "model": {
            "architecture": "DualStreamFusionModel",
            "backbone": config.model.backbone,
            "num_tabular_features": num_features,
            "total_params": model.count_parameters()[0],
            "trainable_params": model.count_parameters()[1],
        },
        "test_results": test_results,
        "config": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "seed": config.training.random_seed,
        },
    }

    summary_path = config.paths.results_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        f"\n{'═' * 60}\n"
        f"  HemaVision Pipeline Complete!\n"
        f"{'═' * 60}\n"
        f"  Duration:   {elapsed / 60:.1f} minutes\n"
        f"  Test AUC:   {test_results.get('auc_roc', 'N/A')}\n"
        f"  Test F1:    {test_results.get('f1', 'N/A')}\n"
        f"  Model:      {model_path}\n"
        f"  ONNX:       {onnx_path}\n"
        f"  Results:    {summary_path}\n"
        f"  Grad-CAMs:  {config.paths.gradcam_dir}\n"
        f"{'═' * 60}\n"
    )

    return summary


def main():
    args = parse_args()
    config = build_config(args)

    # Set random seed for reproducibility
    torch.manual_seed(config.training.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.random_seed)

    run_pipeline(
        config=config,
        eval_only=args.eval_only,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
