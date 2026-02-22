# Kaggle Training Notebook for HemaVision

Complete Kaggle notebook code. Copy-paste each cell into a Kaggle notebook. Enable **GPU T4 x2** or **P100** accelerator and turn **Internet ON**.

> **Note:** This version uses the **hybrid multimodal architecture** — the model fuses deep CNN features (ResNet50) with 20 handcrafted morphological features (geometry, nucleus, colour, texture, shape) extracted from each cell image. Feature extraction adds ~15 minutes to preprocessing but creates a genuinely multimodal model.

---

## Cell 1: Clone the Repo & Install Dependencies

```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/Nili113/HemaVision.git
%cd HemaVision

# Install backend/training dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q timm albumentations opencv-python-headless pillow matplotlib scikit-learn pandas tqdm seaborn
!pip install -q scikit-image  # Required for morphological feature extraction (GLCM, regionprops)
!pip install -q onnx onnxruntime  # Required for ONNX model export

# Check if requirements.txt exists and install from it
import os
if os.path.exists("requirements.txt"):
    !pip install -q -r requirements.txt
elif os.path.exists("backend/requirements.txt"):
    !pip install -q -r backend/requirements.txt
```

---

## Cell 2: Verify GPU & Imports

```python
# Cell 2: Verify GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow.")
```

---

## Cell 3: Explore Repo Structure

```python
# Cell 3: Understand the repo structure
import os

for root, dirs, files in os.walk("/kaggle/working/HemaVision"):
    # Skip hidden dirs, node_modules, __pycache__, .git
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.venv')]
    level = root.replace("/kaggle/working/HemaVision", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")
```

---

## Cell 4: Download / Prepare Dataset

```python
# Cell 4: Dataset Preparation
# ============================================================
# DATASET: AML-Cytomorphology_LMU (Munich / LMU)
#
# Paper:  Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019).
#         "A Single-cell Morphological Dataset of Leukocytes from AML
#         Patients and Non-malignant Controls."
#
# Source: The Cancer Imaging Archive (TCIA)
#         https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
#         DOI: 10.7937/tcia.2019.36f5o9ld
#
# Kaggle: binilj04/aml-cytomorphology   (~6.2 GB, 15 classes)
#         https://www.kaggle.com/datasets/binilj04/aml-cytomorphology
#         Also: umarsani1605/aml-cytomorphology (identical data)
#
# ⚠️ IMPORTANT: All Kaggle uploads renamed the BLA (Blast) folder to MYO.
#    Evidence: BLA=3,268 in the paper → MYO=3,268 in Kaggle (identical count,
#    all other 14 classes match exactly). The code handles this automatically.
#
# Contains: 18,365 single-cell microscopy images (400×400 px, TIFF)
#           200 patients (100 AML + 100 controls), 15 classes
#           May-Grünwald-Giemsa stain, peripheral blood smears
#
# Structure after extraction:
#   PKG - AML-Cytomorphology/
#   └── AML-Cytomorphology/
#       ├── MYO/   (Blasts — renamed from BLA by uploader, AML positive)
#       ├── NGS/   (Neutrophils - segmented, 8,484 images)
#       ├── LYT/   (Lymphocytes - typical, 3,937 images)
#       ├── MON/   (Monocytes, 1,789 images)
#       └── ...    (15 cell-type subdirectories)
#   (No patient_data.csv — synthetic metadata generated automatically)
# ============================================================

import os
import shutil

# ============================================================
# OPTION A (RECOMMENDED): Add via Kaggle "Add Data" sidebar
#   1. Click "+ Add Data" in the right sidebar
#   2. Search: "binilj04/aml-cytomorphology"
#      (or "umarsani1605/aml-cytomorphology" — same data)
#   3. Select: "AML-Cytomorphology-WBC" (~6.2 GB)
#   4. Click "Add" → data appears under /kaggle/input/
#
# NOTE: The blast class is named MYO (not BLA) in all Kaggle uploads.
#       The code handles this automatically.
# ============================================================

# Try multiple known dataset paths (Kaggle uses different path formats)
# New format: /kaggle/input/datasets/<owner>/<dataset-name>
# Old format: /kaggle/input/<dataset-name>
import glob as _glob

KAGGLE_CANDIDATES = [
    # New Kaggle path format (owner/dataset)
    "/kaggle/input/datasets/binilj04/aml-cytomorphology",
    "/kaggle/input/datasets/umarsani1605/aml-cytomorphology",
    "/kaggle/input/datasets/gchan357/human-aml-cytomorphology-dataset",
    "/kaggle/input/datasets/walkersneps/aml-cytomorphology-lmu",
    # Old Kaggle path format (dataset-name only)
    "/kaggle/input/aml-cytomorphology",
    "/kaggle/input/aml-cytomorphology-lmu",
    "/kaggle/input/aml-cytomorphology-2",
    "/kaggle/input/human-aml-cytomorphology-dataset",
]

# Also scan /kaggle/input/ for any matching dataset
for p in _glob.glob("/kaggle/input/**/BLA", recursive=True):
    parent = os.path.dirname(p)
    if parent not in KAGGLE_CANDIDATES:
        KAGGLE_CANDIDATES.insert(0, parent)  # Prioritize auto-detected

LOCAL_DATASET_PATH = "/kaggle/working/HemaVision/AML-Cytomorphology_LMU"

KAGGLE_DATASET_PATH = None
for candidate in KAGGLE_CANDIDATES:
    if os.path.exists(candidate):
        KAGGLE_DATASET_PATH = candidate
        print(f"✓ Dataset found at: {candidate}")
        break

if KAGGLE_DATASET_PATH:
    # Symlink instead of copy to save disk space
    # Ensure parent directory exists (in case Cell 1 clone is skipped)
    os.makedirs(os.path.dirname(LOCAL_DATASET_PATH), exist_ok=True)
    # Remove stale/broken symlink if it exists from a previous run
    if os.path.islink(LOCAL_DATASET_PATH):
        os.remove(LOCAL_DATASET_PATH)
    if not os.path.exists(LOCAL_DATASET_PATH):
        os.symlink(KAGGLE_DATASET_PATH, LOCAL_DATASET_PATH)
        print(f"Symlinked → {LOCAL_DATASET_PATH}")
    else:
        print(f"Already exists: {LOCAL_DATASET_PATH}")
else:
    print("✗ Dataset not found!")
    print("\nPlease add it via the 'Add Data' sidebar:")
    print('  1. Click "+ Add Data"')
    print('  2. Search: "binilj04/aml-cytomorphology"')
    print('  3. Select "AML-Cytomorphology-WBC" and click Add')
    print("\nAlternative (Kaggle CLI):")
    print("  !kaggle datasets download -d binilj04/aml-cytomorphology")
    print("  !unzip -q aml-cytomorphology.zip -d AML-Cytomorphology_LMU")

# Verify dataset structure
if os.path.exists(LOCAL_DATASET_PATH):
    print(f"\nDataset contents:")
    for item in sorted(os.listdir(LOCAL_DATASET_PATH)):
        full = os.path.join(LOCAL_DATASET_PATH, item)
        if os.path.isdir(full):
            # Check for nested image subdirectories
            sub_items = os.listdir(full)
            sub_dirs = [s for s in sub_items if os.path.isdir(os.path.join(full, s))]
            if sub_dirs:
                total_images = sum(
                    len(os.listdir(os.path.join(full, sd)))
                    for sd in sub_dirs
                )
                print(f"  {item}/  ({len(sub_dirs)} classes, {total_images:,} images)")
                for sd in sorted(sub_dirs):
                    n = len(os.listdir(os.path.join(full, sd)))
                    print(f"    {sd}/ ({n:,} images)")
            else:
                print(f"  {item}/  ({len(sub_items)} items)")
        else:
            size = os.path.getsize(full) / 1e6
            print(f"  {item}  ({size:.1f} MB)")
```

---

## Cell 5: Verify Dataset & Class Distribution

```python
# Cell 5: Verify the AML-Cytomorphology_LMU dataset is ready

import os
import matplotlib.pyplot as plt
from collections import Counter

DATASET_ROOT = "/kaggle/working/HemaVision/AML-Cytomorphology_LMU"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")

# Auto-detect: Kaggle uploads have varying folder structures
# e.g. images/, data/data/, AML-Cytomorphology_LMU/data/data/, etc.
# We find whichever directory contains the cell-type subdirectories.
KNOWN_CELL_TYPES = {'BLA', 'LYT', 'NGS', 'MON', 'EOS', 'BAS', 'EBO', 'MYO', 'NGB'}

if not os.path.exists(IMAGES_DIR) or not any(
    d in KNOWN_CELL_TYPES for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))
):
    print("Standard images/ path not found or incomplete, scanning...")
    found = False
    for root, dirs, files in os.walk(DATASET_ROOT):
        matching = set(dirs) & KNOWN_CELL_TYPES
        if len(matching) >= 3:  # At least 3 known cell types
            IMAGES_DIR = root
            print(f"Auto-detected images at: {IMAGES_DIR}")
            print(f"  Found classes: {sorted(matching)}")
            found = True
            break
    if not found:
        raise FileNotFoundError(
            f"Could not find cell-type folders anywhere under {DATASET_ROOT}\n"
            "Go back to Cell 4 and ensure the dataset is added correctly."
        )

# CRITICAL CHECK: Verify blast class exists (BLA in TCIA, MYO in Kaggle)
# All Kaggle uploads renamed BLA → MYO (folder + files).
# Evidence: BLA=3,268 in paper, MYO=3,268 in Kaggle; all other counts identical.
has_blast = os.path.exists(os.path.join(IMAGES_DIR, 'BLA')) or \
            os.path.exists(os.path.join(IMAGES_DIR, 'MYO'))
blast_folder = 'BLA' if os.path.exists(os.path.join(IMAGES_DIR, 'BLA')) else 'MYO'
if not has_blast:
    print("\n" + "!" * 60)
    print("WARNING: Neither BLA nor MYO (blast) class found in this dataset!")
    print("Cannot train AML detection without blast cell images.")
    print("\nFix: Add 'binilj04/aml-cytomorphology' via Add Data sidebar")
    print("!" * 60 + "\n")
else:
    blast_count = len(os.listdir(os.path.join(IMAGES_DIR, blast_folder)))
    print(f"\n✓ Blast class found as '{blast_folder}/' ({blast_count:,} images)")
    if blast_folder == 'MYO':
        print("  (Kaggle uploads renamed BLA→MYO; code handles this automatically)")

# Count images per class
class_counts = {}
for class_name in sorted(os.listdir(IMAGES_DIR)):
    class_path = os.path.join(IMAGES_DIR, class_name)
    if os.path.isdir(class_path):
        n = len([f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))])
        class_counts[class_name] = n

total = sum(class_counts.values())
print(f"Total images: {total:,}")
print(f"Classes: {len(class_counts)}")
print(f"\nPer-class breakdown:")

# AML-positive class (BLA in original TCIA, MYO in Kaggle uploads)
BLAST_CLASSES = {'BLA', 'MYO'}  # Blasts = AML positive (both names)

aml_count = 0
normal_count = 0
for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    marker = " ← AML (positive)" if cls in BLAST_CLASSES else ""
    print(f"  {cls:5s}: {count:>6,} images{marker}")
    if cls in BLAST_CLASSES:
        aml_count += count
    else:
        normal_count += count

print(f"\nBinary split:  AML={aml_count:,}  |  Normal={normal_count:,}")
print(f"Class ratio:   AML={aml_count/total*100:.1f}%  |  Normal={normal_count/total*100:.1f}%")

# Plot distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart of all 21 classes
classes = list(class_counts.keys())
counts = list(class_counts.values())
colors = ['#D62828' if c in BLAST_CLASSES else '#0A2463' for c in classes]
ax1.barh(classes, counts, color=colors)
ax1.set_xlabel('Number of Images')
ax1.set_title(f'Cell Type Distribution ({len(class_counts)} classes)')
ax1.invert_yaxis()

# Pie chart of binary split
ax2.pie([aml_count, normal_count], labels=['AML Blast', 'Normal'],
        colors=['#D62828', '#34C759'], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 14})
ax2.set_title('Binary Classification Split')

plt.tight_layout()
plt.savefig('/kaggle/working/HemaVision/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Dataset verified and ready for training!")
print(f"  Images dir: {IMAGES_DIR}")
```

---

## Cell 6: Discover and Run Training Script

```python
# Cell 6: Find training-related files in the repo
import glob

print("=" * 60)
print("Searching for training scripts...")
print("=" * 60)

# Search for common training file patterns
patterns = [
    "**/*train*.py",
    "**/*Train*.py",
    "**/train*.ipynb",
    "**/*model*.py",
    "**/*Model*.py",
    "**/config*.py",
    "**/config*.yaml",
    "**/config*.yml",
    "**/config*.json",
    "**/*.pth",
    "**/*.pt",
    "**/*.h5",
    "**/*.onnx",
]

base = "/kaggle/working/HemaVision"
for pattern in patterns:
    matches = glob.glob(os.path.join(base, pattern), recursive=True)
    for m in matches:
        rel = os.path.relpath(m, base)
        size = os.path.getsize(m) / 1e6
        print(f"  {rel} ({size:.2f} MB)")

print("\n")
print("=" * 60)
print("Searching for Python files with model/training logic...")
print("=" * 60)

for pyfile in glob.glob(os.path.join(base, "**/*.py"), recursive=True):
    if any(skip in pyfile for skip in ['node_modules', '__pycache__', '.git', 'venv']):
        continue
    try:
        with open(pyfile, 'r') as f:
            content = f.read().lower()
        keywords = ['def train', 'model.fit', 'criterion', 'optimizer', 'dataloader', 'epoch', 'backward()']
        found = [k for k in keywords if k in content]
        if found:
            rel = os.path.relpath(pyfile, base)
            print(f"  {rel} -- keywords: {found}")
    except:
        pass
```

---

## Cell 7: Inspect the Actual Training Code

```python
# Cell 7: Read and display the main training file content
import glob
import os

base = "/kaggle/working/HemaVision"

# Auto-detect the training script
candidates = []
for pyfile in glob.glob(os.path.join(base, "**/*.py"), recursive=True):
    if any(skip in pyfile for skip in ['node_modules', '__pycache__', '.git', 'venv', 'frontend']):
        continue
    try:
        with open(pyfile, 'r') as f:
            content = f.read()
        score = sum([
            'def train' in content.lower(),
            'optimizer' in content.lower(),
            'epoch' in content.lower(),
            'backward' in content.lower(),
            'dataloader' in content.lower(),
            'criterion' in content.lower(),
        ])
        if score >= 2:
            candidates.append((pyfile, score))
    except:
        pass

candidates.sort(key=lambda x: -x[1])

if candidates:
    best = candidates[0][0]
    print(f"Most likely training script: {os.path.relpath(best, base)}")
    print("=" * 80)
    with open(best, 'r') as f:
        print(f.read())
else:
    print("No training script auto-detected.")
    print("Check Cell 3 output and manually identify the training code.")
```

---

## Cell 8: Run Training via main.py

```python
# Cell 8: Run the HemaVision training pipeline
#
# main.py is the project's orchestrator. It runs:
#   1. Data loading & patient-level splitting (core/data_loader.py)
#   2. Dual-stream model init - ResNet50 + MLP (core/model.py)
#   3. Training with weighted BCE loss (core/train.py)
#   4. Test evaluation (metrics, confusion matrix)
#   5. Grad-CAM explainability visualizations (core/gradcam.py)
#   6. Model export
#
# The --data-root flag points to the AML-Cytomorphology_LMU dataset.

import os
os.chdir("/kaggle/working/HemaVision")

# Resolve dataset path — try symlink first, then known Kaggle input paths
import glob as _glob

DATASET_PATH = None
candidates = [
    "/kaggle/working/HemaVision/AML-Cytomorphology_LMU",  # Symlinked in Cell 4
    # New Kaggle path format
    "/kaggle/input/datasets/binilj04/aml-cytomorphology",
    "/kaggle/input/datasets/umarsani1605/aml-cytomorphology",
    "/kaggle/input/datasets/gchan357/human-aml-cytomorphology-dataset",
    # Old Kaggle path format
    "/kaggle/input/aml-cytomorphology",
    "/kaggle/input/aml-cytomorphology-lmu",
    "/kaggle/input/aml-cytomorphology-2",
]

for candidate in candidates:
    if os.path.exists(candidate):
        DATASET_PATH = candidate
        break

# Last resort: scan for any folder containing BLA
if DATASET_PATH is None:
    bla_dirs = _glob.glob("/kaggle/input/**/BLA", recursive=True)
    if bla_dirs:
        DATASET_PATH = os.path.dirname(bla_dirs[0])
        print(f"Auto-detected dataset at: {DATASET_PATH}")

assert DATASET_PATH is not None, (
    "Dataset not found!\n"
    "Go back to Cell 4 and add the dataset via 'Add Data' sidebar:\n"
    '  Search: "binilj04/aml-cytomorphology" or "umarsani1605/aml-cytomorphology"'
)

print(f"Dataset path: {DATASET_PATH}")
print(f"Running: python main.py --data-root {DATASET_PATH}")
print("=" * 60)

!python main.py --data-root "{DATASET_PATH}" --epochs 50 --batch-size 32
```

---

## Cell 9: Plot Training Curves

```python
# Cell 9: Load training history from disk and plot curves
# main.py saves all metrics to outputs/results/training_history.json
# so we load from there instead of relying on in-memory variables.

import json
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = "/kaggle/working/HemaVision"
RESULTS_DIR = os.path.join(BASE, "outputs", "results")

# ── Load training history ──
history_path = os.path.join(RESULTS_DIR, "training_history.json")
assert os.path.exists(history_path), (
    f"training_history.json not found at {history_path}\n"
    "Make sure Cell 8 (main.py) completed successfully."
)

with open(history_path) as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# ── 4-panel training curves ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("HemaVision Training History", fontsize=18, fontweight="bold", y=0.98)

# 1) Loss
ax = axes[0, 0]
ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
ax.plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2) Accuracy
ax = axes[0, 1]
ax.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
ax.plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# 3) AUC-ROC + F1
ax = axes[1, 0]
ax.plot(epochs, history["val_auc"], "g-", label="Val AUC-ROC", linewidth=2)
ax.plot(epochs, history["val_f1"], "m-", label="Val F1", linewidth=2)
if "val_precision" in history:
    ax.plot(epochs, history["val_precision"], "c--", label="Val Precision", linewidth=1.5, alpha=0.7)
if "val_recall" in history:
    ax.plot(epochs, history["val_recall"], "y--", label="Val Recall", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("Validation Metrics")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4) Learning Rate
ax = axes[1, 1]
ax.plot(epochs, history["learning_rate"], "k-", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule")
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves_notebook.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── Summary ──
best_epoch = int(np.argmax(history["val_auc"])) + 1
print(f"\n{'═' * 50}")
print(f"  Best Epoch: {best_epoch}/{len(history['train_loss'])}")
print(f"  Best Val AUC:  {max(history['val_auc']):.4f}")
print(f"  Best Val F1:   {history['val_f1'][best_epoch - 1]:.4f}")
print(f"  Best Val Acc:  {history['val_acc'][best_epoch - 1]:.4f}")
print(f"{'═' * 50}")
```

---

## Cell 10: Test Results & Confusion Matrix

```python
# Cell 10: Load test results and display confusion matrix + classification report
# main.py evaluates the best model on the held-out test set and saves
# all metrics (including confusion matrix) to test_results.json.

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/kaggle/working/HemaVision"
RESULTS_DIR = os.path.join(BASE, "outputs", "results")

# ── Load test results ──
test_path = os.path.join(RESULTS_DIR, "test_results.json")
assert os.path.exists(test_path), (
    f"test_results.json not found at {test_path}\n"
    "Make sure Cell 8 (main.py) completed successfully."
)

with open(test_path) as f:
    results = json.load(f)

# ── Display headline metrics ──
print("=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  Accuracy:   {results['accuracy']:.4f}")
print(f"  Precision:  {results['precision']:.4f}")
print(f"  Recall:     {results['recall']:.4f}")
print(f"  F1 Score:   {results['f1']:.4f}")
print(f"  AUC-ROC:    {results['auc_roc']:.4f}")
print(f"  Samples:    {results['num_samples']}")
print(f"  Pos. Rate:  {results['positive_rate']:.3f}")
print("=" * 55)

# ── Classification Report ──
if "classification_report" in results:
    report = results["classification_report"]
    print("\nClassification Report:")
    print(f"{'':>14s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    print("-" * 55)
    class_names = ["Normal", "AML Blast"]
    for cls in class_names:
        if cls in report:
            r = report[cls]
            print(f"{cls:>14s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")
    print("-" * 55)
    for avg in ["macro avg", "weighted avg"]:
        if avg in report:
            r = report[avg]
            print(f"{avg:>14s} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")

# ── Confusion Matrix ──
cm = np.array(results["confusion_matrix"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "AML Blast"],
    yticklabels=["Normal", "AML Blast"],
    annot_kws={"size": 18},
    ax=ax
)
ax.set_xlabel("Predicted", fontsize=14)
ax.set_ylabel("Actual", fontsize=14)
ax.set_title("Confusion Matrix — Test Set", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_notebook.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── Breakdown ──
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"  True  Negatives (Normal → Normal):    {tn}")
print(f"  False Positives (Normal → AML):        {fp}")
print(f"  False Negatives (AML → Normal):        {fn}")
print(f"  True  Positives (AML → AML):           {tp}")
print(f"\n  Specificity:  {tn / (tn + fp):.4f}")
print(f"  Sensitivity:  {tp / (tp + fn):.4f}")

# ── Also load the full pipeline summary ──
summary_path = os.path.join(RESULTS_DIR, "results_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    print(f"\n{'═' * 55}")
    print(f"  Pipeline Summary")
    print(f"{'═' * 55}")
    print(f"  Duration:   {summary.get('duration_minutes', '?')} min")
    ds = summary.get("dataset", {})
    print(f"  Images:     {ds.get('total_images', '?')}")
    print(f"  Patients:   {ds.get('total_patients', '?')}")
    m = summary.get("model", {})
    print(f"  Backbone:   {m.get('backbone', '?')}")
    print(f"  Params:     {m.get('total_params', '?'):,} total, {m.get('trainable_params', '?'):,} trainable")
    print(f"{'═' * 55}")
```

---

## Cell 11: View Grad-CAM Visualizations

```python
# Cell 11: Display Grad-CAM explainability visualizations
# main.py generates Grad-CAM++ images in outputs/gradcam_results/

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

BASE = "/kaggle/working/HemaVision"
GRADCAM_DIR = os.path.join(BASE, "outputs", "gradcam_results")

gradcam_images = sorted(glob.glob(os.path.join(GRADCAM_DIR, "*.png")))

if not gradcam_images:
    print(f"No Grad-CAM images found in {GRADCAM_DIR}")
    print("This can happen if Grad-CAM generation failed or was skipped.")
else:
    n = min(len(gradcam_images), 12)  # Show up to 12 images
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle("Grad-CAM++ Explainability Visualizations", fontsize=18, fontweight="bold")

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n:
            img = Image.open(gradcam_images[i])
            ax.imshow(img)
            ax.set_title(os.path.basename(gradcam_images[i]), fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gradcam_gallery.png"), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n✓ {len(gradcam_images)} Grad-CAM visualizations found in:")
    print(f"  {GRADCAM_DIR}")
```

---

## Cell 12: Verify & Download All Outputs

```python
# Cell 12: List all output files and copy to /kaggle/working/ for download
# main.py handles both .pt and .onnx export, so we just verify and copy.

import os
import shutil

BASE = "/kaggle/working/HemaVision"
OUTPUT_DIR = os.path.join(BASE, "outputs")

print("=" * 60)
print("  HemaVision Output Files")
print("=" * 60)

total_size = 0
file_list = []

for root, dirs, files in os.walk(OUTPUT_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        total_size += size_mb
        rel = os.path.relpath(fpath, BASE)
        file_list.append((rel, size_mb))
        icon = "🟢" if size_mb > 0.001 else "⚪"
        print(f"  {icon} {rel:55s} {size_mb:>8.2f} MB")

print(f"\n  Total: {len(file_list)} files, {total_size:.1f} MB")

# ── Verify critical files ──
print(f"\n{'─' * 60}")
print("  Critical File Check")
print(f"{'─' * 60}")

critical = {
    "Best model (.pt)":    "outputs/checkpoints/best_model.pt",
    "Final model (.pt)":   "outputs/checkpoints/final_model.pt",
    "ONNX model (.onnx)":  "outputs/checkpoints/final_model.onnx",
    "Training history":    "outputs/results/training_history.json",
    "Test results":        "outputs/results/test_results.json",
    "Pipeline summary":    "outputs/results/results_summary.json",
    "Training curves":     "outputs/results/training_history.png",
}

all_ok = True
for label, rel_path in critical.items():
    full = os.path.join(BASE, rel_path)
    exists = os.path.exists(full)
    size = os.path.getsize(full) / 1e6 if exists else 0
    status = f"✅ ({size:.1f} MB)" if exists else "❌ MISSING"
    print(f"  {label:25s} {status}")
    if not exists:
        all_ok = False

# ── Copy key artifacts to /kaggle/working/ for one-click download ──
print(f"\n{'─' * 60}")
print("  Copying artifacts to /kaggle/working/ for download...")
print(f"{'─' * 60}")

download_files = [
    "outputs/checkpoints/best_model.pt",
    "outputs/checkpoints/final_model.pt",
    "outputs/checkpoints/final_model.onnx",
    "outputs/results/results_summary.json",
    "outputs/results/training_history.json",
    "outputs/results/test_results.json",
    "outputs/results/training_history.png",
]

for rel_path in download_files:
    src = os.path.join(BASE, rel_path)
    dst = os.path.join("/kaggle/working", os.path.basename(rel_path))
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✓ {os.path.basename(rel_path)}")
    else:
        print(f"  ✗ {os.path.basename(rel_path)} (not found)")

# Copy Grad-CAM directory
gradcam_src = os.path.join(BASE, "outputs", "gradcam_results")
gradcam_dst = "/kaggle/working/gradcam_results"
if os.path.exists(gradcam_src):
    if os.path.exists(gradcam_dst):
        shutil.rmtree(gradcam_dst)
    shutil.copytree(gradcam_src, gradcam_dst)
    n_gcam = len(os.listdir(gradcam_dst))
    print(f"  ✓ gradcam_results/ ({n_gcam} files)")

print(f"\n{'═' * 60}")
print("  Done! Download artifacts from the 'Output' tab.")
print(f"{'═' * 60}")
```

---

## Key Notes

| Item | Detail |
|---|---|
| **GPU** | Enable GPU T4 x2 or P100 in Kaggle Settings → Accelerator |
| **Internet** | Must be ON (Settings → Internet → On) for git clone and pip install |
| **Dataset** | **binilj04/aml-cytomorphology** — Add via Kaggle "Add Data" sidebar, search "AML-Cytomorphology-WBC". ~6.2 GB, 18K+ images, 15 classes incl. MYO (blast). ⚠️ Do NOT use `walkersneps/aml-cytomorphology-lmu` (missing blast class!) |
| **Cell 8** | Runs `main.py --data-root <path>` — full pipeline (data loading, patient-level split, dual-stream model, training, evaluation, Grad-CAM, ONNX export) |
| **Cells 9-12** | Post-training cells that load results from disk (JSON files saved by main.py) — no in-memory dependencies on Cell 8 |
| **Runtime** | ~50 min for 50 epochs on T4 GPU (18K+ images) |
| **Outputs** | best_model.pt, final_model.pt, final_model.onnx, training_history.json, test_results.json, results_summary.json, gradcam_results/ — all in outputs/ |
