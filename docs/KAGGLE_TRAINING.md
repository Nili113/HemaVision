# Kaggle Training Notebook for HemaVision

Complete Kaggle notebook code. Copy-paste each cell into a Kaggle notebook. Enable **GPU T4 x2** or **P100** accelerator and turn **Internet ON**.

---

## Cell 1: Clone the Repo & Install Dependencies

```python
# Cell 1: Clone repo and install dependencies
!git clone https://github.com/Nili113/HemaVision.git
%cd HemaVision

# Install backend/training dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q timm albumentations opencv-python-headless pillow matplotlib scikit-learn pandas tqdm seaborn

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

## Cell 9: Custom Training Code (Fallback)

```python
# Cell 9: Fallback training loop
# ONLY USE THIS if Cell 8 (main.py) fails for some reason.
# This is a standalone PyTorch training loop that works directly
# with the AML-Cytomorphology_LMU dataset's folder structure.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======================== CONFIGURATION ========================
# Dataset path: the images/ folder inside AML-Cytomorphology_LMU
# contains 21 subdirectories (BLA, LYT, NGS, MON, etc.)
DATASET_ROOT = "/kaggle/working/HemaVision/AML-Cytomorphology_LMU"
if not os.path.exists(DATASET_ROOT):
    # Try new and old Kaggle path formats
    for fallback in [
        "/kaggle/input/datasets/binilj04/aml-cytomorphology",
        "/kaggle/input/datasets/umarsani1605/aml-cytomorphology",
        "/kaggle/input/aml-cytomorphology",
        "/kaggle/input/aml-cytomorphology-lmu",
    ]:
        if os.path.exists(fallback):
            DATASET_ROOT = fallback
            break

CONFIG = {
    # Dataset — points to the images/ subfolder with 21 class dirs
    "data_dir": os.path.join(DATASET_ROOT, "images"),
    "train_dir": None,  # Will be auto-detected
    "val_dir": None,    # Will be auto-detected

    # Model
    "model_name": "efficientnet_b3",  # Options: resnet50, efficientnet_b3, vit_base_patch16_224
    "num_classes": None,  # Auto-detected from folder structure
    "pretrained": True,

    # Training
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "img_size": 224,

    # Output
    "save_dir": "/kaggle/working/HemaVision/trained_models",
}

# ======================== AUTO-DETECT DATASET STRUCTURE ========================
data_dir = CONFIG["data_dir"]

# Try to find train/val split
if os.path.exists(data_dir):
    subdirs = os.listdir(data_dir)

    # Pattern 1: data/train/ and data/val/ (or test/)
    if "train" in subdirs:
        CONFIG["train_dir"] = os.path.join(data_dir, "train")
        if "val" in subdirs:
            CONFIG["val_dir"] = os.path.join(data_dir, "val")
        elif "test" in subdirs:
            CONFIG["val_dir"] = os.path.join(data_dir, "test")
        elif "valid" in subdirs:
            CONFIG["val_dir"] = os.path.join(data_dir, "valid")

    # Pattern 2: data/Training/ and data/Testing/
    elif "Training" in subdirs:
        CONFIG["train_dir"] = os.path.join(data_dir, "Training")
        if "Testing" in subdirs:
            CONFIG["val_dir"] = os.path.join(data_dir, "Testing")

    # Pattern 3: Flat structure (data/class1/, data/class2/, etc.)
    else:
        # Check if subdirs contain images directly (flat class structure)
        has_class_dirs = all(
            os.path.isdir(os.path.join(data_dir, d))
            for d in subdirs
            if not d.startswith('.')
        )
        if has_class_dirs and len(subdirs) > 1:
            print("Flat dataset structure detected. Will create train/val split.")
            CONFIG["train_dir"] = data_dir  # Will handle split in DataLoader
            CONFIG["val_dir"] = None

print(f"Train dir: {CONFIG['train_dir']}")
print(f"Val dir: {CONFIG['val_dir']}")

# ======================== TRANSFORMS ========================
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ======================== LOAD DATA ========================
if CONFIG["train_dir"] and os.path.exists(CONFIG["train_dir"]):
    if CONFIG["val_dir"] and os.path.exists(CONFIG["val_dir"]):
        # Separate train and val directories
        train_dataset = datasets.ImageFolder(CONFIG["train_dir"], transform=train_transform)
        val_dataset = datasets.ImageFolder(CONFIG["val_dir"], transform=val_transform)
    else:
        # Single directory - do random split
        from torch.utils.data import random_split
        full_dataset = datasets.ImageFolder(CONFIG["train_dir"], transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    CONFIG["num_classes"] = len(train_dataset.dataset.classes if hasattr(train_dataset, 'dataset') else train_dataset.classes)
    class_names = train_dataset.dataset.classes if hasattr(train_dataset, 'dataset') else train_dataset.classes

    print(f"\nDataset Summary:")
    print(f"   Classes ({CONFIG['num_classes']}): {class_names}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
else:
    raise FileNotFoundError(
        f"Dataset not found at {CONFIG['data_dir']}!\n"
        "Please:\n"
        "  1. Add a dataset via Kaggle's 'Add Data' button\n"
        "  2. Update CONFIG['data_dir'] to point to /kaggle/input/<your-dataset>/\n"
    )

# ======================== MODEL ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

model = timm.create_model(CONFIG["model_name"], pretrained=CONFIG["pretrained"], num_classes=CONFIG["num_classes"])
model = model.to(device)

print(f"Model: {CONFIG['model_name']} ({sum(p.numel() for p in model.parameters()):,} params)")

# ======================== TRAINING SETUP ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

# ======================== TRAINING LOOP ========================
os.makedirs(CONFIG["save_dir"], exist_ok=True)

best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print(f"\nStarting training for {CONFIG['epochs']} epochs...\n")

for epoch in range(CONFIG["epochs"]):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # --- Validate ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100. * correct / total

    scheduler.step()

    # Log
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": class_names,
            "config": CONFIG,
        }, save_path)
        print(f"  Saved best model (val_acc: {val_acc:.2f}%)")

    print()

print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
```

---

## Cell 10: Plot Training Curves & Evaluate

```python
# Cell 10: Visualize results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Training & Validation Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/HemaVision/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nFinal Results:")
print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
print(f"   Model saved to: {CONFIG['save_dir']}/best_model.pth")
```

---

## Cell 11: Confusion Matrix & Classification Report

```python
# Cell 11: Detailed evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("/kaggle/working/HemaVision/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Cell 12: Export Model for Deployment

```python
# Cell 12: Export for deployment (ONNX + TorchScript)

# Save as TorchScript
model.eval()
dummy_input = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"]).to(device)

traced_model = torch.jit.trace(model, dummy_input)
traced_model.save(os.path.join(CONFIG["save_dir"], "model_scripted.pt"))
print("Saved TorchScript model: model_scripted.pt")

# Save as ONNX
try:
    torch.onnx.export(
        model, dummy_input,
        os.path.join(CONFIG["save_dir"], "model.onnx"),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Saved ONNX model: model.onnx")
except Exception as e:
    print(f"ONNX export failed: {e}")

# List all saved files
print(f"\nSaved models in {CONFIG['save_dir']}:")
for f in os.listdir(CONFIG["save_dir"]):
    size = os.path.getsize(os.path.join(CONFIG["save_dir"], f)) / 1e6
    print(f"   {f} ({size:.1f} MB)")

# Copy to /kaggle/working for easy download
!cp -r {CONFIG["save_dir"]}/* /kaggle/working/
print("\nModels copied to /kaggle/working/ for download")
```

---

## Key Notes

| Item | Detail |
|---|---|
| **GPU** | Enable GPU T4 x2 or P100 in Kaggle Settings -> Accelerator |
| **Internet** | Must be ON (Settings -> Internet -> On) for git clone and pip install |
| **Dataset** | **binilj04/aml-cytomorphology** — Add via Kaggle "Add Data" sidebar, search "AML-Cytomorphology-WBC". ~6.2 GB, 18K+ images, 21 classes incl. BLA (blast). ⚠️ Do NOT use `walkersneps/aml-cytomorphology-lmu` (missing BLA class!) |
| **Cell 8** | Runs `main.py --data-root <dataset_path>` — the project's full pipeline (data loading, patient-level split, dual-stream model, training, Grad-CAM) |
| **Cell 9** | Fallback standalone training loop. Only use if Cell 8 fails |
| **Runtime** | ~2-3 hours for 50 epochs on T4 GPU (18K+ images) |
| **Output** | best_model.pt, training_history.png, gradcam_results/, results_summary.json — all in outputs/ |
