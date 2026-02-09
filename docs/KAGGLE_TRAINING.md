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
# Option A: If the dataset is already on Kaggle, link it via "Add Data" sidebar.
# Option B: If dataset is from a direct URL or within the repo, handle here.

import os
import shutil

# ============================================================
# OPTION A: Using a Kaggle dataset (recommended)
# If you added a dataset via the Kaggle "Add Data" button,
# it will be at /kaggle/input/<dataset-name>/
# Uncomment and adjust the following:
# ============================================================

# KAGGLE_DATASET_PATH = "/kaggle/input/your-dataset-name"
# LOCAL_DATASET_PATH = "/kaggle/working/HemaVision/data"
# os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
# !cp -r {KAGGLE_DATASET_PATH}/* {LOCAL_DATASET_PATH}/

# ============================================================
# OPTION B: Download from a URL (e.g., Google Drive, Zenodo, etc.)
# ============================================================

# !gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O dataset.zip
# !unzip -q dataset.zip -d /kaggle/working/HemaVision/data

# ============================================================
# OPTION C: Check if data already exists in the repo
# ============================================================

data_candidates = [
    "/kaggle/working/HemaVision/data",
    "/kaggle/working/HemaVision/dataset",
    "/kaggle/working/HemaVision/backend/data",
    "/kaggle/working/HemaVision/ml",
    "/kaggle/working/HemaVision/ml/data",
    "/kaggle/working/HemaVision/model",
]

for path in data_candidates:
    if os.path.exists(path):
        print(f"Found: {path}")
        for item in os.listdir(path):
            full = os.path.join(path, item)
            if os.path.isdir(full):
                count = len(os.listdir(full))
                print(f"   {item}/ ({count} items)")
            else:
                size = os.path.getsize(full) / 1e6
                print(f"   {item} ({size:.1f} MB)")
    else:
        print(f"Not found: {path}")

print("\nIf no data found, use OPTION A or B above to load your dataset.")
```

---

## Cell 5: Setup the Acute Lymphoblastic Leukemia Dataset

```python
# Cell 5: Download the ALL (Acute Lymphoblastic Leukemia) dataset
# This is likely the hematology image classification dataset used by HemaVision

# Install kaggle API if needed
!pip install -q kaggle

# Common blood cell / leukemia datasets on Kaggle:
# - "mehradaria/leukemia"
# - "andrewmvd/leukemia-classification"
# - "paultimothymooney/blood-cells"
# - "unclesamulus/blood-cell-images"

# If using Kaggle's "Add Data" button - just search for:
# "blood cell", "leukemia", "ALL", "hematology"
# and add it to your notebook

DATASET_DIR = "/kaggle/working/HemaVision/data"
os.makedirs(DATASET_DIR, exist_ok=True)

print(f"Dataset directory: {DATASET_DIR}")
print(f"Contents: {os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else 'Empty'}")
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

## Cell 8: Configure and Run Training

```python
# Cell 8: Run training
# IMPORTANT: Adjust these paths based on what you found in
# Cells 3, 6, and 7. Below are common patterns.

import os
os.chdir("/kaggle/working/HemaVision")

# OPTION A: If there's a dedicated training script
training_scripts = [
    "train.py",
    "ml/train.py",
    "backend/train.py",
    "backend/ml/train.py",
    "model/train.py",
    "scripts/train.py",
    "src/train.py",
]

script_found = None
for script in training_scripts:
    if os.path.exists(script):
        script_found = script
        break

if script_found:
    print(f"Found training script: {script_found}")
    print(f"Running: python {script_found}")
    !python {script_found}
else:
    print("No standard training script found.")
    print("Looking for alternative entry points...")

    # Check for Jupyter notebooks
    import glob
    notebooks = glob.glob("**/*train*.ipynb", recursive=True)
    if notebooks:
        print(f"Found notebook(s): {notebooks}")
        print("Convert and run with: !jupyter nbconvert --to script <notebook> && python <script>")

    # Check for main.py
    if os.path.exists("main.py"):
        print("Found main.py - inspecting...")
        !head -50 main.py

    print("\nYou may need to write custom training code (see Cell 9)")
```

---

## Cell 9: Custom Training Code (Fallback)

```python
# Cell 9: Custom training code if no training script exists in the repo
# This is a generic PyTorch training loop for blood cell classification
# Adjust model architecture, dataset paths, and hyperparameters as needed

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
CONFIG = {
    # Dataset
    "data_dir": "/kaggle/working/HemaVision/data",      # <-- ADJUST THIS
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
| **Dataset** | Add via Kaggle "Add Data" sidebar -> search for blood cell / leukemia datasets -> update CONFIG["data_dir"] to /kaggle/input/<dataset-name> |
| **Cell 6-7** | These cells auto-discover the training script in the repo. If one exists, Cell 8 runs it directly. If not, Cell 9's custom training loop kicks in. |
| **Runtime** | ~1-2 hours for 25 epochs on T4 GPU depending on dataset size |
| **Output** | best_model.pth, model_scripted.pt, model.onnx -- all downloadable from the Output tab |
