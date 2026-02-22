<div align="center">

# HemaVision

**Multimodal Diagnostic Assistant for Acute Myeloid Leukemia Detection**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![License](https://img.shields.io/badge/License-Research_Only-yellow?style=flat)](#license)

An AI system that combines microscopic cell imagery with patient clinical data to detect Acute Myeloid Leukemia. Provides explainable predictions with Grad-CAM visualizations for clinical transparency.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Three-Tier Interface Strategy](#three-tier-interface-strategy)
- [API Reference](#api-reference)
- [Database](#database)
- [Design Decisions](#design-decisions)
- [Target Metrics](#target-metrics)
- [Dataset](#dataset)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

HemaVision is a multimodal deep learning platform for detecting AML blasts in peripheral blood smear microscopy images. It fuses visual features from cell images with tabular patient metadata (age, sex, genetic markers) through a dual-stream late fusion network, producing a binary classification with explainable Grad-CAM heatmaps.

The system ships with three interface tiers: a Gradio demo for quick evaluation, a production React dashboard for clinical workflows, and a FastAPI REST backend for programmatic integration. Every prediction is persisted to a local SQLite database for history tracking and aggregate analytics.

---

## Architecture

```
Visual Stream (ResNet50)          Tabular Stream (MLP)
  Cell Image (224x224)             Age, Sex, Genetics
         |                                |
   Conv layers -> AdaptiveAvgPool   Linear(5,32) -> ReLU -> Dropout
         |                                |
    2048-dim features               32-dim features
         |                                |
         +------------ Fusion ------------+
                         |
                   2080-dim concat
                         |
                  Linear(2080, 256) -> ReLU -> Dropout(0.3)
                  Linear(256, 1) -> Sigmoid
                         |
                    P(AML) in [0, 1]
```

**Dual-Stream Late Fusion Network** — the visual stream uses a pretrained ResNet50 backbone (frozen during initial training) to extract spatial features from cell images. The tabular stream encodes patient demographics and genetic markers through a lightweight MLP. Both streams are concatenated and passed through a shared classifier head.

**Grad-CAM** is applied to the final convolutional block (`layer4`) to produce spatial attention heatmaps showing which regions of the cell image most influenced the prediction.

---

## Project Structure

```
HemaVision/
|
|-- core/                              ML pipeline
|   |-- data_loader.py                 Patient-level data preprocessing and splitting
|   |-- model.py                       DualStreamFusionModel (ResNet50 + MLP + Fusion)
|   |-- dataset.py                     PyTorch Dataset, augmentation pipelines, dataloaders
|   |-- train.py                       Training loop, metrics tracking, early stopping
|   +-- gradcam.py                     Grad-CAM generation and overlay visualization
|
|-- interfaces/                        User-facing interfaces
|   |-- gradio_app.py                  Tier 1: Gradio demo with custom styling
|   +-- fastapi_app.py                Tier 3: REST API + WebSocket + DB integration
|
|-- frontend/                          Tier 2: React production dashboard
|   |-- src/
|   |   |-- components/                Reusable UI components
|   |   |   |-- Layout.tsx             Navigation, footer, responsive mobile menu
|   |   |   |-- UploadZone.tsx         Drag-and-drop image upload with validation
|   |   |   |-- PatientForm.tsx        Patient demographics and genetic markers form
|   |   |   |-- ResultCard.tsx         Animated prediction result display
|   |   |   +-- GradCAMViewer.tsx      Interactive heatmap viewer with tabs
|   |   |-- pages/
|   |   |   |-- Home.tsx               Landing page with feature overview
|   |   |   |-- Analyze.tsx            3-step analysis workflow
|   |   |   |-- History.tsx            Analysis history with stats and record browser
|   |   |   +-- About.tsx              Architecture docs, tech stack, dataset info
|   |   |-- hooks/
|   |   |   +-- useAnalysis.ts         Custom hook for analysis state management
|   |   |-- lib/
|   |   |   +-- api.ts                 Typed API client (axios)
|   |   |-- App.tsx                    Route definitions
|   |   +-- main.tsx                   React entry point with QueryClient + Router
|   |-- tailwind.config.js             Design system tokens
|   |-- vite.config.ts                 Build config with API proxy
|   +-- package.json                   Frontend dependencies
|
|-- utils/
|   |-- config.py                      Centralized configuration (paths, model, training)
|   +-- database.py                    SQLite database layer for analysis records
|
|-- main.py                            Pipeline orchestrator (7-phase train/eval pipeline)
|-- requirements.txt                   Python dependencies
|-- Dockerfile                         Container build
|-- railway.json                       Railway deployment config
+-- CLAUDE.md                          Agent build instructions
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | ML pipeline and API server |
| Node.js | 18+ | React frontend build tooling |
| npm | 9+ | Package manager for frontend |
| GPU (optional) | CUDA 11.8+ | Accelerated training and inference |

---

## Installation

Clone the repository, then follow the steps below.

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, FastAPI, Gradio, scikit-learn, and all other backend dependencies.

### 2. Frontend Dependencies

```bash
cd frontend
npm install
```

This installs React, Tailwind CSS, Framer Motion, Axios, React Router, and all other frontend packages.

### 3. Environment Configuration

```bash
cp frontend/.env.example frontend/.env
```

Edit `frontend/.env` if your backend runs on a different host or port:

```
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID={{please-get-this-from-google-cloud}}
```

---

## Running the Application

The system has three independent components. Run the backend first, then the frontend.

### Backend (FastAPI)

```bash
python -m interfaces.fastapi_app
```

This starts the REST API server on `http://localhost:8000`. On startup it:
- Loads the trained model checkpoint (or starts in demo mode if none found)
- Initializes the SQLite database at `data/hemavision.db`
- Enables CORS for frontend connections

API documentation is auto-generated at `http://localhost:8000/docs` (Swagger UI).

### Frontend (React)

```bash
cd frontend
npm run dev
```

This starts the Vite dev server on `http://localhost:5173` with hot reload. The frontend proxies API requests to `localhost:8000` during development.

### Gradio Demo (Alternative)

```bash
python -m interfaces.gradio_app
```

A self-contained demo UI on `http://localhost:7860`. Useful for quick testing and presentations.

### Training the Model

```bash
python main.py --data-root /path/to/AML-Cytomorphology_LMU
```

Training runs a 7-phase pipeline:

| Phase | Description |
|-------|-------------|
| 1 | Load and preprocess data with patient-level splitting |
| 2 | Create PyTorch dataloaders with augmentation |
| 3 | Initialize dual-stream model with frozen ResNet50 backbone |
| 4 | Train with weighted BCE loss and early stopping |
| 5 | Evaluate on held-out test set |
| 6 | Generate Grad-CAM visualizations |
| 7 | Export model and print summary |

Training parameters can be customized:

```bash
python main.py \
    --data-root /path/to/data \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001
```

Output artifacts are saved to `outputs/`:

```
outputs/
|-- checkpoints/best_model.pt       Trained model weights
|-- figures/training_history.png     Loss and metric curves
+-- gradcam/*.png                    Grad-CAM visualizations
```

---

## Three-Tier Interface Strategy

| Tier | Interface | Stack | Use Case |
|------|-----------|-------|----------|
| 1 | Gradio Demo | Gradio + Python | Quick demos, presentations, thesis defense |
| 2 | React Dashboard | React + TypeScript + Tailwind + Framer Motion | Production clinical workflow, YC pitch |
| 3 | REST API | FastAPI + Pydantic + WebSocket + SQLite | Programmatic integration, batch processing |

All three tiers share the same model inference code and produce identical predictions.

---

## API Reference

Base URL: `http://localhost:8000`

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API metadata and endpoint list |
| `GET` | `/health` | Health check — returns model load status and device info |
| `GET` | `/docs` | Swagger UI (interactive API documentation) |
| `GET` | `/redoc` | ReDoc (alternative API documentation) |

### Prediction

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single prediction from JSON body with base64-encoded image |
| `POST` | `/predict/upload` | Single prediction from multipart form upload |
| `POST` | `/batch_predict` | Batch prediction for multiple samples (Grad-CAM disabled for speed) |
| `WS` | `/ws/predict` | WebSocket endpoint for real-time streaming predictions |
| `GET` | `/model/info` | Model architecture, parameter count, device info |

### History and Records

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/analyses` | Paginated analysis history (query: `limit`, `offset`) |
| `GET` | `/analyses/stats` | Aggregate statistics (totals, averages, risk distribution) |
| `GET` | `/analyses/{id}` | Single analysis record by ID |
| `DELETE` | `/analyses/{id}` | Delete a single analysis record |

Every call to `/predict` or `/predict/upload` automatically saves the result to the database.

### Example: Predict via cURL

```bash
# Base64-encoded image
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64 string>",
    "age": 65,
    "sex": "Male",
    "npm1_mutated": true,
    "flt3_mutated": false,
    "genetic_other": false
  }'

# File upload
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@cell_image.png" \
  -F "age=65" \
  -F "sex=Male" \
  -F "npm1_mutated=true"
```

---

## Database

HemaVision uses **SQLite** for persistent storage of analysis records. The database file is created automatically at `data/hemavision.db` on first server startup.

### Schema

```sql
CREATE TABLE analyses (
    id              TEXT PRIMARY KEY,    -- UUID
    prediction      TEXT NOT NULL,       -- "AML Blast (Malignant)" or "Normal Cell (Benign)"
    probability     REAL NOT NULL,       -- Raw sigmoid output [0, 1]
    confidence      REAL NOT NULL,       -- Distance from decision boundary
    risk_level      TEXT NOT NULL,       -- HIGH RISK / MODERATE RISK / LOW RISK
    risk_color      TEXT NOT NULL,       -- Hex color for UI
    inference_time_ms REAL NOT NULL,     -- Model inference latency
    patient_age     INTEGER NOT NULL,
    patient_sex     TEXT NOT NULL,
    npm1_mutated    INTEGER NOT NULL,    -- 0 or 1
    flt3_mutated    INTEGER NOT NULL,    -- 0 or 1
    genetic_other   INTEGER NOT NULL,    -- 0 or 1
    image_filename  TEXT,                -- Original upload filename
    gradcam_base64  TEXT,                -- Grad-CAM overlay (base64 PNG)
    created_at      TEXT NOT NULL        -- ISO 8601 timestamp
);
```

Indexed on `created_at DESC` and `risk_level` for fast queries. Uses WAL journal mode for concurrent read performance.

---

## Design Decisions

1. **Patient-level splitting** — All images from a single patient stay in the same train/val/test split. This prevents data leakage where the model memorizes patient-specific features rather than learning generalizable cell morphology.

2. **Weighted BCE loss** — The dataset has class imbalance between normal cells and AML blasts. Weighted binary cross-entropy upweights the minority class to prevent the model from defaulting to the majority prediction.

3. **Frozen backbone transfer learning** — The ResNet50 backbone is initialized with ImageNet weights and frozen during initial training. Only the fusion layers and classifier head are trained. This prevents catastrophic forgetting and reduces the required training data.

4. **Late fusion** — Visual and tabular streams are encoded independently before concatenation. This preserves modality-specific feature representations and allows each stream to learn at its own pace, compared to early fusion which can cause one modality to dominate.

5. **Grad-CAM on layer4** — The final convolutional block provides the best trade-off between spatial resolution and semantic content. Earlier layers have higher resolution but less meaningful features; later fully-connected layers lose spatial information entirely.

6. **SQLite for persistence** — Zero-configuration, file-based, no external services needed. Suitable for single-server deployments and research contexts. Can be swapped for PostgreSQL in production by modifying `utils/database.py`.

---

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | >= 90% | Overall correct predictions |
| AUC-ROC | >= 0.95 | Area under the ROC curve |
| Precision | >= 90% | Positive predictive value |
| Recall | >= 90% | Sensitivity to AML blasts |
| F1-Score | >= 90% | Harmonic mean of precision and recall |
| Inference | < 50ms | Per-sample latency on GPU |

---

## Dataset

**Munich AML-Cytomorphology** (LMU Munich)

| Property | Value |
|----------|-------|
| Total images | 18,000+ single-cell microscopy images |
| Patients | 200+ |
| Cell type classes | 21 (grouped into binary: blast vs. non-blast) |
| Image source | Peripheral blood smears |
| Staining | May-Grunwald-Giemsa |
| Source | The Cancer Imaging Archive (TCIA) / Kaggle |

The data loader groups images by patient ID and performs a 70/10/20 train/validation/test split at the patient level.

---

## Deployment

### Docker

```bash
docker build -t hemavision .
docker run -p 8000:8000 hemavision
```

### Frontend to Vercel

```bash
cd frontend
npm run build
npx vercel deploy --prod
```

Set `VITE_API_URL` to your deployed backend URL in Vercel environment variables.

### Backend to Railway

```bash
railway login
railway init
railway up
```

The `railway.json` file configures the Dockerfile build, start command, and health check automatically.

---

## License

Research and educational use only. This system is not intended for clinical diagnosis. All predictions should be reviewed by a qualified hematologist.

---

<div align="center">

**Built by Firoj, Nilima, and Aashika**

Crafted with clarity, purpose, and the potential to impact human lives.

</div>
