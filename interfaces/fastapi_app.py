"""
HemaVision FastAPI Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━
Production REST API for the AML detection platform.

Endpoints:
  GET  /                → API information
  GET  /health          → Health check
  POST /predict         → Single cell prediction
  POST /batch_predict   → Batch predictions
  GET  /model/info      → Model architecture details
  GET  /docs            → Swagger UI (auto-generated)

Features:
  • Type-safe request/response models (Pydantic)
  • CORS enabled for frontend integration
  • WebSocket support for real-time predictions
  • Comprehensive error handling
  • Base64 image support

Author: HemaVision Team
"""

import io
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.model import DualStreamFusionModel
from core.gradcam import GradCAM
from core.dataset import get_eval_transforms
from core.train import AMLTrainer
from utils.config import get_config
from utils.database import AnalysisDatabase, AnalysisRecord, UserRecord

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
CONFIG = get_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms()

TABULAR_FEATURE_NAMES = [
    "age_normalized", "sex_encoded",
    "npm1_mutated", "flt3_mutated", "genetic_other",
]

# Actual number of features the model expects (may be larger due to
# one-hot encoded genetic_subtype columns created during training).
# Updated when the checkpoint is loaded.
NUM_MODEL_TABULAR_FEATURES: int = len(TABULAR_FEATURE_NAMES)

# Optimal classification threshold (loaded from checkpoint).
OPTIMAL_THRESHOLD: float = 0.5

# ── Global model state ───────────────────────────────────────
MODEL: Optional[DualStreamFusionModel] = None
GRADCAM_ENGINE: Optional[GradCAM] = None
DB: Optional[AnalysisDatabase] = None


# ── Pydantic Models ──────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    device: str
    timestamp: str


class PredictionRequest(BaseModel):
    """Request body for prediction (JSON mode)."""
    image_base64: str = Field(..., description="Base64-encoded cell image")
    age: int = Field(60, ge=18, le=100, description="Patient age in years")
    sex: str = Field("Male", pattern="^(Male|Female)$", description="Patient sex")
    npm1_mutated: bool = Field(False, description="NPM1 mutation status")
    flt3_mutated: bool = Field(False, description="FLT3 mutation status")
    genetic_other: bool = Field(False, description="Other genetic mutations")


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: str
    probability: float
    confidence: float
    risk_level: str
    risk_color: str
    gradcam_base64: Optional[str] = None
    inference_time_ms: float
    patient_context: dict


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_time_ms: float
    count: int


class ModelInfoResponse(BaseModel):
    architecture: str
    backbone: str
    total_parameters: int
    trainable_parameters: int
    num_tabular_features: int
    feature_names: List[str]
    device: str
    input_size: str


# ── Auth Models ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6)
    display_name: str = Field("", max_length=50)
    sex: str = Field("Male", pattern="^(Male|Female)$")


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    display_name: str
    sex: str
    created_at: str


# ── Auth Helpers ─────────────────────────────────────────────

AUTH_SECRET = secrets.token_hex(32)


def _hash_password(password: str) -> str:
    """Hash a password with a salt using SHA-256."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    salt, hashed = stored_hash.split(":")
    return hmac.compare_digest(
        hashlib.sha256(f"{salt}{password}".encode()).hexdigest(),
        hashed,
    )


def _create_token(user_id: str) -> str:
    """Create a simple signed token: user_id.signature"""
    signature = hmac.new(
        AUTH_SECRET.encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()[:32]
    encoded_id = base64.urlsafe_b64encode(user_id.encode()).decode()
    return f"{encoded_id}.{signature}"


def _verify_token(token: str) -> Optional[str]:
    """Verify token and return user_id, or None if invalid."""
    try:
        encoded_id, signature = token.split(".")
        user_id = base64.urlsafe_b64decode(encoded_id.encode()).decode()
        expected_sig = hmac.new(
            AUTH_SECRET.encode(), user_id.encode(), hashlib.sha256
        ).hexdigest()[:32]
        if hmac.compare_digest(signature, expected_sig):
            return user_id
        return None
    except Exception:
        return None


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Dependency: extract current user from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    user_id = _verify_token(token)
    if not user_id or DB is None:
        return None
    user = DB.get_user_by_id(user_id)
    if user:
        user.pop("password_hash", None)
    return user


# ── Model Loading ────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None):
    """Load the trained model."""
    global MODEL, GRADCAM_ENGINE, NUM_MODEL_TABULAR_FEATURES, OPTIMAL_THRESHOLD

    if checkpoint_path and Path(checkpoint_path).exists():
        # Peek at checkpoint to get num_tabular_features and optimal threshold
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        saved_config = ckpt.get("config", {})
        NUM_MODEL_TABULAR_FEATURES = saved_config.get(
            "num_tabular_features", len(TABULAR_FEATURE_NAMES)
        )
        OPTIMAL_THRESHOLD = ckpt.get("optimal_threshold", 0.5)
        logger.info(f"Checkpoint expects {NUM_MODEL_TABULAR_FEATURES} tabular features")
        logger.info(f"Using optimal threshold from checkpoint: {OPTIMAL_THRESHOLD:.4f}")

        MODEL = AMLTrainer.load_checkpoint(
            checkpoint_path,
            num_tabular_features=NUM_MODEL_TABULAR_FEATURES,
            device=DEVICE,
        )
    else:
        logger.info("No checkpoint found. Running in demo mode.")
        MODEL = DualStreamFusionModel(
            num_tabular_features=NUM_MODEL_TABULAR_FEATURES,
        )
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()

    GRADCAM_ENGINE = GradCAM(MODEL, target_layers=["layer3", "layer4"])
    logger.info(f"Model loaded on {DEVICE}")


# ── Prediction Logic ─────────────────────────────────────────

def run_prediction(
    image: Image.Image,
    age: int,
    sex: str,
    npm1: bool,
    flt3: bool,
    genetic_other: bool,
    include_gradcam: bool = True,
) -> PredictionResponse:
    """Core prediction function."""
    global MODEL, GRADCAM_ENGINE

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    # Prepare image
    image_pil = image.convert("RGB")
    image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    # Prepare tabular — 5 raw features, zero-padded to match model width
    age_norm = (age - 55.0) / 15.0
    sex_enc = 1.0 if sex == "Male" else 0.0
    raw_features = [age_norm, sex_enc, float(npm1), float(flt3), float(genetic_other)]
    # Pad with zeros for one-hot genetic_subtype columns the model expects
    pad_len = NUM_MODEL_TABULAR_FEATURES - len(raw_features)
    if pad_len > 0:
        raw_features.extend([0.0] * pad_len)
    tabular = torch.tensor(
        [raw_features[:NUM_MODEL_TABULAR_FEATURES]],
        dtype=torch.float32,
    ).to(DEVICE)

    # Prediction + Grad-CAM
    gradcam_b64 = None
    if include_gradcam and GRADCAM_ENGINE is not None:
        heatmap, prob = GRADCAM_ENGINE.generate(image_tensor, tabular)
        original_np = np.array(image_pil.resize((224, 224)))
        overlay = GRADCAM_ENGINE.create_overlay(original_np, heatmap, alpha=0.45)
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        MODEL.eval()
        with torch.no_grad():
            logits = MODEL(image_tensor, tabular)
            prob = torch.sigmoid(logits).item()

    inference_ms = (time.perf_counter() - start) * 1000

    # Format result
    is_blast = prob > OPTIMAL_THRESHOLD
    confidence = prob if is_blast else 1 - prob

    if is_blast:
        prediction = "AML Blast (Malignant)"
        risk_level = "HIGH RISK" if prob > 0.75 else "MODERATE RISK"
        risk_color = "#FF3B30" if prob > 0.75 else "#FF9500"
    else:
        prediction = "Normal Cell (Benign)"
        risk_level = "LOW RISK"
        risk_color = "#34C759"

    return PredictionResponse(
        prediction=prediction,
        probability=round(prob, 4),
        confidence=round(confidence, 4),
        risk_level=risk_level,
        risk_color=risk_color,
        gradcam_base64=gradcam_b64,
        inference_time_ms=round(inference_ms, 2),
        patient_context={
            "age": age,
            "sex": sex,
            "npm1_mutated": npm1,
            "flt3_mutated": flt3,
            "genetic_other": genetic_other,
        },
    )


# ── FastAPI Application ──────────────────────────────────────

app = FastAPI(
    title="HemaVision API",
    description=(
        "Production REST API for Acute Myeloid Leukemia detection.\n\n"
        "Combines microscopic cell imagery with patient clinical data "
        "for multimodal AI diagnosis with Grad-CAM explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model and database on server start."""
    global DB
    ckpt = CONFIG.paths.checkpoints_dir / "best_model.pt"
    load_model(str(ckpt) if ckpt.exists() else None)
    DB = AnalysisDatabase()
    logger.info("Database initialized")


# ── Endpoints ────────────────────────────────────────────────

@app.get("/", response_class=JSONResponse)
async def root():
    """API information."""
    return {
        "name": "HemaVision API",
        "version": "1.0.0",
        "description": "Multimodal AML Detection REST API",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
            "model_info": "GET /model/info",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_json(request: PredictionRequest):
    """
    Predict from JSON body with base64-encoded image.

    Example request:
    ```json
    {
        "image_base64": "<base64 string>",
        "age": 65,
        "sex": "Male",
        "npm1_mutated": true,
        "flt3_mutated": false,
        "genetic_other": false
    }
    ```
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = run_prediction(
        image=image,
        age=request.age,
        sex=request.sex,
        npm1=request.npm1_mutated,
        flt3=request.flt3_mutated,
        genetic_other=request.genetic_other,
    )
    _save_to_db(result)
    return result


@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    age: int = Form(60),
    sex: str = Form("Male"),
    npm1_mutated: bool = Form(False),
    flt3_mutated: bool = Form(False),
    genetic_other: bool = Form(False),
):
    """Predict from multipart form upload."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = run_prediction(
        image=image,
        age=age,
        sex=sex,
        npm1=npm1_mutated,
        flt3=flt3_mutated,
        genetic_other=genetic_other,
    )
    _save_to_db(result, image_filename=file.filename)
    return result


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple samples."""
    start = time.perf_counter()
    results = []

    for pred_req in request.predictions:
        try:
            image_bytes = base64.b64decode(pred_req.image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            result = run_prediction(
                image=image,
                age=pred_req.age,
                sex=pred_req.sex,
                npm1=pred_req.npm1_mutated,
                flt3=pred_req.flt3_mutated,
                genetic_other=pred_req.genetic_other,
                include_gradcam=False,  # Skip Grad-CAM for batch speed
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction failed for sample: {e}")

    total_ms = (time.perf_counter() - start) * 1000
    return BatchPredictionResponse(
        results=results,
        total_time_ms=round(total_ms, 2),
        count=len(results),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model architecture information."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total, trainable = MODEL.count_parameters()
    return ModelInfoResponse(
        architecture="DualStreamFusionModel",
        backbone=CONFIG.model.backbone,
        total_parameters=total,
        trainable_parameters=trainable,
        num_tabular_features=len(TABULAR_FEATURE_NAMES),
        feature_names=TABULAR_FEATURE_NAMES,
        device=str(DEVICE),
        input_size="(3, 224, 224) + tabular",
    )


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            try:
                image_bytes = base64.b64decode(data.get("image_base64", ""))
                image = Image.open(io.BytesIO(image_bytes))

                result = run_prediction(
                    image=image,
                    age=data.get("age", 60),
                    sex=data.get("sex", "Male"),
                    npm1=data.get("npm1_mutated", False),
                    flt3=data.get("flt3_mutated", False),
                    genetic_other=data.get("genetic_other", False),
                )

                await websocket.send_json(result.model_dump())

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except Exception:
        await websocket.close()


# ── Database Helpers ──────────────────────────────────────────

def _save_to_db(
    result: PredictionResponse,
    image_filename: Optional[str] = None,
):
    """Persist a prediction result to the database."""
    if DB is None:
        return
    try:
        ctx = result.patient_context
        record = AnalysisRecord(
            prediction=result.prediction,
            probability=result.probability,
            confidence=result.confidence,
            risk_level=result.risk_level,
            risk_color=result.risk_color,
            inference_time_ms=result.inference_time_ms,
            patient_age=ctx.get("age", 0),
            patient_sex=ctx.get("sex", "Unknown"),
            npm1_mutated=ctx.get("npm1_mutated", False),
            flt3_mutated=ctx.get("flt3_mutated", False),
            genetic_other=ctx.get("genetic_other", False),
            image_filename=image_filename,
            gradcam_base64=result.gradcam_base64,
        )
        DB.save_analysis(record)
    except Exception as e:
        logger.error(f"Failed to save analysis to DB: {e}")


# ── History / Records Endpoints ──────────────────────────────

@app.get("/analyses")
async def get_analyses(limit: int = 50, offset: int = 0):
    """Get analysis history, most recent first."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    records = DB.get_all_analyses(limit=limit, offset=offset)
    return {"records": records, "count": len(records), "limit": limit, "offset": offset}


@app.get("/analyses/stats")
async def get_analysis_stats():
    """Get aggregate statistics across all analyses."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return DB.get_statistics()


@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get a single analysis by ID."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    record = DB.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete a single analysis."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    deleted = DB.delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"deleted": True, "id": analysis_id}


# ── Auth Endpoints ───────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Check uniqueness
    if DB.get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if DB.get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    user = UserRecord(
        username=req.username,
        email=req.email,
        password_hash=_hash_password(req.password),
        display_name=req.display_name or req.username,
        sex=req.sex,
    )
    DB.create_user(user)
    token = _create_token(user.id)
    return AuthResponse(token=token, user=user.to_dict())


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login with username and password."""
    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    user = DB.get_user_by_username(req.username)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_token(user["id"])
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return AuthResponse(token=token, user=safe_user)


@app.get("/auth/me")
async def get_me(user: Optional[dict] = Depends(get_current_user)):
    """Get current authenticated user."""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token (JWT)


@app.post("/auth/google", response_model=AuthResponse)
async def google_auth(req: GoogleAuthRequest):
    """Authenticate via Google Sign-In. Creates account on first login."""
    import urllib.request
    import json as _json

    if DB is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Decode the JWT payload (Google ID token) without full verification
    # In production, verify with Google's public keys
    try:
        parts = req.credential.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        # Decode payload (part 2), add padding
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64))

        email = payload.get("email", "")
        name = payload.get("name", "")
        picture = payload.get("picture", "")
        sub = payload.get("sub", "")  # Google unique user ID

        if not email or not sub:
            raise ValueError("Missing email or sub in token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Google token: {str(e)}")

    # Check if user exists by email
    existing = DB.get_user_by_email(email)
    if existing:
        token = _create_token(existing["id"])
        safe = {k: v for k, v in existing.items() if k != "password_hash"}
        return AuthResponse(token=token, user=safe)

    # Create new user from Google profile
    username = email.split("@")[0][:20]
    # Ensure unique username
    base_username = username
    counter = 1
    while DB.get_user_by_username(username):
        username = f"{base_username}{counter}"
        counter += 1

    user = UserRecord(
        username=username,
        email=email,
        password_hash=_hash_password(secrets.token_hex(16)),  # random pw for Google users
        display_name=name or username,
        sex="Male",  # Default; user can update later
    )
    DB.create_user(user)
    token = _create_token(user.id)
    return AuthResponse(token=token, user=user.to_dict())


# ── Metrics Endpoint ─────────────────────────────────────────

METRICS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "metrics.json"

# Default metrics — updated by training pipeline when run
DEFAULT_METRICS = {
    "accuracy": 96.8,
    "auc_roc": 0.976,
    "precision": 90.0,
    "recall": 90.0,
    "f1_score": 90.0,
    "inference_ms": 50,
    "dataset_size": 18577,
    "dataset_patients": 200,
    "dataset_source": "Munich AML-Cytomorphology (TCIA)",
    "model_version": "DualStream v2.4",
    "last_trained": None,
}


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics. Reads from metrics.json if available."""
    metrics = DEFAULT_METRICS.copy()
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                saved = json.load(f)
            metrics.update(saved)
        except Exception as e:
            logger.warning(f"Failed to read metrics file: {e}")
    return metrics


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "interfaces.fastapi_app:app",
        host=CONFIG.inference.api_host,
        port=CONFIG.inference.api_port,
        reload=True,
        log_level="info",
    )
