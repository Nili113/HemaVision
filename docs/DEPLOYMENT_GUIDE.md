# Deployment Guide

## Local Development

### One Command (Recommended)

```bash
./scripts/bootstrap.sh
```

This installs dependencies and starts both backend and frontend.
Python dependencies are installed inside `.venv` automatically.

Useful script-only workflow:

```bash
./scripts/install_all.sh
./scripts/start_all.sh
./scripts/status_all.sh
./scripts/stop_all.sh
```

`start_all.sh` automatically loads:
- root `.env` (backend variables like `DATABASE_URL`)
- `frontend/.env` (frontend variables)

If model warm-up is slow, increase timeouts:

```bash
BACKEND_STARTUP_TIMEOUT=300 FRONTEND_STARTUP_TIMEOUT=120 ./scripts/start_all.sh
```

After startup:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Swagger: http://localhost:8000/docs

Optional helpers:

```bash
./scripts/stop_all.sh
./scripts/status_all.sh
```

### Backend (FastAPI)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the API server
python -m interfaces.fastapi_app
# → http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Gradio Demo

```bash
python -m interfaces.gradio_app
# → http://localhost:7860
```

---

## Production Deployment

## Important: Is Vercel Alone Enough?

No. Use Vercel for frontend only.

- Vercel: React/Vite static frontend
- Railway (or similar): FastAPI + PyTorch inference backend
- Neon: PostgreSQL database

### Option 1: Docker

```bash
# Build
docker build -t hemavision .

# Run
docker run -p 8000:8000 hemavision

# With GPU support
docker run --gpus all -p 8000:8000 hemavision
```

### Option 2: Railway (Backend)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`

The `railway.json` file configures the build and health check automatically.

**Environment variables to set in Railway:**
- `DATABASE_URL=postgresql://...` (Neon connection string)
- `MODEL_PATH=/app/outputs/checkpoints/best_model.pt`

### Option 3: Vercel (Frontend)

```bash
cd frontend
npm run build
npx vercel deploy --prod
```

**Environment variables to set in Vercel:**
- `VITE_API_URL=https://your-railway-backend.up.railway.app`
- `VITE_GOOGLE_CLIENT_ID=<your-google-client-id>`

The `vercel.json` file handles SPA routing and asset caching.

---

## Training Pipeline

### On Local Machine

```bash
python main.py \
    --data-root /path/to/AML-Cytomorphology_LMU \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001
```

### On Google Colab

1. Upload dataset to Google Drive
2. Mount drive in Colab
3. Clone repo and install requirements
4. Run main.py with paths pointing to mounted drive

### Expected Output

```
outputs/
├── checkpoints/
│   └── best_model.pth          # Trained model weights
├── figures/
│   ├── training_history.png    # Loss and metric curves
│   └── confusion_matrix.png    # Evaluation matrix
└── gradcam/
    └── *.png                   # Grad-CAM visualizations
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | _unset_ | Neon/PostgreSQL connection string (fallback to SQLite when unset) |
| `MODEL_PATH` | `outputs/checkpoints/best_model.pt` | Path to trained model checkpoint |
| `VITE_API_URL` | `/api` in local dev | Backend API URL used by frontend |
| `VITE_GOOGLE_CLIENT_ID` | _unset_ | Google OAuth client ID for frontend login |

## Recommended End-to-End Deployment Order

1. Deploy backend to Railway.
2. Add `DATABASE_URL` and `MODEL_PATH` in Railway.
3. Verify backend health endpoint: `GET /health`.
4. Deploy frontend to Vercel.
5. Set `VITE_API_URL` in Vercel to Railway backend URL.
6. Set `VITE_GOOGLE_CLIENT_ID` in Vercel.
7. Validate app flow: auth, analysis, history image + Grad-CAM persistence.

---

## Health Monitoring

- **Health endpoint**: `GET /health` returns model load status
- **Docker healthcheck**: Pings `/health` every 30s
- **Railway**: Auto-monitors via `healthcheckPath` in railway.json
