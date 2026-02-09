# Deployment Guide

## Local Development

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
- `ENVIRONMENT=production`
- `MODEL_PATH=/app/outputs/checkpoints/best_model.pth`

### Option 3: Vercel (Frontend)

```bash
cd frontend
npm run build
npx vercel deploy --prod
```

**Environment variables to set in Vercel:**
- `VITE_API_URL=https://your-railway-backend.up.railway.app`

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
| `ENVIRONMENT` | `development` | Runtime environment |
| `MODEL_PATH` | `outputs/checkpoints/best_model.pth` | Path to trained model |
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL for frontend |

---

## Health Monitoring

- **Health endpoint**: `GET /health` returns model load status
- **Docker healthcheck**: Pings `/health` every 30s
- **Railway**: Auto-monitors via `healthcheckPath` in railway.json
