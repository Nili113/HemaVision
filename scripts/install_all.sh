#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/4] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[2/4] Installing frontend dependencies..."
cd "$ROOT_DIR/frontend"
npm install

cd "$ROOT_DIR"
echo "[3/4] Preparing runtime directories..."
mkdir -p outputs/checkpoints data .runlogs

echo "[4/4] Syncing model artifacts into outputs/checkpoints..."
if [[ -f "$ROOT_DIR/best_model.pt" && ! -f "$ROOT_DIR/outputs/checkpoints/best_model.pt" ]]; then
  cp "$ROOT_DIR/best_model.pt" "$ROOT_DIR/outputs/checkpoints/best_model.pt"
fi
if [[ -f "$ROOT_DIR/final_model.pt" && ! -f "$ROOT_DIR/outputs/checkpoints/final_model.pt" ]]; then
  cp "$ROOT_DIR/final_model.pt" "$ROOT_DIR/outputs/checkpoints/final_model.pt"
fi

echo "Install complete."
echo "Next: run ./scripts/start_all.sh or ./scripts/bootstrap.sh"
