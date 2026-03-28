#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

echo "[1/5] Creating virtual environment (.venv) if needed..."
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[2/5] Installing Python dependencies into .venv..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "[3/5] Installing frontend dependencies..."
cd "$ROOT_DIR/frontend"
npm install

cd "$ROOT_DIR"
echo "[4/5] Preparing runtime directories..."
mkdir -p outputs/checkpoints data .runlogs

echo "[5/5] Syncing model artifacts into outputs/checkpoints..."
if [[ -f "$ROOT_DIR/best_model.pt" && ! -f "$ROOT_DIR/outputs/checkpoints/best_model.pt" ]]; then
  cp "$ROOT_DIR/best_model.pt" "$ROOT_DIR/outputs/checkpoints/best_model.pt"
fi
if [[ -f "$ROOT_DIR/final_model.pt" && ! -f "$ROOT_DIR/outputs/checkpoints/final_model.pt" ]]; then
  cp "$ROOT_DIR/final_model.pt" "$ROOT_DIR/outputs/checkpoints/final_model.pt"
fi

echo "Install complete."
echo "Next: run ./scripts/start_all.sh or ./scripts/bootstrap.sh"
