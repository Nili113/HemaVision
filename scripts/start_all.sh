#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.runlogs"
VENV_DIR="$ROOT_DIR/.venv"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing .venv. Run ./scripts/install_all.sh first."
  exit 1
fi

pid_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1 || true
  else
    echo ""
  fi
}

cleanup_stale_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
    fi
  fi
}

cleanup_stale_pid_file "$BACKEND_PID_FILE"
cleanup_stale_pid_file "$FRONTEND_PID_FILE"

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  echo "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
else
  existing_backend_pid="$(pid_on_port "$BACKEND_PORT")"
  if [[ -n "$existing_backend_pid" ]]; then
    if curl -sf "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
      echo "$existing_backend_pid" > "$BACKEND_PID_FILE"
      echo "Backend already listening on port $BACKEND_PORT (PID $existing_backend_pid)"
    else
      echo "Port $BACKEND_PORT is in use by PID $existing_backend_pid, but health check failed."
      echo "Stop that process or run: BACKEND_PORT=8001 ./scripts/start_all.sh"
      exit 1
    fi
  else
  echo "Starting backend on port $BACKEND_PORT..."
  cd "$ROOT_DIR"
  nohup "$VENV_DIR/bin/python" -m uvicorn interfaces.fastapi_app:app --host 0.0.0.0 --port "$BACKEND_PORT" > "$LOG_DIR/backend.log" 2>&1 &
  echo $! > "$BACKEND_PID_FILE"
  fi
fi

if [[ -f "$FRONTEND_PID_FILE" ]] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
  echo "Frontend already running with PID $(cat "$FRONTEND_PID_FILE")"
else
  existing_frontend_pid="$(pid_on_port "$FRONTEND_PORT")"
  if [[ -n "$existing_frontend_pid" ]]; then
    echo "$existing_frontend_pid" > "$FRONTEND_PID_FILE"
    echo "Frontend already listening on port $FRONTEND_PORT (PID $existing_frontend_pid)"
  else
  echo "Starting frontend on port $FRONTEND_PORT..."
  cd "$ROOT_DIR/frontend"
  nohup npm run dev -- --host 0.0.0.0 --strictPort --port "$FRONTEND_PORT" > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$FRONTEND_PID_FILE"
  fi
fi

cd "$ROOT_DIR"

echo "Waiting for backend health check..."
backend_ready=false
for _ in {1..30}; do
  if curl -s "http://localhost:${BACKEND_PORT}/health" >/dev/null; then
    backend_ready=true
    break
  fi
  sleep 1
done

if [[ "$backend_ready" != "true" ]]; then
  echo "Backend failed to become healthy on port $BACKEND_PORT."
  echo "Check log: $LOG_DIR/backend.log"
  exit 1
fi

frontend_ready=false
for _ in {1..20}; do
  if curl -s "http://localhost:${FRONTEND_PORT}" >/dev/null 2>&1; then
    frontend_ready=true
    break
  fi
  sleep 1
done

if [[ "$frontend_ready" != "true" ]]; then
  echo "Frontend failed to become ready on port $FRONTEND_PORT."
  echo "Check log: $LOG_DIR/frontend.log"
  exit 1
fi

echo "Backend URL:  http://localhost:${BACKEND_PORT}"
echo "Frontend URL: http://localhost:${FRONTEND_PORT}"
echo "Logs:"
echo "  Backend:  $LOG_DIR/backend.log"
echo "  Frontend: $LOG_DIR/frontend.log"
echo "Use ./scripts/stop_all.sh to stop both services."
