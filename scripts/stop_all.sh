#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.runlogs"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

stop_service() {
  local name="$1"
  local pid_file="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID $pid)..."
      kill "$pid"
    else
      echo "$name PID file exists but process is not running."
    fi
    rm -f "$pid_file"
  else
    echo "$name is not running."
  fi
}

stop_service "backend" "$BACKEND_PID_FILE"
stop_service "frontend" "$FRONTEND_PID_FILE"

echo "Done."
