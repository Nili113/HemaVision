#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.runlogs"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

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

check_service() {
  local name="$1"
  local pid_file="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: running (PID $pid)"
      return
    fi
  fi

  local port_pid
  port_pid="$(pid_on_port "$3")"
  if [[ -n "$port_pid" ]]; then
    echo "$name: running on port $3 (PID $port_pid, not tracked by PID file)"
    return
  fi

  echo "$name: stopped"
}

check_service "backend" "$BACKEND_PID_FILE" "$BACKEND_PORT"
check_service "frontend" "$FRONTEND_PID_FILE" "$FRONTEND_PORT"
