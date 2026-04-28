#!/usr/bin/env bash
# Start Triton/Koina in the background, wait for it to become ready, then exec
# the command supplied to the container (e.g. `make compute_train_features`).
set -euo pipefail

KOINA_HEALTH_URL="${KOINA_HEALTH_URL:-http://localhost:8501/v2/health/ready}"
KOINA_READY_TIMEOUT_SECS="${KOINA_READY_TIMEOUT_SECS:-1800}"

echo "[entrypoint.koina] starting Triton/Koina in background..."
/models/start.py &
KOINA_PID=$!

trap 'echo "[entrypoint.koina] stopping Triton (pid $KOINA_PID)"; kill $KOINA_PID 2>/dev/null || true; wait $KOINA_PID 2>/dev/null || true' EXIT

echo "[entrypoint.koina] waiting for $KOINA_HEALTH_URL (timeout=${KOINA_READY_TIMEOUT_SECS}s) ..."
DEADLINE=$((SECONDS + KOINA_READY_TIMEOUT_SECS))
until curl -fsS "$KOINA_HEALTH_URL" >/dev/null 2>&1; do
  if ! kill -0 "$KOINA_PID" 2>/dev/null; then
    echo "[entrypoint.koina] Triton exited before becoming ready" >&2
    wait "$KOINA_PID" || true
    exit 1
  fi
  if (( SECONDS > DEADLINE )); then
    echo "[entrypoint.koina] timeout after ${KOINA_READY_TIMEOUT_SECS}s waiting for Koina readiness" >&2
    exit 1
  fi
  sleep 5
done

echo "[entrypoint.koina] Koina is ready. Running: $*"
exec "$@"
