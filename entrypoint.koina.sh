#!/usr/bin/env bash
# Start Triton/Koina in the background, wait for it to become ready, then exec
# the command supplied to the container (e.g. `make compute_train_features`).
set -euo pipefail

KOINA_HEALTH_URL="${KOINA_HEALTH_URL:-http://localhost:8501/v2/health/ready}"
KOINA_READY_TIMEOUT_SECS="${KOINA_READY_TIMEOUT_SECS:-1800}"

# Triton's Python backend stub (linked against libpython3.10.so) computes its
# sys.path from the stub binary's location under /opt/tritonserver/, so it does
# not pick up /usr/local/lib/python3.10/dist-packages where pip installed numpy
# / pandas / etc. Force the stub to look there via PYTHONPATH on the Triton
# invocation only -- the stubs inherit it from Triton, but if we exported it
# globally the foreground command (e.g. winnow inside a Python 3.12 venv) would
# also prepend /usr/local/lib/python3.10/dist-packages to its sys.path and pick
# up stale pure-Python copies of `click` etc. instead of the venv's versions.
echo "[entrypoint.koina] starting Triton/Koina in background..."
PYTHONPATH="/usr/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}" \
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
