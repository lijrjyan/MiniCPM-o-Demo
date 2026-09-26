#!/usr/bin/env bash
# sglang-omni backend + worker in one container (see docker/Dockerfile.sglang-omni-worker-backend).
set -euo pipefail

UPSTREAM_URL="${UPSTREAM_URL:?set UPSTREAM_URL to the sglang-omni /v1/realtime URL, e.g. ws://host.docker.internal:18260/v1/realtime}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_PORT="${WORKER_PORT:-22400}"
WORKER_BIND_HOST="${WORKER_BIND_HOST:-0.0.0.0}"
MAX_SESSIONS="${MAX_SESSIONS:-1}"
BACKEND_EXTRA_ARGS="${BACKEND_EXTRA_ARGS:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-300}"
GATEWAY_REGISTRY_URL="${GATEWAY_REGISTRY_URL:-}"
WORKER_ID="${WORKER_ID:-$(hostname)}"
WORKER_ENDPOINT="${WORKER_ENDPOINT:-${WORKER_ID}:${WORKER_PORT}}"
WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-sglang-omni}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"

cd /app
backend_pid=""
worker_pid=""

cleanup() {
    echo "[entrypoint] stopping child processes..."
    [ -n "$worker_pid" ] && kill "$worker_pid" 2>/dev/null || true
    [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "[entrypoint] starting sglang-omni backend -> ${UPSTREAM_URL}"
# shellcheck disable=SC2086
python -m sglang_omni_backend --host 127.0.0.1 --port "$BACKEND_PORT" \
    --upstream-url "$UPSTREAM_URL" --max-sessions "$MAX_SESSIONS" $BACKEND_EXTRA_ARGS &
backend_pid=$!

# /health is 200 only once the upstream sglang-omni server answers its own /health.
for i in $(seq 1 $((READY_TIMEOUT_S / 2))); do
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] backend ready (upstream reachable) after ~$((i * 2))s"
        break
    fi
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] backend exited" >&2
        cleanup
    fi
    sleep 2
done

echo "[entrypoint] starting worker..."
python worker.py --host "$WORKER_BIND_HOST" --port "$WORKER_PORT" --gpu-id 0 --backend-server-url "$BACKEND_URL" &
worker_pid=$!

if [ -n "$GATEWAY_REGISTRY_URL" ]; then
    register_url="${GATEWAY_REGISTRY_URL%/}/${WORKER_ID}"
    echo "[entrypoint] registering worker: ${register_url} endpoint=${WORKER_ENDPOINT}"
    for i in $(seq 1 60); do
        payload="{\"endpoint\":\"${WORKER_ENDPOINT}\",\"gpu_group\":\"${WORKER_GPU_GROUP}\"}"
        if curl -sf -X PUT -H "content-type: application/json" --data "$payload" "$register_url" >/dev/null 2>&1; then
            echo "[entrypoint] worker registered"
            break
        fi
        [ "$i" -eq 60 ] && echo "[entrypoint] warning: worker registration failed; continuing" >&2
        sleep 2
    done
fi

echo "[entrypoint] running. backend pid=${backend_pid} worker pid=${worker_pid}"
while true; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then echo "[entrypoint] backend exited" >&2; cleanup; fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then echo "[entrypoint] worker exited" >&2; cleanup; fi
    sleep 5
done
