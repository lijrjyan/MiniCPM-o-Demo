#!/usr/bin/env bash
# Bare-metal gateway + worker + sglang-omni backend (no model weights, no GPU on this side).
#
#   scripts/sglang_omni_local.sh start|stop|status|restart
#
# The sglang-omni server must already run (UPSTREAM_URL). Everything binds to BIND_HOST.
# Settings come from the environment (defaults below) or scripts/sglang_omni_local.env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f scripts/sglang_omni_local.env ] && . scripts/sglang_omni_local.env

PYTHON="${PYTHON:-python3}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
UPSTREAM_URL="${UPSTREAM_URL:-ws://127.0.0.1:18260/v1/realtime}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_PORT="${WORKER_PORT:-22400}"
GATEWAY_PORT="${GATEWAY_PORT:-8006}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8007}"
WORKERS="${WORKERS:-1}"            # one worker per concurrent session; the backend gets --max-sessions $WORKERS
BACKEND_ARGS="${BACKEND_ARGS:-}"   # e.g. "--no-silence-fill" or "--no-forward-images"
RUN_DIR="${RUN_DIR:-$ROOT/run}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$RUN_DIR" "$LOG_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

alive() { [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

launch() {  # name, command...
    local name="$1"; shift
    if alive "$name"; then echo "$name already running (pid $(cat "$RUN_DIR/$name.pid"))"; return; fi
    nohup "$@" >> "$LOG_DIR/$name.log" 2>&1 &
    echo $! > "$RUN_DIR/$name.pid"
    echo "$name started (pid $!), log $LOG_DIR/$name.log"
}

wait_http() {  # url, seconds
    for _ in $(seq 1 "$2"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done
    echo "timeout waiting for $1" >&2; return 1
}

start() {
    launch backend "$PYTHON" -m sglang_omni_backend --host "$BIND_HOST" --port "$BACKEND_PORT" \
        --upstream-url "$UPSTREAM_URL" --max-sessions "$WORKERS" $BACKEND_ARGS
    wait_http "http://$BIND_HOST:$BACKEND_PORT/health" 30 || { echo "backend not ready (is the sglang-omni server up at $UPSTREAM_URL?)"; }
    for i in $(seq 0 $((WORKERS - 1))); do
        launch "worker-$i" "$PYTHON" worker.py --host "$BIND_HOST" --port $((WORKER_PORT + i)) --gpu-id 0 \
            --backend-server-url "http://$BIND_HOST:$BACKEND_PORT"
    done
    launch gateway "$PYTHON" gateway.py --host "$BIND_HOST" --port "$GATEWAY_PORT" \
        --internal-port "$GATEWAY_INTERNAL_PORT" --http
    wait_http "http://$BIND_HOST:$GATEWAY_PORT/health" 60
    for i in $(seq 0 $((WORKERS - 1))); do
        wait_http "http://$BIND_HOST:$((WORKER_PORT + i))/health" 30
        curl -sf -X PUT "http://$BIND_HOST:$GATEWAY_INTERNAL_PORT/internal/workers/sglang-omni-$i" \
            -H 'content-type: application/json' \
            --data "{\"endpoint\":\"$BIND_HOST:$((WORKER_PORT + i))\",\"gpu_group\":\"sglang-omni\"}" >/dev/null
        echo "worker-$i registered with the gateway"
    done
    status
}

stop() {
    for pidfile in "$RUN_DIR"/*.pid; do
        [ -e "$pidfile" ] || continue
        pid="$(cat "$pidfile")"
        kill "$pid" 2>/dev/null && echo "stopped $(basename "$pidfile" .pid) (pid $pid)" || true
        rm -f "$pidfile"
    done
}

status() {
    echo "backend: $(curl -s "http://$BIND_HOST:$BACKEND_PORT/health" || echo down)"
    for i in $(seq 0 $((WORKERS - 1))); do
        echo "worker-$i: $(curl -s "http://$BIND_HOST:$((WORKER_PORT + i))/health" || echo down)"
    done
    echo "gateway: $(curl -s "http://$BIND_HOST:$GATEWAY_PORT/health" || echo down)"
    echo "page: http://$BIND_HOST:$GATEWAY_PORT/  (omni: /omni, audio duplex: /audio_duplex)"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; sleep 1; start ;;
    *) echo "usage: $0 start|stop|status|restart" >&2; exit 2 ;;
esac
