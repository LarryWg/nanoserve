#!/usr/bin/env bash
# The whole matrix: both engines online across the rate sweep, all three
# offline. Writes one json per run into results/.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RATES="${RATES:-1 2 4 8 16 32}"
RUNS="${RUNS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
PORT="${PORT:-8000}"
OUT="${OUT:-results}"

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
vllm_python="$here/.venv-vllm/bin/python"
cd "$repo"                      # uv run needs the project, wherever this was called from
mkdir -p "$OUT"

server_pid=""
stop_server() {
    [ -n "$server_pid" ] || return 0
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}
trap stop_server EXIT

wait_for_health() {
    for _ in $(seq 1 300); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null; then return 0; fi
        sleep 2
    done
    echo "server never came up" >&2
    exit 1
}

sweep() {                       # sweep <engine label>
    local engine="$1"
    for rate in $RATES; do
        for run in $(seq 1 "$RUNS"); do
            uv run python "$here/bench.py" \
                --url "http://localhost:$PORT/v1/completions" \
                --model "$MODEL" --engine "$engine" \
                --request-rate "$rate" --num-prompts "$NUM_PROMPTS" \
                --seed "$run" \
                --out "$OUT/online-$engine-r$rate-run$run.json"
        done
    done
}

echo "=== nanoserve, online ==="
uv run python -m nanoserve.server "$MODEL" --port "$PORT" &
server_pid=$!
wait_for_health
sweep nanoserve
stop_server

echo "=== vLLM, online ==="
"$vllm_python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --dtype bfloat16 &
server_pid=$!
wait_for_health
sweep vllm
stop_server

echo "=== offline ==="
for run in $(seq 1 "$RUNS"); do
    uv run python "$here/offline.py" nanoserve --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-nanoserve-run$run.json"
    "$vllm_python" "$here/offline.py" vllm --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-vllm-run$run.json"
    uv run python "$here/offline.py" hf --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-hf-run$run.json"
done

echo "=== done, $(ls "$OUT"/*.json | wc -l) runs in $OUT ==="
