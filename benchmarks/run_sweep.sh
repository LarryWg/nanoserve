#!/usr/bin/env bash
# The whole matrix: both engines online across the rate sweep, all three
# offline. Writes one json per run into results/.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RATES="${RATES:-1 2 4 8 16 32}"
RUNS="${RUNS:-3}"
# HF is a reference point rather than the subject, and its static batching
# sweep is slow, so it defaults to a single run.
HF_RUNS="${HF_RUNS:-1}"
HF_BATCH_SIZES="${HF_BATCH_SIZES:-16,32,64}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
PORT="${PORT:-8000}"
OUT="${OUT:-results}"
# Which stages to run. An hour-long sweep that dies halfway should be
# resumable without throwing away the runs that already landed.
STAGES="${STAGES:-nanoserve vllm offline}"

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
vllm_python="$here/.venv-vllm/bin/python"
cd "$repo"                      # uv run needs the project, wherever this was called from
mkdir -p "$OUT"

server_pid=""

# Servers start with setsid so each owns a process group, and teardown
# kills the group. vLLM runs its engine as a child that renames itself, so
# signalling the parent alone leaves it holding the whole GPU.
start_server() {
    setsid "$@" &
    server_pid=$!
}

stop_server() {
    [ -n "$server_pid" ] || return 0
    kill -TERM -"$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -"$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
    wait_for_gpu_free
}
trap stop_server EXIT

wait_for_gpu_free() {
    # Both engines size their KV cache from free VRAM, so anything left
    # behind would quietly shrink the next one's cache and every number
    # after it would be measuring a different machine.
    local used
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$used" -lt 1000 ]; then return 0; fi
        sleep 2
    done
    echo "GPU still holds ${used} MiB; refusing to benchmark against it" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
}

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

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if has_stage nanoserve; then
echo "=== nanoserve, online ==="
wait_for_gpu_free
start_server uv run python -m nanoserve.server "$MODEL" --port "$PORT"
wait_for_health
sweep nanoserve
stop_server
fi

if has_stage vllm; then
echo "=== vLLM, online ==="
wait_for_gpu_free
# The venv's bin joins PATH so vLLM can find ninja when it compiles.
PATH="$here/.venv-vllm/bin:$PATH" start_server "$vllm_python" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --dtype bfloat16
wait_for_health
sweep vllm
stop_server
fi

if has_stage offline; then
echo "=== offline ==="
wait_for_gpu_free
for run in $(seq 1 "$RUNS"); do
    uv run python "$here/offline.py" nanoserve --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-nanoserve-run$run.json"
    "$vllm_python" "$here/offline.py" vllm --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-vllm-run$run.json"
done

for run in $(seq 1 "$HF_RUNS"); do
    uv run python "$here/offline.py" hf --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --hf-batch-sizes "$HF_BATCH_SIZES" \
        --out "$OUT/offline-hf-run$run.json"
done
fi

echo "=== done, $(ls "$OUT"/*.json | wc -l) runs in $OUT ==="
