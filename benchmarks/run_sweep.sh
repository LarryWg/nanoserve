#!/usr/bin/env bash
# The whole matrix: both engines across the rate sweep, all three offline.
# Writes one json per run into results/.
#
# Each engine loads once and sweeps every rate in that one process, so the
# weights and the measured cache size are identical across its points.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RATES="${RATES:-1 2 4 8 16 32}"
RUNS="${RUNS:-3}"
HF_RUNS="${HF_RUNS:-1}"
HF_BATCH_SIZES="${HF_BATCH_SIZES:-16,32,64}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUT="${OUT:-results}"
STAGES="${STAGES:-nanoserve vllm offline}"
EXTRA="${EXTRA:-}"                  # extra flags for the nanoserve driver

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
vllm_python="$here/.venv-vllm/bin/python"
vllm_path="$here/.venv-vllm/bin:$PATH"    # vLLM shells out to ninja
cd "$repo"
mkdir -p "$OUT"

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if has_stage nanoserve; then
echo "=== nanoserve, online ==="
uv run python "$here/online.py" --engine nanoserve --model-path "$MODEL" \
    --rates "$(echo $RATES | tr ' ' ,)" --runs "$RUNS" \
    --num-prompts "$NUM_PROMPTS" $EXTRA --out "$OUT/online-nanoserve.json"
fi

if has_stage vllm; then
echo "=== vLLM, online ==="
PATH="$vllm_path" "$vllm_python" "$here/online.py" --engine vllm \
    --model-path "$MODEL" --rates "$(echo $RATES | tr ' ' ,)" --runs "$RUNS" \
    --num-prompts "$NUM_PROMPTS" --out "$OUT/online-vllm.json"
fi

if has_stage offline; then
echo "=== offline ==="
for run in $(seq 1 "$RUNS"); do
    uv run python "$here/offline.py" nanoserve --model-path "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --seed "$run" \
        --out "$OUT/offline-nanoserve-run$run.json"
    PATH="$vllm_path" "$vllm_python" "$here/offline.py" vllm --model-path "$MODEL" \
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
