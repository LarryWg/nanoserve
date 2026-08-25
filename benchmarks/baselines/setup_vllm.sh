#!/usr/bin/env bash
# vLLM in its own venv.
#
# vLLM pins its own torch build, which would fight the torch 2.8 and
# flash-attn pins in pyproject.toml. Keeping it separate means running the
# baseline cannot change the thing being measured.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
venv="$here/../.venv-vllm"

uv venv --python 3.12 "$venv"
if [ -n "${VLLM_VERSION:-}" ]; then
    VIRTUAL_ENV="$venv" uv pip install "vllm==$VLLM_VERSION"
else
    VIRTUAL_ENV="$venv" uv pip install vllm
fi

echo
echo "vLLM installed:"
"$venv/bin/python" -c "import vllm; print(vllm.__version__)"
echo
echo "Pin that version in the results, and pass VLLM_VERSION next time to"
echo "reproduce it exactly."
