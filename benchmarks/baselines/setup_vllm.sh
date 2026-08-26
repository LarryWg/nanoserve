#!/usr/bin/env bash
# vLLM in its own venv.
#
# vLLM pins its own torch build, which would fight the torch 2.8 and
# flash-attn pins in pyproject.toml. Keeping it separate means running the
# baseline cannot change the thing being measured.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
venv="$here/../.venv-vllm"

# vLLM ships a torch built for CUDA 13, which will not initialise against
# an older driver. Cheaper to say so now than after the install.
# Captured rather than piped: under pipefail a `grep -q` closes the pipe on
# its first match, nvidia-smi dies of SIGPIPE, and the check inverts.
smi=$(nvidia-smi 2>/dev/null || true)
case "$smi" in
    *"CUDA Version: 13"*) ;;
    *)
        echo "This host does not report CUDA 13; vLLM's torch build will not start." >&2
        echo "$smi" | head -4 >&2
        exit 1
        ;;
esac

uv venv --python 3.12 "$venv"
# ninja too: vLLM's flashinfer sampler compiles itself on first start and
# shells out to it. Without it the engine dies before serving anything.
if [ -n "${VLLM_VERSION:-}" ]; then
    VIRTUAL_ENV="$venv" uv pip install "vllm==$VLLM_VERSION" ninja
else
    VIRTUAL_ENV="$venv" uv pip install vllm ninja
fi

# vLLM's engine runs in a child process that shells out to ninja, so it has
# to be findable on the system path, not just inside the venv.
if [ -w /usr/local/bin ] && [ ! -e /usr/local/bin/ninja ]; then
    ln -sf "$venv/bin/ninja" /usr/local/bin/ninja
fi

echo
echo "vLLM installed:"
"$venv/bin/python" -c "import vllm; print(vllm.__version__)"
echo
echo "Pin that version in the results, and pass VLLM_VERSION next time to"
echo "reproduce it exactly."
