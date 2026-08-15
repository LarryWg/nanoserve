# Running the paged path on a GPU

Everything except attention runs anywhere, including a laptop CPU. The paged
KV cache path needs CUDA and flash-attn, which ships as prebuilt wheels
pinned to an exact torch version.

## The version pin that matters

flash-attn does not publish a wheel for every torch release, and its CUDA
extension links against libtorch, so a wheel built for torch 2.8 will not
import under a different minor version. As of flash-attn 2.8.3.post1 the
newest torch with a cu12 wheel is **2.8**; `uv.lock` pins torch 2.13 for the
CPU work, so the GPU box gets its own environment rather than the locked one.

```bash
uv venv --python 3.12 .venv-gpu
VIRTUAL_ENV=$PWD/.venv-gpu uv pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=$PWD/.venv-gpu uv pip install transformers safetensors pytest huggingface_hub
VIRTUAL_ENV=$PWD/.venv-gpu uv pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Pick the wheel by four things, all of which must match: python version
(`cp312`), torch minor (`torch2.8`), CUDA major (`cu12` unless the driver is
new enough for `cu13`), and the C++ ABI flag
(`python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` — True
means `cxx11abiTRUE`). A mismatch shows up as an undefined-symbol error at
`import flash_attn`, not as a failed install.

Building from source instead takes upwards of an hour and needs nvcc; the
wheel is worth the version pin.

## Running the tests

```bash
.venv-gpu/bin/python -m pytest                 # fast suite, GPU files included
.venv-gpu/bin/python -m pytest -m slow         # + the HF equivalence gates
```

Without CUDA or without flash-attn, `test_paged_attention_gpu.py` and
`test_paged_decode_hf.py` skip themselves and everything else still runs.

## Verified on

- RTX 4090 (sm89, 24 GB), driver 570.195, CUDA 12.8
- torch 2.8.0+cu128, flash-attn 2.8.3.post1, transformers 5.15.0, python 3.12
