# Running the paged path on a GPU

Everything except attention runs anywhere, including a laptop CPU. The paged
KV cache path needs CUDA and flash attn.

## Setup

Same two commands as anywhere else:

```bash
uv sync
uv run pytest              # fast tests
uv run pytest -m slow      # adds the HF equivalence gates
```

On a linux box `uv sync` installs the CUDA kernels too. On macOS the markers
in `pyproject.toml` leave them out. Nothing to remember per machine.

The pod needs python 3.12, because that is what the prebuilt kernel wheel was
built for:

```bash
uv sync --python 3.12
```

## Why the versions are pinned

flash attn does not build at install time, it ships prebuilt wheels, and each
wheel is compiled against one exact torch version. Its CUDA extension links
against libtorch, so a wheel built for torch 2.8 fails to import under any
other torch with an undefined symbol error, not a clean message.

So `pyproject.toml` pins four things that have to agree, all of them visible
in the wheel filename `flash_attn 2.8.3.post1+cu12torch2.8cxx11abiTRUE
cp312 linux_x86_64`:

| what | value | how to check |
| --- | --- | --- |
| python | 3.12 | `uv run python -V` |
| torch | 2.8 | `uv run python -c "import torch; print(torch.__version__)"` |
| CUDA major | 12 | `nvidia-smi`, needs a driver that supports it |
| C++ ABI | true | `uv run python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` |

To move to a newer torch, find the matching wheel on the flash attn releases
page and update both the torch pin and the wheel URL together. As of flash
attn 2.8.3.post1 the newest torch with a CUDA 12 wheel is 2.8.

## When the environment is wrong

The GPU test files import flash attn directly whenever CUDA is present, so a
half installed environment fails at collection instead of quietly skipping.
A green run with those files skipped means no GPU, never a bad install.

## Verified on

- RTX 4090 (sm89, 24 GB), driver 570.195, CUDA 12.8
- torch 2.8.0+cu126, flash attn 2.8.3.post1, transformers 5.15.0, python 3.12
