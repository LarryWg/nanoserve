"""HTTP server: an OpenAI-compatible front end for the engine.

Two jobs. Turn text into token ids and back, and hand each request's tokens
to the client as they appear. Everything else already happened downstream.

The engine loop runs in its own thread, so nothing here may block: each
handler awaits its own queue and the loop keeps stepping for everybody.
Being OpenAI-compatible is not about the ecosystem, it is so the standard
load-test tools point at this server without a shim.

Not supported yet: string stop sequences (stop token ids work), logprobs,
n > 1, and chat completions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import Engine
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False      # load tests want a fixed output length
    model: str | None = None      # accepted and ignored; we serve one model


class Detokenizer:
    """Turns a growing list of tokens into the text to send next.

    It decodes the whole output every time instead of one token at a time,
    because a token is not a character: spacing and multi-byte characters
    only come out right alongside their neighbours. Quadratic in the output
    length, and still nothing next to a forward pass.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self.text = ""

    def add(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        text = self._tokenizer.decode(self._token_ids)
        if text.endswith("�"):
            # Half a character. Hold it back until the next token finishes it.
            return ""
        delta, self.text = text[len(self.text):], text
        return delta


def create_app(engine: Engine, tokenizer, model_name: str = "nanoserve") -> FastAPI:
    """Wire an engine and a tokenizer up to HTTP.

    Both are passed in rather than built here, so tests can serve a fake
    model without a GPU.
    """
    stop_ids = tuple(i for i in [getattr(tokenizer, "eos_token_id", None)] if i is not None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start(asyncio.get_running_loop())
        yield
        engine.stop()

    app = FastAPI(title="nanoserve", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        prompt_ids = tokenizer.encode(body.prompt)
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt is empty")
        sampling = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            max_new_tokens=body.max_tokens,
            stop_token_ids=() if body.ignore_eos else stop_ids,
        )
        try:
            seq_id = engine.submit(prompt_ids, sampling)
        except ValueError as exc:
            # The prompt is bigger than a step or bigger than the cache.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ctx = _Completion(engine, tokenizer, seq_id, len(prompt_ids), model_name, stop_ids)
        if body.stream:
            return StreamingResponse(ctx.stream(), media_type="text/event-stream")
        return await ctx.collect()

    return app


class _Completion:
    """One in-flight request, in either of its two shapes."""

    def __init__(self, engine, tokenizer, seq_id, num_prompt_tokens, model_name, stop_ids):
        self.engine = engine
        self.seq_id = seq_id
        self.num_prompt_tokens = num_prompt_tokens
        self.model_name = model_name
        self.stop_ids = stop_ids
        self.detokenizer = Detokenizer(tokenizer)
        self.created = int(time.time())
        self.num_output_tokens = 0

    async def stream(self):
        try:
            async for out in self.engine.outputs(self.seq_id):
                delta = self._consume(out)
                if not delta and not out.finished:
                    continue        # a held-back partial character
                yield _sse(self._body(delta, self._reason(out)))
            yield "data: [DONE]\n\n"
        finally:
            # Reached on a client disconnect too, which is the point: the
            # request keeps its KV blocks until somebody says otherwise.
            self.engine.abort(self.seq_id)

    async def collect(self) -> dict:
        reason = None
        try:
            async for out in self.engine.outputs(self.seq_id):
                self._consume(out)
                reason = self._reason(out)
        finally:
            self.engine.abort(self.seq_id)
        body = self._body(self.detokenizer.text, reason)
        body["usage"] = {
            "prompt_tokens": self.num_prompt_tokens,
            "completion_tokens": self.num_output_tokens,
            "total_tokens": self.num_prompt_tokens + self.num_output_tokens,
        }
        return body

    def _consume(self, out) -> str:
        self.num_output_tokens += 1
        return self.detokenizer.add(out.token_id)

    def _reason(self, out) -> str | None:
        if not out.finished:
            return None
        return "stop" if out.token_id in self.stop_ids else "length"

    def _body(self, text: str, reason: str | None) -> dict:
        return {
            "id": f"cmpl-{self.seq_id}",
            "object": "text_completion",
            "created": self.created,
            "model": self.model_name,
            "choices": [
                {"index": 0, "text": text, "logprobs": None, "finish_reason": reason}
            ],
        }


def _sse(body: dict) -> str:
    return f"data: {json.dumps(body)}\n\n"


def build_app(model: str, **runner_kwargs) -> FastAPI:
    """Load the model and build a server around it."""
    from transformers import AutoTokenizer

    path = model
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(model)

    max_num_seqs = runner_kwargs.pop("max_num_seqs", 64)
    runner = ModelRunner(path, **runner_kwargs)
    scheduler = Scheduler(
        block_manager=runner.block_manager,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=runner.max_num_batched_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    return create_app(Engine(scheduler, runner), tokenizer, model_name=model)


def main():
    parser = argparse.ArgumentParser(description="Serve a model over HTTP.")
    parser.add_argument("model", help="HF repo id or a local checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    import uvicorn

    app = build_app(
        args.model,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
