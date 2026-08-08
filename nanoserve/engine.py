"""Engine: the step loop that drives the model.

Loop:
    while True:
        batch = scheduler.step()
        if batch is None: wait for new requests
        logits = model_runner.forward(batch)
        new_tokens = model_runner.sample(logits, batch)
        for seq, tok in zip(batch.seqs, new_tokens):
            seq.on_token(tok, now)  # appends, counts tokens, stamps TTFT
            if seq.is_stopped: scheduler.finish(seq)
        stream tokens out to per-request asyncio queues

Serving architecture (one process per GPU when tensor-parallel):
- FastAPI (async) receives requests and hands each one to the scheduler
  through a thread-safe handoff. The engine loop runs in its own thread.
  GPU kernels release the GIL, so a thread is good enough here; a separate
  process is the alternative if profiling says otherwise.
- /v1/completions streams tokens with SSE, OpenAI-compatible enough that
  standard load-test tools work against it.

Tensor parallelism: see model_runner.py for the sharding plan. Rank 0 owns
the HTTP server, the scheduler, and this loop; it broadcasts the batch
metadata each step and the other ranks run a receive loop.
"""
import threading

from .scheduler import Scheduler
from .sequence import SamplingParams
from .model_runner import ModelRunner


class Engine:
    def __init__(self, scheduler: Scheduler, runner: ModelRunner):
        self.scheduler = scheduler
        self.runner = runner
        self._lock = threading.Lock()
        # TODO: per-seq output queues for streaming; started thread runs _loop.

    def submit(self, prompt_ids: list[int], sampling: SamplingParams) -> int:
        raise NotImplementedError

    def _loop(self):
        raise NotImplementedError
