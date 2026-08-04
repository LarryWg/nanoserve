"""Engine: the step loop. Plumbing — fine to build quickly, but understand it.

Loop:
    while True:
        batch = scheduler.step()
        if batch is None: wait for new requests
        logits = model_runner.forward(batch)
        new_tokens = model_runner.sample(logits, batch)
        for seq, tok in zip(batch.seqs, new_tokens):
            seq.output_token_ids.append(tok)
            record first_token_time on first decode token (TTFT!)
            if stop condition (eos / max_new_tokens): scheduler.finish(seq)
        stream tokens out to per-request asyncio queues

Serving architecture (single process per TP rank):
- FastAPI (async) receives requests -> puts Sequence into scheduler via
  a thread-safe handoff -> engine loop runs in its own thread (GPU work
  releases the GIL during kernels; good enough here — note this in the
  writeup and mention the multiprocess alternative).
- /v1/completions with streaming (SSE), OpenAI-compatible enough that
  standard load-test tools work against it.

TP note (Stage 2): rank 0 owns the HTTP server + scheduler; broadcasts
batch metadata to ranks 1..N-1 each step, all ranks call forward(),
rank 0 samples and streams. Ranks 1..N-1 run a receive-loop.
"""
import threading
import time

from .scheduler import Scheduler
from .sequence import Sequence, SamplingParams
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
