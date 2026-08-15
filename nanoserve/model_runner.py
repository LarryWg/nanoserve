"""Model execution: turn a ScheduledBatch into logits, then into tokens.

The runner owns the physical side of the engine: the weights, the paged KV
cache, and (because it is the only object that can measure how much VRAM is
left after the weights load) the BlockManager that hands out block ids for
that cache. The scheduler is handed `runner.block_manager`; it decides which
blocks a sequence gets, the runner decides what goes in them.

Two kinds of step, one entry point:
- Prefill: every token of each admitted sequence, packed flat, one
  flash_attn_varlen_func call per layer, K/V scattered into the cache.
  Logits come back for the LAST position of each sequence only.
- Decode: exactly one token per running sequence, gathered through block
  tables by flash_attn_with_kvcache, which also appends the new K/V.

The contract with the caller (the engine loop, and the tests that stand in
for it until it exists) is a strict order per step:

    batch = scheduler.step()      # reserves the KV slot for this step
    logits = runner.forward(batch)
    tokens = runner.sample(logits, batch)
    seq.on_prefilled() / seq.on_token(tok)   # advance the sequence

`Sequence.num_computed_tokens` therefore already counts the token being fed
when forward runs a decode step. That off-by-one is deliberate (the block
was reserved before the step) and is where every paged-decode bug hides, so
it is spelled out again at the point of use in _forward_decode.

Tensor parallelism (raw torch.distributed, NCCL backend), still to come:
- Attention: shard heads across ranks. qkv_proj is column-parallel, o_proj
  is row-parallel, with one all_reduce after o_proj.
- MLP: gate/up_proj are column-parallel, down_proj is row-parallel, with
  one all_reduce after down_proj.
- Exactly 2 all_reduces per layer. Deriving why on paper before coding is
  the exercise.
- Embedding and lm_head: replicate on every rank (fine at this scale;
  quantify the memory cost when writing it up).
- Every rank must run the same scheduler decisions: rank 0 schedules and
  broadcasts the batch metadata each step.

Profiling to capture as you go:
- Nsight Systems trace showing compute/NCCL overlap (or lack of it)
- decode step time vs batch size (find where the step goes memory-bound)
"""
from __future__ import annotations

import torch

from .block_manager import BlockManager
from .kv_cache import KVCache, bytes_per_block, pad_block_tables, slot_mapping
from .model.attention import AttentionMetadata
from .model.model import NanoForCausalLM


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        block_size: int = 256,   # the kernel's minimum page; see kv_cache.py
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        num_blocks: int | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        if tp_size != 1:
            # TODO (multi-GPU): shard weights at load time based on
            #   (tp_rank, tp_size); never materialize the full weight on
            #   every rank. Measuring load time and peak host RAM is a nice
            #   detail for the writeup.
            raise NotImplementedError("tensor parallelism is not implemented yet")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = torch.device(device)
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens

        self.model = NanoForCausalLM.from_pretrained(model_path, device=device)
        self.model.eval()
        self.config = self.model.config

        if num_blocks is None:
            num_blocks = self._profile_num_blocks(gpu_memory_utilization)
        self.kv_cache = self._new_cache(num_blocks)
        self.block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)

    # ---------- setup ----------

    def _new_cache(self, num_blocks: int) -> KVCache:
        return KVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.dtype,
            device=self.device,
        )

    def _profile_num_blocks(self, utilization: float) -> int:
        """How many blocks fit? Measure, do not estimate.

        Three things share the GPU: weights (already resident), activations
        (transient, peaking on the largest prefill the scheduler can send),
        and the KV cache (whatever is left). Activations are the term nobody
        can compute on paper for an arbitrary model, so we run one worst-case
        prefill against a throwaway cache and read the allocator's peak.

        The two numbers come from different places on purpose: the peak is a
        torch-allocator number, the free VRAM is a driver number that also
        sees other processes and the CUDA context. utilization is the margin
        for everything neither of them models (fragmentation, cuBLAS
        workspaces), so keep it below 1.0.
        """
        if self.device.type != "cuda":
            raise ValueError(
                f"cannot profile VRAM on device {self.device}; "
                "pass num_blocks explicitly for CPU runs"
            )
        tokens = min(self.max_num_batched_tokens, self.config.max_position_embeddings)
        probe = self._new_cache(num_blocks=-(-tokens // self.block_size))
        torch.cuda.synchronize(self.device)
        resident = torch.cuda.memory_allocated(self.device)  # weights + probe
        torch.cuda.reset_peak_memory_stats(self.device)

        # One sequence of `tokens` tokens is the largest prefill the token
        # budget allows, and the largest activation footprint per step.
        meta = AttentionMetadata(
            is_prefill=True,
            kv_cache=probe,
            slot_mapping=self._tensor(range(tokens), torch.int64),
            cu_seqlens=self._tensor([0, tokens], torch.int32),
            max_seqlen=tokens,
        )
        with torch.inference_mode():
            self.model(
                self._tensor([0] * tokens, torch.int64),
                self._tensor(range(tokens), torch.int64),
                meta,
                self._tensor([tokens - 1], torch.int64),
            )
        torch.cuda.synchronize(self.device)
        activation_peak = torch.cuda.max_memory_allocated(self.device) - resident

        del probe, meta
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        budget = free - (1.0 - utilization) * total - activation_peak
        per_block = bytes_per_block(
            self.config.num_hidden_layers,
            self.block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.dtype,
        )
        num_blocks = int(budget // per_block)
        if num_blocks <= 0:
            raise RuntimeError(
                f"no VRAM left for the KV cache: {free / 2**30:.1f} GiB free, "
                f"{activation_peak / 2**30:.1f} GiB of activations at "
                f"{tokens} batched tokens"
            )
        return num_blocks

    # ---------- execution ----------

    @torch.inference_mode()
    def forward(self, batch) -> torch.Tensor:
        """Run one step for a ScheduledBatch.

        Returns logits [num_seqs, vocab]: one row per sequence, for the
        position that sequence is about to sample from. Rows are in
        batch.seqs order, which is what sample() and the engine rely on.
        """
        if batch.is_prefill:
            return self._forward_prefill(batch.seqs)
        return self._forward_decode(batch.seqs)

    def _forward_prefill(self, seqs) -> torch.Tensor:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        cu_seqlens = [0]
        for seq in seqs:
            # A preempted sequence re-prefills prompt + already-generated
            # tokens, so this is token_ids, not prompt_token_ids.
            tokens = seq.token_ids
            table = self.block_manager.get_block_table(seq.seq_id)
            input_ids.extend(tokens)
            positions.extend(range(len(tokens)))
            slots.extend(slot_mapping(table, range(len(tokens)), self.block_size))
            cu_seqlens.append(cu_seqlens[-1] + len(tokens))

        meta = AttentionMetadata(
            is_prefill=True,
            kv_cache=self.kv_cache,
            slot_mapping=self._tensor(slots, torch.int64),
            cu_seqlens=self._tensor(cu_seqlens, torch.int32),
            max_seqlen=max(len(seq.token_ids) for seq in seqs),
        )
        # cu_seqlens[1:] are the ends of each sequence, so end - 1 is its last
        # token: the only row the head has to compute (see NanoForCausalLM).
        logits_indices = self._tensor([end - 1 for end in cu_seqlens[1:]], torch.int64)
        return self.model(
            self._tensor(input_ids, torch.int64),
            self._tensor(positions, torch.int64),
            meta,
            logits_indices,
        )

    def _forward_decode(self, seqs) -> torch.Tensor:
        input_ids: list[int] = []
        # num_computed_tokens already counts the token this step feeds: the
        # scheduler reserved its KV slot before calling us. So for a sequence
        # with n computed tokens, exactly n - 1 tokens are already cached,
        # and the new token sits at position n - 1 -- a count and a
        # 0-indexed position that happen to be the same integer. One list
        # serves as both, which is safe precisely because the engine
        # reserves the slot before the step and fills it during it.
        cached_lens: list[int] = []
        tables: list[list[int]] = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            cached_lens.append(seq.num_computed_tokens - 1)
            tables.append(self.block_manager.get_block_table(seq.seq_id))

        positions = self._tensor(cached_lens, torch.int64)
        meta = AttentionMetadata(
            is_prefill=False,
            kv_cache=self.kv_cache,
            block_tables=self._tensor(pad_block_tables(tables), torch.int32),
            cache_seqlens=self._tensor(cached_lens, torch.int32),
        )
        # Every decode token IS a last position, so no logits_indices.
        return self.model(self._tensor(input_ids, torch.int64), positions, meta)

    def _tensor(self, data, dtype: torch.dtype) -> torch.Tensor:
        """Build step metadata host-side, then move it in one copy.

        Per-sequence tensor construction would mean dozens of tiny H2D
        transfers per step, each with its own launch latency. Whether this
        shows up at all is a profiling question; the shape of the answer is
        pinned host buffers, and this is where that change would land.
        """
        return torch.tensor(list(data), dtype=dtype, device=self.device)

    # ---------- sampling ----------

    def sample(self, logits: torch.Tensor, batch) -> list[int]:
        """Pick one token per sequence, in batch.seqs order."""
        return sample_tokens(
            logits,
            self._tensor([s.sampling.temperature for s in batch.seqs], torch.float32),
            self._tensor([s.sampling.top_p for s in batch.seqs], torch.float32),
        ).tolist()


def sample_tokens(
    logits: torch.Tensor,        # [num_seqs, vocab]
    temperatures: torch.Tensor,  # [num_seqs]
    top_p: torch.Tensor,         # [num_seqs]
) -> torch.Tensor:               # [num_seqs] int64
    """Temperature + top-p sampling for a whole batch at once.

    Sampling params are per request, so temperature and top_p are vectors,
    not scalars, and the batch goes through one set of kernels.
    temperature == 0 means greedy; those rows still ride along through the
    softmax (a clamped temperature keeps it finite) and are overwritten with
    their argmax at the end. Branching per row would mean a Python loop over
    the batch, which costs more than the arithmetic it saves.

    fp32 throughout: sampling reads probability ratios, and bf16 rounding
    at the tail of a 150k-entry distribution is not a rounding error there,
    it is a different distribution.
    """
    logits = logits.float()
    greedy_ids = logits.argmax(dim=-1)
    greedy = temperatures == 0
    if bool(greedy.all()):
        return greedy_ids

    probs = torch.softmax(logits / temperatures.clamp(min=1e-5).unsqueeze(1), dim=-1)
    if bool((top_p < 1.0).any()):
        probs = _top_p_filter(probs, top_p)
    # multinomial renormalizes, so the zeroed tail costs nothing extra.
    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.where(greedy, greedy_ids, sampled_ids)


def _top_p_filter(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Zero every token outside the smallest set whose mass reaches top_p.

    The mask compares the cumulative mass BEFORE each token against top_p,
    so the token that crosses the threshold is kept. Dropping it instead
    would make top_p slightly smaller than asked, and would zero the whole
    row whenever one token already carries more than top_p of the mass.
    """
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(mass_before >= top_p.unsqueeze(1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_ids, sorted_probs)
