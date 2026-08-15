"""Model execution: turn a scheduled batch into logits, then into tokens.

The runner owns the weights, the paged KV cache, and the BlockManager for
that cache. It is the only object that can see how much VRAM is free once
the weights are loaded, so it sizes the cache and builds the manager. The
scheduler is handed runner.block_manager and decides who gets which block.

A prefill step runs every token of each admitted sequence, packed flat, and
returns logits for the last position of each one. A decode step runs one
token per running sequence and reads the rest of the context from the cache.

The caller drives one step in this order:

    batch = scheduler.step()      # reserves this step's slot
    logits = runner.forward(batch)
    tokens = runner.sample(logits, batch)
    seq.on_prefilled() / seq.on_token(tok)

So num_computed_tokens already counts the token being fed when forward runs
a decode step. That off by one is where paged decode bugs hide, so it is
spelled out again where it is used.

Tensor parallelism is not written yet. The plan, using torch.distributed
with the NCCL backend:
  attention shards heads, qkv_proj is column parallel and o_proj is row
    parallel, with one all_reduce after o_proj
  the mlp shards the same way, with one all_reduce after down_proj
  that is two all_reduces per layer, worth deriving on paper first
  embedding and lm_head are copied on every rank, which is fine at this size
  rank 0 schedules and broadcasts the batch metadata each step
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
        block_size: int = 256,   # the smallest page the kernel accepts
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        num_blocks: int | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        if tp_size != 1:
            # TODO: shard the weights while loading them, so no rank ever
            # holds a full one.
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
        """How many blocks fit? Measure it, do not guess it.

        Three things share the GPU: the weights, the activations, and the
        cache. Activations peak on the biggest prefill the scheduler can
        send, and nobody can work that out on paper for any model, so we run
        one worst case prefill and read the peak from the allocator.

        The peak comes from torch and the free VRAM comes from the driver,
        and the two do not count quite the same bytes. utilization is the
        margin for the rest, so keep it under 1.0.
        """
        if self.device.type != "cuda":
            raise ValueError(
                f"cannot profile VRAM on device {self.device}, "
                "pass num_blocks explicitly for CPU runs"
            )
        tokens = min(self.max_num_batched_tokens, self.config.max_position_embeddings)
        probe = self._new_cache(num_blocks=-(-tokens // self.block_size))
        torch.cuda.synchronize(self.device)
        resident = torch.cuda.memory_allocated(self.device)  # weights + probe
        torch.cuda.reset_peak_memory_stats(self.device)

        # One sequence this long is the biggest prefill the budget allows,
        # so it is also the biggest activation footprint.
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

    @torch.inference_mode()
    def forward(self, batch) -> torch.Tensor:
        """Run one step for a ScheduledBatch.

        Returns logits [num_seqs, vocab], one row per sequence, for the
        position it is about to sample from. Rows follow batch.seqs order.
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
            # An evicted sequence prefills its prompt plus the tokens it
            # already generated, so this is token_ids.
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
        # cu_seqlens[1:] holds the end of each sequence, so end minus 1 is
        # its last token, the only row the head has to compute.
        logits_indices = self._tensor([end - 1 for end in cu_seqlens[1:]], torch.int64)
        return self.model(
            self._tensor(input_ids, torch.int64),
            self._tensor(positions, torch.int64),
            meta,
            logits_indices,
        )

    def _forward_decode(self, seqs) -> torch.Tensor:
        input_ids: list[int] = []
        # num_computed_tokens already counts the token this step feeds,
        # because the scheduler reserved its slot before calling us. So a
        # sequence with n computed tokens has n minus 1 tokens cached, and
        # feeds a token sitting at position n minus 1. A count and a
        # position that happen to be the same number, so one list does both.
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
        # Every decode token is already a last position, so no indices.
        return self.model(self._tensor(input_ids, torch.int64), positions, meta)

    def _tensor(self, data, dtype: torch.dtype) -> torch.Tensor:
        """Build the step metadata on the host, then move it in one copy.

        Building a tensor per sequence would mean dozens of tiny transfers
        per step. If profiling ever says this matters, pinned host buffers
        would go here.
        """
        return torch.tensor(list(data), dtype=dtype, device=self.device)

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
    """Temperature and top p sampling for a whole batch at once.

    Every request brings its own settings, so temperature and top_p are
    vectors and the batch goes through one set of kernels. A temperature of
    0 means greedy. Those rows still ride through the softmax (clamping
    keeps it finite) and get their argmax back at the end. Branching per row
    would mean a Python loop, which costs more than the maths it saves.

    All of it in fp32. Sampling compares probabilities, and bf16 rounding in
    the tail of a 150k word vocabulary is a different distribution, not a
    rounding error.
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
    """Zero every token outside the smallest set that reaches top_p.

    The mask compares the mass BEFORE each token against top_p, so the token
    that crosses the line is kept. Dropping it would zero the whole row
    whenever one token already carries more than top_p on its own.
    """
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(mass_before >= top_p.unsqueeze(1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_ids, sorted_probs)
