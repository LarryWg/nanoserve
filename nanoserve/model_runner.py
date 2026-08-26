from __future__ import annotations

import torch

from .block_manager import BlockManager
from .kv_cache import KVCache, bytes_per_block, pad_block_tables, slot_mapping
from .model.attention import AttentionMetadata
from .model.model import NanoForCausalLM
from .scheduler import ScheduledBatch
from .sequence import Sequence


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        block_size: int = 256,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        num_blocks: int | None = None,
    ):
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
        if self.device.type != "cuda":
            raise ValueError(
                f"cannot profile VRAM on device {self.device}, "
                "pass num_blocks explicitly for CPU runs"
            )
        tokens = min(self.max_num_batched_tokens, self.config.max_position_embeddings)
        activation_peak = self._measure_activation_peak(tokens)

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

    def _measure_activation_peak(self, tokens: int) -> int:
        probe = self._new_cache(num_blocks=-(-tokens // self.block_size))
        torch.cuda.synchronize(self.device)
        resident = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

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
        peak = torch.cuda.max_memory_allocated(self.device) - resident

        del probe, meta
        torch.cuda.empty_cache()
        return peak

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> torch.Tensor:
        if batch.is_prefill:
            return self._forward_prefill(batch.seqs)
        return self._forward_decode(batch.seqs)

    def _forward_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        cu_seqlens = [0]
        max_seqlen = 0
        for seq in seqs:
            tokens = seq.token_ids
            table = self.block_manager.get_block_table(seq.seq_id)
            input_ids.extend(tokens)
            positions.extend(range(len(tokens)))
            slots.extend(slot_mapping(table, range(len(tokens)), self.block_size))
            cu_seqlens.append(cu_seqlens[-1] + len(tokens))
            max_seqlen = max(max_seqlen, len(tokens))

        meta = AttentionMetadata(
            is_prefill=True,
            kv_cache=self.kv_cache,
            slot_mapping=self._tensor(slots, torch.int64),
            cu_seqlens=self._tensor(cu_seqlens, torch.int32),
            max_seqlen=max_seqlen,
        )
        logits_indices = self._tensor([end - 1 for end in cu_seqlens[1:]], torch.int64)
        return self.model(
            self._tensor(input_ids, torch.int64),
            self._tensor(positions, torch.int64),
            meta,
            logits_indices,
        )

    def _forward_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids: list[int] = []
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
        return self.model(self._tensor(input_ids, torch.int64), positions, meta)

    def _tensor(self, data, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(list(data), dtype=dtype, device=self.device)

    def sample(self, logits: torch.Tensor, batch: ScheduledBatch) -> list[int]:
        return sample_tokens(
            logits,
            self._tensor([s.sampling.temperature for s in batch.seqs], torch.float32),
            self._tensor([s.sampling.top_p for s in batch.seqs], torch.float32),
        ).tolist()


def sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    greedy_ids = logits.argmax(dim=-1)
    greedy = temperatures == 0
    if bool(greedy.all()):
        return greedy_ids

    probs = torch.softmax(logits / temperatures.clamp(min=1e-5).unsqueeze(1), dim=-1)
    if bool((top_p < 1.0).any()):
        probs = _top_p_filter(probs, top_p)
    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.where(greedy, greedy_ids, sampled_ids)


def _top_p_filter(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(mass_before >= top_p.unsqueeze(1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_ids, sorted_probs)
