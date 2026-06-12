"""GPU probe state for streaming sentence-level attention features."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class ProbeState:
    """Buffers for last-row attention probe + in-layer sentence pooling."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_layers: int,
        num_heads: int,
        use_triton: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_triton = use_triton

        self.active = False
        self.batch_size = 1
        self.context_start = 0
        self.context_end = 0
        self.query_pos: Optional[int] = None

        self._sent_masks: Optional[torch.Tensor] = None
        self._feat_acc: Optional[torch.Tensor] = None
        self._num_sents = 0
        self._layer_idx = 0
        self._token_sent_id: Optional[torch.Tensor] = None
        self._sent_lengths: Optional[torch.Tensor] = None
        self._valid_sents: Optional[torch.Tensor] = None
        self._batch_ctx_starts: Optional[torch.Tensor] = None
        self._batch_ctx_ends: Optional[torch.Tensor] = None

        self._cached_num_sents = 0
        self._cached_ctx_len = 0
        self._cached_sent_key: Optional[Tuple] = None
        self._cached_batch_key: Optional[Tuple] = None

    def _build_sentence_buffers(
        self,
        sent_positions: List[Tuple[int, int]],
        context_start: int,
        context_end: int,
    ) -> int:
        max_ctx_len = max(context_end - context_start + 1, 0)
        num_sents = len(sent_positions)
        if max_ctx_len <= 0 or not sent_positions:
            self._sent_masks = None
            self._token_sent_id = None
            self._sent_lengths = None
            return max_ctx_len

        starts, ends = zip(*sent_positions)
        rel_starts = (
            torch.tensor(starts, dtype=torch.long, device=self.device) - context_start
        ).clamp(0, max_ctx_len - 1)
        rel_ends = (
            torch.tensor(ends, dtype=torch.long, device=self.device) - context_start
        ).clamp(0, max_ctx_len - 1)
        rel_ends = torch.maximum(rel_ends, rel_starts)
        lengths = (rel_ends - rel_starts + 1).clamp(min=1)
        self._sent_lengths = lengths.to(self.dtype)

        j = torch.arange(max_ctx_len, device=self.device, dtype=torch.long)
        in_range = (j.unsqueeze(0) >= rel_starts.unsqueeze(1)) & (
            j.unsqueeze(0) <= rel_ends.unsqueeze(1)
        )
        self._sent_masks = in_range.to(self.dtype) / lengths.unsqueeze(1).clamp(min=1)

        token_sent = torch.full((max_ctx_len,), -1, dtype=torch.long, device=self.device)
        for s_idx, (rs, re) in enumerate(zip(rel_starts.tolist(), rel_ends.tolist())):
            token_sent[rs : re + 1] = s_idx
        self._token_sent_id = token_sent
        return max_ctx_len

    def begin(
        self,
        sent_positions: List[Tuple[int, int]],
        context_start: int,
        context_end: int,
        query_pos: Optional[int] = None,
    ) -> None:
        self.active = True
        self.batch_size = 1
        self.context_start = context_start
        self.context_end = context_end
        self.query_pos = query_pos
        self._layer_idx = 0
        self._num_sents = len(sent_positions)
        self._valid_sents = None

        if self._cached_batch_key is not None or (
            self._feat_acc is not None and self._feat_acc.ndim != 2
        ):
            self._cached_batch_key = None
            self._cached_sent_key = None
            self._sent_masks = None
            self._feat_acc = None

        sent_key = tuple(sent_positions)
        max_ctx_len = max(context_end - context_start + 1, 0)
        if (
            max_ctx_len > 0
            and self._num_sents > 0
            and self._cached_sent_key == sent_key
            and self._cached_ctx_len == max_ctx_len
            and self._cached_batch_key is None
            and self._sent_masks is not None
            and self._feat_acc is not None
            and self._feat_acc.ndim == 2
            and self._feat_acc.shape[0] == self._num_sents
        ):
            self._feat_acc.zero_()
            return

        self._cached_sent_key = sent_key
        self._cached_batch_key = None
        self._cached_num_sents = self._num_sents
        max_ctx_len = self._build_sentence_buffers(sent_positions, context_start, context_end)
        self._cached_ctx_len = max_ctx_len
        if max_ctx_len <= 0 or not sent_positions:
            self._feat_acc = None
            return

        self._feat_acc = torch.zeros(
            self._num_sents,
            self.num_layers * self.num_heads,
            device=self.device,
            dtype=self.dtype,
        )

    def begin_batch(
        self,
        batch_meta: List[dict],
    ) -> None:
        """Batch prefill: each item has sent_positions, context_start, context_end."""
        self.active = True
        self.batch_size = len(batch_meta)
        self.query_pos = None
        self._layer_idx = 0
        self._cached_sent_key = None

        if self._cached_sent_key is not None or (
            self._feat_acc is not None and self._feat_acc.ndim != 3
        ):
            self._cached_sent_key = None
            self._cached_batch_key = None
            self._sent_masks = None
            self._feat_acc = None

        batch_key = tuple(
            (tuple(m["sent_positions"]), m["context_start"], m["context_end"])
            for m in batch_meta
        )
        max_sents = max(len(m["sent_positions"]) for m in batch_meta)
        max_ctx_len = max(
            m["context_end"] - m["context_start"] + 1 for m in batch_meta
        )

        if (
            self._cached_batch_key == batch_key
            and self._sent_masks is not None
            and self._feat_acc is not None
            and self._feat_acc.ndim == 3
            and self._sent_masks.shape == (self.batch_size, max_sents, max_ctx_len)
        ):
            self._feat_acc.zero_()
            self._num_sents = max_sents
            self.context_start = 0
            self.context_end = max_ctx_len - 1
            return

        self._cached_batch_key = batch_key
        self._cached_ctx_len = max_ctx_len
        self._num_sents = max_sents
        self.context_start = 0
        self.context_end = max_ctx_len - 1

        masks = torch.zeros(
            self.batch_size, max_sents, max_ctx_len,
            device=self.device, dtype=self.dtype,
        )
        valid = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        ctx_starts = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        ctx_ends = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        for b, meta in enumerate(batch_meta):
            ctx_start = meta["context_start"]
            ctx_end = meta["context_end"]
            ctx_starts[b] = ctx_start
            ctx_ends[b] = ctx_end
            sent_positions = meta["sent_positions"]
            valid[b] = len(sent_positions)
            if not sent_positions:
                continue
            starts, ends = zip(*sent_positions)
            rel_starts = (
                torch.tensor(starts, dtype=torch.long, device=self.device) - ctx_start
            ).clamp(0, max_ctx_len - 1)
            rel_ends = (
                torch.tensor(ends, dtype=torch.long, device=self.device) - ctx_start
            ).clamp(0, max_ctx_len - 1)
            rel_ends = torch.maximum(rel_ends, rel_starts)
            lengths = (rel_ends - rel_starts + 1).clamp(min=1)
            j = torch.arange(max_ctx_len, device=self.device, dtype=torch.long)
            in_range = (j.unsqueeze(0) >= rel_starts.unsqueeze(1)) & (
                j.unsqueeze(0) <= rel_ends.unsqueeze(1)
            )
            n = len(sent_positions)
            masks[b, :n] = in_range.to(self.dtype) / lengths.unsqueeze(1).clamp(min=1)

        self._sent_masks = masks
        self._valid_sents = valid
        self._batch_ctx_starts = ctx_starts
        self._batch_ctx_ends = ctx_ends
        self._token_sent_id = None
        self._sent_lengths = None
        self._feat_acc = torch.zeros(
            self.batch_size,
            max_sents,
            self.num_layers * self.num_heads,
            device=self.device,
            dtype=self.dtype,
        )

    def record_layer_ratio(self, ratio: torch.Tensor) -> None:
        """Legacy path. ratio: [H, T_ctx]."""
        if not self.active or self._sent_masks is None or self._feat_acc is None:
            return
        ctx_len = ratio.shape[-1]
        sent_masks = self._sent_masks
        if sent_masks.shape[-1] > ctx_len:
            sent_masks = sent_masks[..., :ctx_len]
        elif sent_masks.shape[-1] < ctx_len:
            ratio = ratio[:, : sent_masks.shape[-1]]
        sent_attn = sent_masks @ ratio.T
        self.record_layer_ratio_from_sent_attn(sent_attn)

    def record_layer_ratio_from_sent_attn(self, sent_attn: torch.Tensor) -> None:
        """sent_attn: [S, H] for batch_size=1."""
        if not self.active or self._feat_acc is None:
            return
        h0 = self._layer_idx * self.num_heads
        h1 = h0 + self.num_heads
        self._feat_acc[:, h0:h1] = sent_attn
        self._layer_idx += 1

    def record_layer_ratio_batch(self, sent_attn: torch.Tensor) -> None:
        """sent_attn: [B, S, H]."""
        if not self.active or self._feat_acc is None:
            return
        h0 = self._layer_idx * self.num_heads
        h1 = h0 + self.num_heads
        self._feat_acc[:, :, h0:h1] = sent_attn
        self._layer_idx += 1

    def finalize_vectors(self) -> Optional[torch.Tensor]:
        if not self.active or self._feat_acc is None or self._layer_idx == 0:
            self.clear(active_only=True)
            return None

        vectors = self._feat_acc
        self.clear(active_only=True)
        return vectors

    def finalize_batch_vectors(self) -> Optional[torch.Tensor]:
        """Return [B, S, L*H] sentence feature vectors."""
        return self.finalize_vectors()

    def clear(self, active_only: bool = False) -> None:
        self.active = False
        self.batch_size = 1
        self._layer_idx = 0
        self.context_start = 0
        self.context_end = 0
        self.query_pos = None
        if not active_only:
            self._sent_masks = None
            self._feat_acc = None
            self._num_sents = 0
            self._token_sent_id = None
            self._sent_lengths = None
            self._valid_sents = None
            self._batch_ctx_starts = None
            self._batch_ctx_ends = None
            self._cached_sent_key = None
            self._cached_batch_key = None
