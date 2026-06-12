"""Fused last-row QK + context renorm + sentence mean (Triton block GEMM or torch).

Triton path (two kernels, batch=1 only):
  1. ``_last_row_qk_kernel`` — block GEMM ``q[head] @ K[kv_head, s:s+BLOCK_S].T``
     grid ``(num_heads, ceil(seq_len / BLOCK_S))``, no per-token Python loop.
  2. ``_probe_sentmean_kernel`` — blocked softmax max + context renorm + sentence
     mean via ``token_sent_id`` scatter into scratch, grid ``(num_heads,)``.

Torch path (default): cuBLAS ``matmul`` for QK + ``sent_masks @ ctx.T``; still
~15–25%% faster than Triton on ~5k tokens (A800, Qwen2.5-0.5B) because probe QK
reuses highly tuned GEMM and Triton adds extra kernel launches × 24 layers.
Enable Triton with ``use_triton_probe=True`` for experimentation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from probe.state import ProbeState

_TRITON_OK = False
try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore

# Block sizes tuned for Qwen2.5 (head_dim=64/128, seq up to ~32k)
_BLOCK_S = 128
_BLOCK_D = 64


def triton_probe_available() -> bool:
    return _TRITON_OK


def _repeat_kv(key_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return key_states
    batch, num_kv, slen, dim = key_states.shape
    key_states = key_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, dim)
    return key_states.reshape(batch, num_kv * n_rep, slen, dim)


def _apply_attn_mask(scores: torch.Tensor, attention_mask: Optional[torch.Tensor], q_idx: int) -> torch.Tensor:
    if attention_mask is None:
        return scores
    seq_k = scores.shape[-1]
    if attention_mask.dim() == 4:
        return scores + attention_mask[:, :, q_idx : q_idx + 1, :seq_k]
    if attention_mask.dim() == 2:
        valid = attention_mask[:, :seq_k].unsqueeze(1).unsqueeze(1).to(scores.dtype)
        return scores + (1.0 - valid) * -1.0e4
    return scores


def _compute_sent_features_torch(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    probe_state: ProbeState,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Sentence features [B, S, H] — fused ratio + segment mean."""
    key_states = _repeat_kv(key_states, num_key_value_groups)
    q_idx = query_states.shape[2] - 1 if probe_state.query_pos is None else probe_state.query_pos
    q_last = query_states[:, :, q_idx : q_idx + 1, :]
    scores = torch.matmul(q_last, key_states.transpose(2, 3)) * scaling
    scores = _apply_attn_mask(scores, attention_mask, q_idx)

    probs = F.softmax(scores.float(), dim=-1).to(query_states.dtype)

    sent_masks = probe_state._sent_masks
    if sent_masks is None:
        raise RuntimeError("probe_state._sent_masks is None")

    # begin() → [S, T]; begin_batch() → [B, S, T] (including B=1)
    if sent_masks.dim() == 2:
        ctx_end = min(probe_state.context_end, probs.shape[-1] - 1)
        ctx = probs[..., probe_state.context_start : ctx_end + 1]
        ctx = ctx / ctx.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        ctx = torch.nan_to_num(ctx, 0.0).squeeze(2)  # [1, H, T_ctx]
        t_ctx = ctx.shape[-1]
        masks = sent_masks if sent_masks.shape[1] >= t_ctx else sent_masks[:, :t_ctx]
        ctx_h = ctx[:, :, : masks.shape[1]]
        return (masks @ ctx_h.squeeze(0).T).unsqueeze(0)  # [1, S, H]

    assert probe_state._batch_ctx_starts is not None
    assert probe_state._batch_ctx_ends is not None
    assert probe_state._valid_sents is not None

    probs_bh = probs.squeeze(2)  # [B, H, seq_len]
    batch_size = probe_state.batch_size
    num_heads = probs_bh.shape[1]
    seq_len = probs_bh.shape[2]
    max_ctx = sent_masks.shape[2]
    device = probs_bh.device
    dtype = probs_bh.dtype

    cs = probe_state._batch_ctx_starts
    ce = probe_state._batch_ctx_ends.clamp(max=seq_len - 1)
    lengths = (ce - cs + 1).clamp(min=1)  # [B]

    rel_idx = torch.arange(max_ctx, device=device, dtype=torch.long).unsqueeze(0)
    abs_idx = cs.unsqueeze(1) + rel_idx  # [B, max_ctx]
    abs_idx = abs_idx.clamp(max=seq_len - 1)
    valid = rel_idx < lengths.unsqueeze(1)  # [B, max_ctx]

    gathered = torch.gather(
        probs_bh,
        2,
        abs_idx.unsqueeze(1).expand(batch_size, num_heads, max_ctx),
    )
    gathered = gathered * valid.unsqueeze(1).to(dtype)
    ctx_sum = gathered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    ctx = torch.nan_to_num(gathered / ctx_sum, 0.0)  # [B, H, max_ctx]

    # [B, S, max_ctx] @ [B, max_ctx, H] -> [B, S, H]
    return torch.bmm(sent_masks, ctx.transpose(1, 2))


if _TRITON_OK:

    @triton.jit
    def _last_row_qk_kernel(
        q_ptr,
        k_ptr,
        scores_ptr,
        num_kv_groups,
        seq_len,
        head_dim,
        scale,
        stride_kh,
        stride_ks,
        stride_kd,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Block GEMM: q[head, :] @ k[kv_head, s:s+BLOCK_S, :].T -> scores[head, s:s+BLOCK_S]."""
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)
        kv_h = pid_h // num_kv_groups

        offs_s = pid_m * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < seq_len

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < head_dim
            q = tl.load(q_ptr + pid_h * head_dim + offs_d, mask=mask_d, other=0.0)
            k = tl.load(
                k_ptr
                + kv_h * stride_kh
                + offs_s[:, None] * stride_ks
                + offs_d[None, :] * stride_kd,
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc += tl.sum(k * q[None, :], axis=1)

        acc = acc * scale
        tl.store(scores_ptr + pid_h * seq_len + offs_s, acc, mask=mask_s)

    @triton.jit
    def _probe_sentmean_kernel(
        scores_ptr,
        scratch_ptr,
        out_ptr,
        token_sent_ptr,
        sent_lens_ptr,
        num_sents,
        seq_len,
        ctx_start,
        ctx_end,
        BLOCK_S: tl.constexpr,
    ):
        """Blocked softmax + context renorm + sentence mean from precomputed scores."""
        head_id = tl.program_id(0)
        row = scores_ptr + head_id * seq_len

        max_val = -1.0e9
        for start in range(0, seq_len, BLOCK_S):
            offs = start + tl.arange(0, BLOCK_S)
            m = offs < seq_len
            v = tl.load(row + offs, mask=m, other=-1.0e9)
            max_val = tl.maximum(max_val, tl.max(v))

        ctx_sum = 0.0
        for start in range(0, seq_len, BLOCK_S):
            offs = start + tl.arange(0, BLOCK_S)
            m = offs < seq_len
            v = tl.load(row + offs, mask=m, other=-1.0e9)
            e = tl.exp(v - max_val)
            in_ctx = m & (offs >= ctx_start) & (offs <= ctx_end)
            ctx_sum += tl.sum(tl.where(in_ctx, e, 0.0))

            for i in tl.static_range(BLOCK_S):
                pos = start + i
                valid = (pos < seq_len) & (pos >= ctx_start) & (pos <= ctx_end)
                sc_i = tl.load(row + pos, mask=pos < seq_len, other=-1.0e9)
                e_i = tl.exp(sc_i - max_val)
                t_sent = tl.load(token_sent_ptr + pos - ctx_start, mask=valid, other=-1)
                t_ok = valid & (t_sent >= 0)
                idx = head_id * num_sents + t_sent
                prev = tl.load(scratch_ptr + idx, mask=t_ok, other=0.0)
                tl.store(scratch_ptr + idx, prev + tl.where(t_ok, e_i, 0.0), mask=t_ok)

        ctx_sum = tl.maximum(ctx_sum, 1.0e-8)
        for s_idx in range(num_sents):
            acc = tl.load(scratch_ptr + head_id * num_sents + s_idx)
            length = tl.maximum(tl.load(sent_lens_ptr + s_idx), 1.0)
            tl.store(out_ptr + s_idx * tl.num_programs(0) + head_id, acc / length / ctx_sum)


_triton_scores: Optional[torch.Tensor] = None
_triton_scores_shape: Optional[Tuple[int, int]] = None
_triton_scratch: Optional[torch.Tensor] = None
_triton_scratch_shape: Optional[Tuple[int, int]] = None


def _triton_fused_single(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    probe_state: ProbeState,
    num_key_value_groups: int,
    scaling: float,
) -> None:
    global _triton_scores, _triton_scores_shape, _triton_scratch, _triton_scratch_shape

    num_heads = query_states.shape[1]
    num_sents = probe_state._num_sents
    seq_len = key_states.shape[2]
    head_dim = query_states.shape[3]
    q_idx = query_states.shape[2] - 1 if probe_state.query_pos is None else probe_state.query_pos

    q_last = query_states[0, :, q_idx, :].contiguous().float()
    k = key_states[0].contiguous().float()
    ctx_start = probe_state.context_start
    ctx_end = min(probe_state.context_end, seq_len - 1)

    scores_shape = (num_heads, seq_len)
    if _triton_scores_shape != scores_shape or _triton_scores is None:
        _triton_scores = torch.empty(scores_shape, device=q_last.device, dtype=torch.float32)
        _triton_scores_shape = scores_shape

    scratch_shape = (num_heads, num_sents)
    if _triton_scratch_shape != scratch_shape or _triton_scratch is None:
        _triton_scratch = torch.zeros(scratch_shape, device=q_last.device, dtype=torch.float32)
        _triton_scratch_shape = scratch_shape
    else:
        _triton_scratch.zero_()

    block_d = triton.next_power_of_2(max(head_dim, 16))
    grid_qk = (num_heads, triton.cdiv(seq_len, _BLOCK_S))
    _last_row_qk_kernel[grid_qk](
        q_last,
        k,
        _triton_scores,
        num_key_value_groups,
        seq_len,
        head_dim,
        scaling,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        BLOCK_S=_BLOCK_S,
        BLOCK_D=block_d,
    )

    out = torch.empty(num_sents, num_heads, device=q_last.device, dtype=torch.float32)
    _probe_sentmean_kernel[(num_heads,)](
        _triton_scores,
        _triton_scratch,
        out,
        probe_state._token_sent_id,
        probe_state._sent_lengths.to(torch.float32),
        num_sents,
        seq_len,
        ctx_start,
        ctx_end,
        BLOCK_S=_BLOCK_S,
    )
    probe_state.record_layer_ratio_from_sent_attn(out.to(probe_state.dtype))


def fused_probe_layer(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    probe_state: ProbeState,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """Probe side-channel: last-row QK → context renorm → sentence mean."""
    if not probe_state.active or probe_state._feat_acc is None:
        return

    use_triton = (
        _TRITON_OK
        and probe_state.use_triton
        and probe_state.batch_size == 1
        and attention_mask is None
        and probe_state._token_sent_id is not None
    )

    if use_triton:
        _triton_fused_single(
            query_states, key_states, probe_state,
            num_key_value_groups, scaling,
        )
        return

    sent_attn = _compute_sent_features_torch(
        query_states, key_states, probe_state,
        num_key_value_groups, scaling, attention_mask,
    )
    if probe_state._sent_masks is not None and probe_state._sent_masks.dim() == 3:
        probe_state.record_layer_ratio_batch(sent_attn)
    else:
        probe_state.record_layer_ratio_from_sent_attn(sent_attn.squeeze(0))
