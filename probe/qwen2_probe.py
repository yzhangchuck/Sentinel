"""Last-row attention probe for Qwen2 models (SDPA prefill + side-channel features)."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from probe.kernels.fused_probe import fused_probe_layer
from probe.state import ProbeState

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        eager_attention_forward,
        repeat_kv,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    Qwen2Attention = None  # type: ignore
    apply_rotary_pos_emb = None  # type: ignore
    eager_attention_forward = None  # type: ignore
    repeat_kv = None  # type: ignore
    ALL_ATTENTION_FUNCTIONS = {}  # type: ignore


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if repeat_kv is not None:
        return repeat_kv(hidden_states, n_rep)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def compute_last_row_context_ratio(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    num_key_value_groups: int,
    attention_mask: Optional[torch.Tensor],
    context_start: int,
    context_end: int,
    scaling: float,
    query_pos: Optional[int] = None,
) -> torch.Tensor:
    """
    Match training/inference convention:
      full-sequence softmax(last query row) → slice context → renorm within context.

    Returns:
        ratio: [B, H, T_ctx]
    """
    key_states = _repeat_kv(key_states, num_key_value_groups)
    q_idx = query_states.shape[2] - 1 if query_pos is None else query_pos
    q_last = query_states[:, :, q_idx : q_idx + 1, :]
    scores = torch.matmul(q_last, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        seq_k = key_states.shape[-2]
        mask = attention_mask[:, :, q_idx : q_idx + 1, :seq_k]
        scores = scores + mask

    probs = F.softmax(scores.float(), dim=-1).to(query_states.dtype)
    ctx_end = min(context_end, probs.shape[-1] - 1)
    ctx = probs[..., context_start : ctx_end + 1]
    ratio = ctx / ctx.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.nan_to_num(ratio, 0.0).squeeze(2)  # [B, H, T_ctx]


def _make_probed_forward(probe_state: ProbeState) -> Callable:
    """Return a Qwen2Attention.forward replacement (transformers >= 4.45 API)."""

    def probed_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if probe_state.active:
            fused_probe_layer(
                query_states,
                key_states,
                probe_state,
                self.num_key_value_groups,
                self.scaling,
                attention_mask,
            )

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if not (
                self.config._attn_implementation == "sdpa"
                and kwargs.get("output_attentions", False)
            ):
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_kwargs = dict(kwargs)
        attn_kwargs.pop("attention_mask", None)
        attn_kwargs["output_attentions"] = False

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **attn_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return probed_forward


def patch_qwen2_attention_for_probe(
    model: torch.nn.Module,
    probe_state: ProbeState,
) -> int:
    """Patch all Qwen2Attention modules in-place. Returns layer count."""
    if Qwen2Attention is None:
        raise RuntimeError("transformers Qwen2Attention not available")

    patched = 0
    originals: List[Tuple[torch.nn.Module, Callable]] = []
    probed = _make_probed_forward(probe_state)

    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            originals.append((module, module.forward))
            module.forward = probed.__get__(module, module.__class__)
            patched += 1

    model._probe_original_forwards = originals  # type: ignore[attr-defined]
    return patched


def unpatch_qwen2_attention_probe(model: torch.nn.Module) -> None:
    originals = getattr(model, "_probe_original_forwards", None)
    if not originals:
        return
    for module, orig in originals:
        module.forward = orig
    del model._probe_original_forwards
