"""Last-row attention probe for Qwen2 (SDPA + side-channel)."""

from probe.state import ProbeState
from probe.qwen2_probe import patch_qwen2_attention_for_probe, unpatch_qwen2_attention_probe

__all__ = ["ProbeState", "patch_qwen2_attention_for_probe", "unpatch_qwen2_attention_probe"]
