"""Model: modernized-Llama decoder-only transformer (Phase 1).

GQA-native attention, RoPE, RMSNorm, SwiGLU, optional QK-norm, KV cache.
"""

from lithos.model.attention import KVCache
from lithos.model.config import ModelConfig
from lithos.model.generation import generate
from lithos.model.transformer import LithosForCausalLM

__all__ = ["KVCache", "LithosForCausalLM", "ModelConfig", "generate"]
