"""Full decoder-only model (PRD §6.1).

Embedding -> N pre-norm transformer blocks -> final RMSNorm -> output head, with
tied/untied embeddings, vocab padding masked out of the loss/logits, and
GPT-2/Llama-style depth-scaled initialization for residual projections.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lithos.model.attention import KVCache
from lithos.model.config import ModelConfig
from lithos.model.layers import TransformerBlock
from lithos.model.norm import RMSNorm
from lithos.model.rope import RotaryEmbedding


class LithosForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        vocab = cfg.padded_vocab_size

        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(vocab, cfg.hidden)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden, vocab, bias=False)

        self.apply(self._init_weights)
        # Depth-scaled init for residual output projections (GPT-2/Llama style).
        residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = input_ids.shape
        past = kv_caches[0].length if kv_caches is not None else 0
        positions = torch.arange(past, past + t, device=input_ids.device)
        cos, sin = self.rope(positions)

        # Explicit causal mask only needed when there is cached context; otherwise
        # the fast (square) causal path is used inside attention.
        attn_mask: torch.Tensor | None = None
        if past > 0:
            k_pos = torch.arange(past + t, device=input_ids.device)
            attn_mask = k_pos[None, :] <= positions[:, None]  # (T, past+T) bool

        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and kv is None:
                x = checkpoint(layer, x, cos, sin, attn_mask, kv, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attn_mask, kv)
        x = self.norm(x)
        logits = self.lm_head(x)

        # Padding vocab columns must never be predicted or contribute to the loss.
        if self.cfg.padded_vocab_size > self.cfg.vocab_size:
            pad_cols = (
                torch.arange(self.cfg.padded_vocab_size, device=logits.device)
                >= self.cfg.vocab_size
            )
            logits = logits.masked_fill(pad_cols, torch.finfo(logits.dtype).min)

        # ``targets`` are already aligned with ``logits`` (the dataloader yields
        # pre-shifted (x, y) windows), so no internal shift is applied here.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    def init_kv_caches(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.cfg.n_layers)]

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n
