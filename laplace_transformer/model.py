"""LaplaceTransformer: a tiny, pre-norm transformer with Laplace attention.

The overall architecture follows the nanoGPT / x-transformers shape:

    token_emb -> (+ positional_emb) -> N x LaplaceBlock -> LayerNorm -> head

Each ``LaplaceBlock`` is pre-norm:

    x = x + Attn(LN(x))
    x = x + MLP(LN(x))

with the attention replaced by :class:`LaplaceAttention` and — optionally —
an s-domain :class:`LaplaceFeatureMap` that runs in parallel to the MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LaplaceAttention, LaplaceFeatureMap
from .positional import LaplacianPositional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LaplaceConfig:
    """Hyperparameters for :class:`LaplaceTransformer`."""

    vocab_size: int = 256
    max_len: int = 512
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    causal: bool = True
    use_feature_map: bool = True
    feature_map_poles: int = 8
    pos_num_modes: int = 64
    init_bandwidth: float = 1.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.pos_num_modes > self.max_len:
            raise ValueError("pos_num_modes must be <= max_len")


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class LaplaceBlock(nn.Module):
    """Pre-norm transformer block with Laplace attention and optional s-domain mixer."""

    def __init__(self, cfg: LaplaceConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LaplaceAttention(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            causal=cfg.causal,
            init_bandwidth=cfg.init_bandwidth,
        )

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        if cfg.use_feature_map:
            self.ln3 = nn.LayerNorm(cfg.d_model)
            self.feature_map = LaplaceFeatureMap(
                d_model=cfg.d_model, num_poles=cfg.feature_map_poles
            )
        else:
            self.ln3 = None
            self.feature_map = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        if self.feature_map is not None:
            x = x + self.feature_map(self.ln3(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LaplaceTransformer(nn.Module):
    """A tiny stackable Laplace transformer / causal language model."""

    def __init__(self, cfg: LaplaceConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = LaplacianPositional(
            d_model=cfg.d_model,
            max_len=cfg.max_len,
            num_modes=cfg.pos_num_modes,
        )
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [LaplaceBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: ``(B, T)`` long tensor of token IDs.
            targets: optional ``(B, T)`` long tensor for LM training. Positions
                set to ``-100`` are ignored in the loss.
            key_padding_mask: optional ``(B, T)`` boolean mask, ``True`` for
                padded positions.

        Returns:
            ``{"logits": (B, T, V), "loss": scalar | None}``.
        """
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(
                f"sequence length {T} exceeds max_len {self.cfg.max_len}"
            )

        h = self.tok_emb(input_ids) + self.pos_enc(T)
        h = self.emb_dropout(h)
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    # ------------------------------------------------------------------ #
    # Generation / pattern matching utilities                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy / top-k sampling (for quick demos; not the point of the model)."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_len:]
            logits = self(ctx)["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    @torch.no_grad()
    def match_pattern(
        self,
        sequence: torch.Tensor,
        pattern: torch.Tensor,
        layer: int = -1,
        head: Optional[int] = None,
    ) -> torch.Tensor:
        """Score how well each position in ``sequence`` matches ``pattern``.

        We embed the concatenation ``[sequence, pattern]`` and then, at the
        chosen attention layer, look at how strongly the *pattern* tokens
        attend to each position in the *sequence*. Summing over pattern tokens
        gives a per-position match score — this is the natural pattern-matching
        read-out from a Laplace-kernel attention map.

        Args:
            sequence: ``(T_s,)`` long tensor of token IDs.
            pattern: ``(T_p,)`` long tensor of token IDs.
            layer: which block's attention to use (``-1`` = last).
            head: which head to use, or ``None`` to average over heads.

        Returns:
            ``(T_s,)`` float tensor of match scores (higher = better match).
        """
        self.eval()
        device = next(self.parameters()).device
        seq = sequence.to(device)
        pat = pattern.to(device)
        T_s, T_p = seq.size(0), pat.size(0)
        if T_s + T_p > self.cfg.max_len:
            raise ValueError("sequence + pattern exceed max_len")

        ids = torch.cat([seq, pat]).unsqueeze(0)  # (1, T_s + T_p)

        h = self.tok_emb(ids) + self.pos_enc(ids.size(1))
        block_idx = layer if layer >= 0 else len(self.blocks) + layer
        for i, block in enumerate(self.blocks):
            if i < block_idx:
                h = block(h)
            else:
                break

        # Inspect the attention weights of the target block.
        target = self.blocks[block_idx]
        normed = target.ln1(h)
        _, weights = target.attn(normed, return_weights=True)
        # weights: (1, H, T, T). Pattern rows are the last T_p queries.
        pat_weights = weights[0, :, T_s:, :T_s]  # (H, T_p, T_s)
        if head is None:
            scores = pat_weights.mean(dim=0).sum(dim=0)
        else:
            scores = pat_weights[head].sum(dim=0)
        return scores.cpu()
