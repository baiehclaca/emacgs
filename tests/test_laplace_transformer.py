"""Smoke tests for the Laplace Transformer package."""

from __future__ import annotations

import math

import pytest
import torch

from laplace_transformer import (
    CharTokenizer,
    LaplaceAttention,
    LaplaceConfig,
    LaplaceFeatureMap,
    LaplacianPositional,
    LaplaceTransformer,
)
from laplace_transformer.positional import laplacian_eigenvectors


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def test_laplace_attention_shapes_and_normalisation():
    torch.manual_seed(0)
    attn = LaplaceAttention(d_model=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 7, 16)
    out, w = attn(x, return_weights=True)
    assert out.shape == (2, 7, 16)
    assert w.shape == (2, 4, 7, 7)
    # Each query's attention weights sum to 1.
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_laplace_attention_is_causal_when_requested():
    torch.manual_seed(0)
    attn = LaplaceAttention(d_model=16, num_heads=2, dropout=0.0, causal=True)
    x = torch.randn(1, 5, 16)
    _, w = attn(x, return_weights=True)
    # Upper-triangular (strict) entries must be zero.
    mask = torch.ones(5, 5, dtype=torch.bool).triu_(1)
    assert torch.all(w[..., mask] == 0)


def test_laplace_attention_bandwidth_is_positive():
    attn = LaplaceAttention(d_model=8, num_heads=2, init_bandwidth=0.5)
    # Force raw_bandwidth very negative — softplus should still give > 0.
    with torch.no_grad():
        attn.raw_bandwidth.fill_(-10.0)
    assert torch.all(attn.bandwidth > 0)


def test_laplace_attention_key_padding_mask():
    torch.manual_seed(0)
    attn = LaplaceAttention(d_model=8, num_heads=2, dropout=0.0)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[False, False, True, True]])  # last two are padding
    _, w = attn(x, key_padding_mask=mask, return_weights=True)
    assert torch.all(w[..., 2:] == 0)


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------
def test_feature_map_preserves_shape():
    fm = LaplaceFeatureMap(d_model=8, num_poles=4)
    x = torch.randn(2, 6, 8)
    y = fm(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
def test_laplacian_eigenvectors_orthonormal():
    phi = laplacian_eigenvectors(seq_len=32, num_modes=16)
    gram = phi.T @ phi  # (16, 16)
    assert torch.allclose(gram, torch.eye(16), atol=1e-5)


def test_positional_forward_shapes():
    pe = LaplacianPositional(d_model=24, max_len=128, num_modes=16)
    out = pe(10)
    assert out.shape == (1, 10, 24)


def test_positional_rejects_overlong_sequence():
    pe = LaplacianPositional(d_model=8, max_len=8, num_modes=4)
    with pytest.raises(ValueError):
        pe(16)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def _tiny_cfg(**overrides) -> LaplaceConfig:
    base = dict(
        vocab_size=20,
        max_len=24,
        d_model=16,
        num_heads=4,
        num_layers=2,
        ffn_mult=2,
        dropout=0.0,
        pos_num_modes=8,
        use_feature_map=False,
    )
    base.update(overrides)
    return LaplaceConfig(**base)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = LaplaceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 10))
    y = torch.randint(0, cfg.vocab_size, (3, 10))
    out = model(x, targets=y)
    assert out["logits"].shape == (3, 10, cfg.vocab_size)
    assert out["loss"].ndim == 0
    assert math.isfinite(out["loss"].item())


def test_forward_without_targets_has_none_loss():
    cfg = _tiny_cfg()
    model = LaplaceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model(x)
    assert out["loss"] is None


def test_generate_extends_sequence():
    cfg = _tiny_cfg()
    model = LaplaceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    y = model.generate(x, max_new_tokens=6)
    assert y.shape == (1, 10)


def test_feature_map_block_still_runs():
    cfg = _tiny_cfg(use_feature_map=True, feature_map_poles=4)
    model = LaplaceTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(x)
    assert out["logits"].shape == (1, 8, cfg.vocab_size)


def test_match_pattern_score_shape():
    cfg = _tiny_cfg()
    model = LaplaceTransformer(cfg)
    seq = torch.randint(0, cfg.vocab_size, (10,))
    pat = torch.randint(0, cfg.vocab_size, (3,))
    scores = model.match_pattern(seq, pat)
    assert scores.shape == (10,)


def test_tied_embeddings_share_storage():
    cfg = _tiny_cfg(tie_embeddings=True)
    model = LaplaceTransformer(cfg)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"


def test_bytes_tokenizer_roundtrip():
    tok = CharTokenizer.bytes_tokenizer()
    text = "héllo 🌊"
    assert tok.decode(tok.encode(text)) == text


def test_pattern_matching_learns_copy():
    """End-to-end: a tiny model should drop loss on the copy task."""
    torch.manual_seed(0)
    alphabet = "abcd"
    sep = "|"
    tok = CharTokenizer.from_text(alphabet + sep)
    pat_len = 4
    cfg = LaplaceConfig(
        vocab_size=tok.vocab_size,
        max_len=2 * pat_len + 2,
        d_model=32,
        num_heads=4,
        num_layers=2,
        pos_num_modes=8,
        dropout=0.0,
        use_feature_map=False,
    )
    model = LaplaceTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    def batch():
        import random
        xs, ys = [], []
        for _ in range(32):
            p = "".join(random.choice(alphabet) for _ in range(pat_len))
            ids = tok.encode(p + sep + p)
            xs.append(ids[:-1])
            y = ids[1:]
            y[:pat_len] = [-100] * pat_len  # supervise only the copy
            ys.append(y)
        return torch.tensor(xs), torch.tensor(ys)

    import random
    random.seed(0)
    initial = None
    final = None
    for step in range(120):
        x, y = batch()
        out = model(x, targets=y)
        if step == 0:
            initial = out["loss"].item()
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        final = out["loss"].item()

    assert final < initial - 0.3, f"loss did not drop enough: {initial:.3f} -> {final:.3f}"
