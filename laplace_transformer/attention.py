"""Laplace-kernel attention and s-domain feature maps.

Design notes
------------
Standard softmax attention computes ``softmax(QK^T / sqrt(d)) V``, which is a
Gibbs distribution over the dot-product scores. The Laplace kernel instead uses

    K(q, k) = exp(-||q - k||_1 / b)

which has sharper, sparser peaks than Gaussian/softmax attention — a property
that is useful for *pattern matching*, where we want tokens to "latch on" to
their best match rather than distribute mass smoothly.

We make two practical choices:

1. **L1 distance in the head dimension.** Cheap, robust to outliers, and
   matches the classical Laplace-distribution likelihood.
2. **Per-head learnable bandwidth ``b``.** Each head can pick its own scale,
   from near-exact match (small ``b``) to broad context (large ``b``).

We also row-normalise the kernel so each query sums its attention weights to 1
— a useful inductive bias that keeps activations well-scaled.

The :class:`LaplaceFeatureMap` converts a length-``T`` sequence into a bank of
s-domain coefficients via a discrete Laplace transform with learnable ``s``
values. It is a lightweight feature layer you can stack before or between
attention blocks for signal-like inputs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class LaplaceAttention(nn.Module):
    """Multi-head attention with a Laplace kernel.

    Attention weights are ``exp(-||q - k||_1 / b)`` (row-normalised), replacing
    the usual softmax over dot products. Causal masking and key-padding masks
    are supported.

    Args:
        d_model: model dimension.
        num_heads: number of attention heads.
        dropout: attention-weight dropout probability.
        causal: if True, apply an upper-triangular mask (no peeking ahead).
        init_bandwidth: initial value for per-head bandwidth ``b``. It is
            parameterised as ``softplus(raw_b)`` so it is always positive.
        bias: include bias terms in Q/K/V/out projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        init_bandwidth: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # b = softplus(raw_b); start at softplus^-1(init_bandwidth).
        inv_sp = math.log(math.expm1(init_bandwidth))
        self.raw_bandwidth = nn.Parameter(torch.full((num_heads,), inv_sp))

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    @property
    def bandwidth(self) -> torch.Tensor:
        """Per-head positive bandwidth ``b``, shape ``(num_heads,)``."""
        return F.softplus(self.raw_bandwidth) + 1e-6

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply Laplace-kernel multi-head self-attention.

        Args:
            x: ``(B, T, d_model)`` input sequence.
            key_padding_mask: optional boolean mask ``(B, T)`` where ``True``
                marks *padded* positions that should be ignored.
            return_weights: if True, also return the attention weights.

        Returns:
            ``(B, T, d_model)`` output, optionally plus ``(B, H, T, T)`` weights.
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # L1 distance between every query and key: (B, H, Tq, Tk)
        # |q_i - k_j| summed over head_dim.
        # Broadcast: q[..., :, None, :] - k[..., None, :, :] is (B,H,Tq,Tk,D).
        # That tensor is O(T^2 D) memory; fine for the modest sequence lengths
        # this lightweight model is designed for.
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        dist = diff.abs().sum(dim=-1)  # (B, H, T, T)

        # Per-head bandwidth, broadcast over (B, T, T).
        b = self.bandwidth.view(1, H, 1, 1)
        logits = -dist / b  # log-unnormalised Laplace weights

        if self.causal:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu_(1)
            logits = logits.masked_fill(causal_mask, float("-inf"))

        if key_padding_mask is not None:
            # True == pad -> mask out
            logits = logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        # Row-normalise; using a softmax over the (already negative) logits is
        # equivalent to exp(logits) / sum(exp(logits)), which is what we want
        # for the normalised Laplace kernel and is numerically stable.
        weights = torch.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)

        out = torch.matmul(weights, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if return_weights:
            return out, weights
        return out


# ---------------------------------------------------------------------------
# Laplace-transform feature map
# ---------------------------------------------------------------------------
class LaplaceFeatureMap(nn.Module):
    r"""Discrete Laplace-transform feature layer.

    For each of ``num_poles`` learnable complex exponents ``s_k = sigma_k + i*omega_k``
    and each input channel, compute

        F_k[c] = sum_{t=0}^{T-1} x[t, c] * exp(-s_k * t)

    then concatenate the real and imaginary parts back into a per-channel
    feature vector. The layer is applied causally along the sequence
    dimension, producing a *running* Laplace transform:

        F_k[t, c] = sum_{u=0}^{t} x[u, c] * exp(-s_k * (t - u))

    This is equivalent to convolution of ``x`` with the causal kernel
    ``exp(-s_k * n)``, and captures exponentially-decaying pattern energy —
    the s-plane analogue of a running Fourier transform. We implement it with
    a first-order IIR recurrence that is O(T * num_poles * d_model).

    It is handy as a cheap feature extractor in front of attention when the
    input has signal-like / quasi-periodic structure.
    """

    def __init__(
        self,
        d_model: int,
        num_poles: int = 8,
        init_sigma: float = 0.1,
        init_omega_max: float = math.pi / 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_poles = num_poles

        # Initialise sigmas at +init_sigma (decaying pole) and omegas spread
        # evenly in [-omega_max, omega_max].
        sigma0 = torch.full((num_poles,), float(init_sigma))
        omega0 = torch.linspace(-init_omega_max, init_omega_max, num_poles)
        self.sigma = nn.Parameter(sigma0)
        self.omega = nn.Parameter(omega0)

        # Mix the 2 * num_poles (real + imag) per-channel features back down
        # to d_model so this layer is a drop-in residual block.
        self.mix = nn.Linear(2 * num_poles * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the running Laplace transform.

        Args:
            x: ``(B, T, d_model)`` input.

        Returns:
            ``(B, T, d_model)`` output of the same shape.
        """
        B, T, C = x.shape
        K = self.num_poles

        # Ensure sigma stays non-negative (for stability of the IIR).
        sigma = F.softplus(self.sigma)  # (K,)
        omega = self.omega              # (K,)

        # Complex pole: a_k = exp(-(sigma_k + i*omega_k))
        decay = torch.exp(-sigma)                        # (K,)
        cos_w = torch.cos(omega)                         # (K,)
        sin_w = torch.sin(omega)                         # (K,)
        a_re = decay * cos_w                             # (K,)
        a_im = -decay * sin_w                            # (K,)

        # Running state: F[t] = x[t] + a * F[t-1]
        f_re = x.new_zeros(B, K, C)
        f_im = x.new_zeros(B, K, C)
        outs = []
        for t in range(T):
            xt = x[:, t, :].unsqueeze(1)  # (B, 1, C)
            new_re = xt + a_re.view(1, K, 1) * f_re - a_im.view(1, K, 1) * f_im
            new_im = a_re.view(1, K, 1) * f_im + a_im.view(1, K, 1) * f_re
            f_re, f_im = new_re, new_im
            outs.append(torch.stack([f_re, f_im], dim=1))  # (B, 2, K, C)

        # (B, T, 2, K, C) -> (B, T, 2*K*C) -> (B, T, d_model)
        feats = torch.stack(outs, dim=1)
        feats = feats.reshape(B, T, 2 * K * C)
        return self.mix(feats)
