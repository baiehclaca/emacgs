"""Laplacian spectral positional encoding.

For a chain (line) graph of length ``T``, the graph Laplacian
``L = D - A`` has eigenvectors

    phi_k[n] = cos( pi * (k + 0.5) * (n + 0.5) / T ),   k = 0, 1, ..., T-1

(up to normalisation), known as the Discrete Cosine Transform basis (DCT-II).
These eigenvectors are the 1D analogue of the graph-Laplacian positional
encodings used in modern graph transformers (e.g. Dwivedi & Bresson, 2020 —
"A Generalization of Transformer Networks to Graphs"), and they have three
properties we want for pattern matching:

1. **Orthogonal** across positions, so they embed unique absolute positions.
2. **Smooth**: low-``k`` modes vary slowly, high-``k`` modes encode fine
   detail. Pattern-matching attention can pick the resolution it needs.
3. **Deterministic & cheap**: no learned parameters required (though we
   optionally apply a learned linear projection so the model can re-weight
   modes).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def laplacian_eigenvectors(seq_len: int, num_modes: int) -> torch.Tensor:
    """Return the first ``num_modes`` eigenvectors of the chain-graph Laplacian.

    These are DCT-II basis vectors, orthonormal across positions:

        phi_k[n] = sqrt(2/T) * cos( pi * (k + 1) * (n + 0.5) / T ),   k >= 1
        phi_0[n] = sqrt(1/T)

    The k=0 mode is constant, so we skip it by default and start at k=1; this
    matches the convention in graph transformers where the zero eigenvalue
    (all-ones eigenvector) carries no positional information.

    Args:
        seq_len: sequence length ``T``.
        num_modes: number of eigenvectors to return.

    Returns:
        Tensor of shape ``(seq_len, num_modes)``.
    """
    T = seq_len
    n = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    k = torch.arange(1, num_modes + 1, dtype=torch.float32).unsqueeze(0)  # (1, M)
    phi = math.sqrt(2.0 / T) * torch.cos(math.pi * k * (n + 0.5) / T)
    return phi  # (T, M)


class LaplacianPositional(nn.Module):
    """Spectral positional encoding from the chain-graph Laplacian.

    At construction time we compute an eigenvector cache up to ``max_len``;
    at forward time we slice the first ``T`` rows and project up to the
    embedding dimension with a learned linear layer.

    Args:
        d_model: embedding dimension to produce.
        max_len: maximum sequence length for the precomputed cache.
        num_modes: number of Laplacian eigenvectors to use.
        learnable: if True, add a learned per-mode scale on top of the fixed
            eigenvectors (gives the model a gentle knob over mode weights
            while preserving orthogonality across positions).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        num_modes: int = 64,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if num_modes > max_len:
            raise ValueError("num_modes must be <= max_len")
        self.d_model = d_model
        self.max_len = max_len
        self.num_modes = num_modes

        phi = laplacian_eigenvectors(max_len, num_modes)  # (max_len, num_modes)
        # Cache as a non-trainable buffer so .to(device) moves it automatically.
        self.register_buffer("eigvecs", phi, persistent=False)

        self.scale = (
            nn.Parameter(torch.ones(num_modes)) if learnable else None
        )
        self.proj = nn.Linear(num_modes, d_model, bias=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return a ``(1, seq_len, d_model)`` positional embedding."""
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_len ({self.max_len})"
            )
        phi = self.eigvecs[:seq_len]  # (T, num_modes)
        if self.scale is not None:
            phi = phi * self.scale
        return self.proj(phi).unsqueeze(0)  # (1, T, d_model)
