"""Laplace Transformer: a lightweight pattern-matching transformer.

The package exposes:
    - :class:`LaplaceAttention`      — Laplace-kernel multi-head attention.
    - :class:`LaplaceFeatureMap`     — small s-domain (Laplace transform) feature layer.
    - :class:`LaplacianPositional`   — spectral (graph-Laplacian) positional encoding.
    - :class:`LaplaceBlock`          — pre-norm transformer block using Laplace attention.
    - :class:`LaplaceTransformer`    — stackable transformer / language model.
    - :class:`CharTokenizer`         — tiny byte-level tokenizer for quick demos.
"""

from .attention import LaplaceAttention, LaplaceFeatureMap
from .positional import LaplacianPositional
from .model import LaplaceBlock, LaplaceTransformer, LaplaceConfig
from .tokenizer import CharTokenizer

__all__ = [
    "LaplaceAttention",
    "LaplaceFeatureMap",
    "LaplacianPositional",
    "LaplaceBlock",
    "LaplaceTransformer",
    "LaplaceConfig",
    "CharTokenizer",
]

__version__ = "0.1.0"
