# Laplace Transformer

A **lightweight transformer for pattern matching** built around the Laplace
kernel. It is small enough to read end-to-end (≈500 lines of PyTorch), trains
on CPU, and still makes three concrete architectural choices that set it apart
from a vanilla transformer:

1. **Laplace-kernel attention** — attention weights are
   `exp(-‖q − k‖₁ / b)` (row-normalised) instead of `softmax(qᵀk/√d)`.
   The Laplace kernel is *sharper and sparser* than Gaussian/softmax
   attention, which is exactly what you want for pattern matching: a query
   latches onto its closest key rather than spreading mass over all
   context. Each head learns its own positive bandwidth `b` via a
   softplus reparameterisation.
2. **Laplacian spectral positional encoding** — the positional embedding is
   a learned projection of the first `M` eigenvectors of the chain-graph
   Laplacian (the DCT-II basis). This mirrors the graph-Laplacian PEs used
   in modern graph transformers (Dwivedi & Bresson 2020) and gives the model
   smooth, orthogonal, multi-resolution position features.
3. **Optional s-domain feature map** — a
   [Laplace transform](https://en.wikipedia.org/wiki/Laplace_transform) layer
   with learnable complex poles that produces a running s-plane
   representation of the sequence. It is implemented as a first-order IIR
   recurrence and runs in parallel to each block's MLP; useful for
   signal-like inputs.

The overall scaffolding (pre-norm blocks, tied embeddings, tiny `generate`,
character tokenizer, minimal training loop) follows the style of
[nanoGPT](https://github.com/karpathy/nanoGPT),
[minGPT](https://github.com/karpathy/minGPT), and
[x-transformers](https://github.com/lucidrains/x-transformers) — the
state-of-the-art open-source references for readable transformer code.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .
```

## Quick start

Train a tiny model to copy random patterns (the classic "induction heads" task):

```bash
python examples/pattern_matching.py --steps 400 --pattern-len 8
```

On CPU this drops the copy-loss from ~2.7 to <0.01 in a few seconds and
produces perfect completions:

```
step    1  loss 2.7693
step   50  loss 0.0255
step  400  loss 0.0009

--- copy completions ---
  target=ihbgfcfa  got=ihbgfcfa
  target=gihdidaj  got=gihdidaj
```

Train a tiny character LM on any text file:

```bash
python examples/train.py --data path/to/text.txt --steps 2000 --prompt "The "
```

## Using the library

```python
import torch
from laplace_transformer import LaplaceConfig, LaplaceTransformer, CharTokenizer

tok = CharTokenizer.from_text("abcdefghij|")
cfg = LaplaceConfig(
    vocab_size=tok.vocab_size,
    max_len=64,
    d_model=128,
    num_heads=4,
    num_layers=4,
    use_feature_map=True,   # turn on the s-domain mixer
)
model = LaplaceTransformer(cfg)

ids = tok.encode_tensor("abc|").unsqueeze(0)
logits = model(ids)["logits"]                    # (1, 4, vocab_size)
gen = model.generate(ids, max_new_tokens=16)     # greedy / top-k sampling
```

### Pattern-matching read-out

Given a trained (or untrained) model, you can score how strongly each
position of a *haystack* sequence matches a *needle* pattern using the
attention map directly:

```python
scores = model.match_pattern(
    sequence=tok.encode_tensor("xxxabcdefxxxabcabc"),
    pattern=tok.encode_tensor("abc"),
)
# scores: (18,) — high at positions where the needle matches the haystack
```

Internally this concatenates `[haystack, needle]`, runs the model up to the
chosen attention layer, and reads off how much probability mass the needle
tokens put on each haystack position. Because Laplace-kernel attention is
sharp, these scores localise well even before training.

## Web demo (try it on your phone)

A zero-dependency web demo ships in `web/`. It trains a tiny copy-task
model on startup (~10 seconds on CPU) and serves two endpoints backed by
Python's stdlib `http.server`:

- `POST /api/match` — returns per-position Laplace-attention scores for
  a `{ haystack, needle }` pair.
- `POST /api/copy` — asks the trained model to copy any short pattern.

Run it:

```bash
python web/server.py --port 8000
```

The server binds to `0.0.0.0` and prints candidate URLs on startup,
including a LAN URL like `http://192.168.x.y:8000/`. Open that URL on
your phone (same Wi-Fi network) to try it on-device.

## Files

```
laplace_transformer/
├── attention.py   # LaplaceAttention + LaplaceFeatureMap
├── positional.py  # Laplacian spectral positional encoding
├── model.py       # LaplaceBlock, LaplaceTransformer, LaplaceConfig
└── tokenizer.py   # CharTokenizer
examples/
├── pattern_matching.py  # induction-heads / copy-task demo
└── train.py             # minimal char-LM training loop
web/
├── server.py            # stdlib HTTP server (match + copy APIs)
├── index.html           # mobile-first UI
├── style.css            # dark, touch-friendly styles
└── app.js               # small vanilla-JS frontend
tests/
└── test_laplace_transformer.py
```

## Tests

```bash
pip install pytest
pytest -q
```

17 tests cover attention shape + normalisation, causal masking,
key-padding masks, bandwidth positivity, Laplacian PE orthonormality,
model forward / generate / `match_pattern` shapes, tokenizer round-trips,
and an end-to-end learn-the-copy-task smoke test.

## Inspirations and references

- **nanoGPT** ([karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)) and
  **minGPT** ([karpathy/minGPT](https://github.com/karpathy/minGPT)) — the
  canonical "small readable transformer" codebases; the block / training
  loop shape borrows heavily from these.
- **x-transformers**
  ([lucidrains/x-transformers](https://github.com/lucidrains/x-transformers))
  — a reference for modular attention variants.
- **Graph-Laplacian positional encodings** — Dwivedi & Bresson, *A
  Generalization of Transformer Networks to Graphs* (2020).
- **Laplace kernel attention** — the Laplace / exponential kernel is a
  long-standing alternative to the RBF / softmax kernel in kernel methods;
  using it inside attention follows the broader "kernelised attention"
  line of work (Performer, cosFormer, etc.).

## License

MIT.
