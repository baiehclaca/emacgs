"""Pattern-matching demo.

Train a tiny LaplaceTransformer to *copy* a short pattern after it has been
shown once in the preceding context — the "induction heads" task popularised
by Anthropic's work on in-context learning. It is the canonical minimum test
for whether a model can latch onto and match repeated patterns.

Run:
    python examples/pattern_matching.py

You should see the training loss drop and the model produce copy completions
that look like ``... A B C D | A B C D``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

# Allow `python examples/pattern_matching.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace_transformer import (
    CharTokenizer,
    LaplaceConfig,
    LaplaceTransformer,
)


def make_copy_batch(
    tokenizer: CharTokenizer,
    batch_size: int,
    pattern_len: int,
    alphabet: str,
    sep: str = "|",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch where each row is ``pattern SEP pattern``.

    The model is trained to predict the second copy of the pattern given the
    first copy plus the separator.
    """
    xs, ys = [], []
    for _ in range(batch_size):
        pat = "".join(random.choice(alphabet) for _ in range(pattern_len))
        text = pat + sep + pat
        ids = tokenizer.encode(text)
        xs.append(ids[:-1])
        ys.append(ids[1:])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    # Only supervise the *second* copy: mask out the prompt tokens.
    prompt_len = pattern_len  # plus SEP lives at index pattern_len in y
    mask_upto = pattern_len  # positions 0..pattern_len-1 in y correspond to
    y_masked = y.clone()
    y_masked[:, :mask_upto] = -100
    return x, y_masked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pattern-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    alphabet = "abcdefghij"
    sep = "|"
    tokenizer = CharTokenizer.from_text(alphabet + sep)

    max_len = 2 * args.pattern_len + 4
    cfg = LaplaceConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=max_len,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ffn_mult=4,
        dropout=0.0,
        causal=True,
        use_feature_map=False,   # not needed for this toy task
        pos_num_modes=min(16, max_len),
    )
    model = LaplaceTransformer(cfg).to(args.device)
    print(f"Model parameters: {model.num_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for step in range(1, args.steps + 1):
        x, y = make_copy_batch(tokenizer, args.batch_size, args.pattern_len, alphabet, sep)
        x, y = x.to(args.device), y.to(args.device)
        out = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {out['loss'].item():.4f}")

    # --- sample a few copy completions -----------------------------------
    print("\n--- copy completions ---")
    model.eval()
    for _ in range(5):
        pat = "".join(random.choice(alphabet) for _ in range(args.pattern_len))
        prompt = pat + sep
        ids = tokenizer.encode_tensor(prompt).unsqueeze(0).to(args.device)
        out = model.generate(ids, max_new_tokens=args.pattern_len, temperature=1e-6)
        print(f"  target={pat}  got={tokenizer.decode(out[0].tolist())[-args.pattern_len:]}")

    # --- demonstrate match_pattern read-out ------------------------------
    print("\n--- match_pattern scores ---")
    haystack_text = "xxxabcdefxxxabcabc"
    needle_text = "abc"
    # Re-build a tokenizer that knows the extra 'x' character.
    tokenizer2 = CharTokenizer.from_text(alphabet + sep + "x")
    cfg2 = LaplaceConfig(
        vocab_size=tokenizer2.vocab_size,
        max_len=64,
        d_model=64,
        num_heads=4,
        num_layers=2,
        pos_num_modes=16,
        use_feature_map=False,
    )
    demo_model = LaplaceTransformer(cfg2)  # untrained, just for shape demo
    scores = demo_model.match_pattern(
        tokenizer2.encode_tensor(haystack_text),
        tokenizer2.encode_tensor(needle_text),
    )
    print(f"  haystack: {haystack_text}")
    print(f"  needle:   {needle_text}")
    print(f"  scores:   {scores.tolist()}")


if __name__ == "__main__":
    main()
