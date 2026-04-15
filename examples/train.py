"""Minimal character-level language-model training loop.

Takes a path to a plain-text file, trains a small LaplaceTransformer on it,
then samples a continuation. The loop is deliberately tiny (a few dozen lines)
so it is easy to read end-to-end, in the spirit of nanoGPT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laplace_transformer import CharTokenizer, LaplaceConfig, LaplaceTransformer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--sample-tokens", type=int, default=200)
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode_tensor(text)
    print(f"corpus: {len(text):,} chars, vocab: {tokenizer.vocab_size}")

    cfg = LaplaceConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        pos_num_modes=min(64, args.block_size),
        use_feature_map=False,  # keep the LM fast; feature_map is O(T*K*C)
    )
    model = LaplaceTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"model params: {model.num_parameters():,}")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.block_size, args.batch_size, args.device)
        out = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  loss {out['loss'].item():.4f}")

    print("\n--- sample ---")
    ids = tokenizer.encode_tensor(args.prompt).unsqueeze(0).to(args.device)
    out = model.generate(ids, max_new_tokens=args.sample_tokens, temperature=0.9, top_k=40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
