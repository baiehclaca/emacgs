"""A minimal byte-level / character tokenizer for quick demos.

This is intentionally tiny — it is not meant to replace sentencepiece / tiktoken,
but it makes the examples and tests runnable with zero external downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


@dataclass
class CharTokenizer:
    """Character-level tokenizer built from a training corpus or explicit vocab.

    Special tokens: ``<pad>`` (0), ``<bos>`` (1), ``<eos>`` (2), ``<unk>`` (3).
    """

    stoi: dict[str, int] = field(default_factory=dict)
    itos: dict[int, str] = field(default_factory=dict)

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    UNK: int = 3

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from the unique characters in ``text``."""
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        chars = sorted(set(text))
        stoi = {tok: i for i, tok in enumerate(specials)}
        for ch in chars:
            if ch not in stoi:
                stoi[ch] = len(stoi)
        itos = {i: tok for tok, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def bytes_tokenizer(cls) -> "CharTokenizer":
        """Tokenizer that treats each byte (0..255) as a token, plus specials."""
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        stoi = {tok: i for i, tok in enumerate(specials)}
        for b in range(256):
            stoi[f"<b{b}>"] = len(stoi)
        itos = {i: tok for tok, i in stoi.items()}
        t = cls(stoi=stoi, itos=itos)
        t._byte_mode = True  # type: ignore[attr-defined]
        return t

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        byte_mode = getattr(self, "_byte_mode", False)
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS)
        if byte_mode:
            for b in text.encode("utf-8"):
                ids.append(self.stoi[f"<b{b}>"])
        else:
            for ch in text:
                ids.append(self.stoi.get(ch, self.UNK))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        byte_mode = getattr(self, "_byte_mode", False)
        if byte_mode:
            out = bytearray()
            for i in ids:
                tok = self.itos.get(int(i), "")
                if tok.startswith("<b") and tok.endswith(">") and tok[2:-1].isdigit():
                    out.append(int(tok[2:-1]))
            return out.decode("utf-8", errors="replace")
        return "".join(
            self.itos.get(int(i), "")
            for i in ids
            if self.itos.get(int(i), "") not in {"<pad>", "<bos>", "<eos>"}
        )

    def encode_tensor(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> torch.Tensor:
        return torch.tensor(self.encode(text, add_bos, add_eos), dtype=torch.long)
