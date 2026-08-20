"""
Byte-level BPE tokenizer from scratch (GPT-2 / tiktoken style).

The corpus is treated as raw UTF-8 bytes — 256 base tokens. Repeatedly, the
most frequent adjacent pair (a, b) of tokens is merged into one new token id:
    256 + len(merges_so_far)
until the vocabulary budget is reached. The merge list IS the tokenizer; the
training corpus is discarded after learning.

Encoding walks the token stream with the same greedy rule that created the
merges: at every step merge the LOWEST-ranked adjacent pair first.

Why byte-level?
    - no unknown tokens: any UTF-8 text encodes, whatever the vocabulary
    - merges learn real word/character fragments ("th", "ing", "To") and
      compress the stream (fewer tokens per char than a char tokenizer)
    - decoding is exact: every token id maps back to its bytes

Class summary:
    BPETokenizer(text, vocab_size)   learn merges from a corpus
    encode(text)  -> list[int]
    decode(ids)   -> str
    vocab_size    -> int   (256 + number of merges)
    merges        -> dict[(a, b), id]   (the learned tokenizer)
"""


class BPETokenizer:
    """Byte-level BPE. Pure Python, no external deps (as everything else here)."""

    def __init__(self, text: str, vocab_size: int = 512):
        # Base vocab: every byte 0..255 is its own token id.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # pair (a, b) -> merge token id. Insertion order = rank = id.
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab_size = self._learn(text, vocab_size)

    # -- learning -----------------------------------------------------------

    def _learn(self, text: str, vocab_size: int) -> int:
        ids = list(text.encode("utf-8"))
        while 256 + len(self.merges) < vocab_size and len(ids) > 1:
            # 1. count every adjacent pair
            counts: dict[tuple[int, int], int] = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            # 2. merge the most frequent one
            top = max(counts.items(), key=lambda kv: kv[1])[0]
            new_id = 256 + len(self.merges)
            self.merges[top] = new_id
            self.vocab[new_id] = self.vocab[top[0]] + self.vocab[top[1]]
            ids = self._replace(ids, top, new_id)
        return 256 + len(self.merges)

    @staticmethod
    def _replace(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Scan ids left to right, merging every adjacent occurrence of pair."""
        out: list[int] = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == pair[0] and i + 1 < n and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # -- inference ----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            # lowest-rank pair present in the stream (rank == merge token id)
            rank, pair = None, None
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                r = self.merges.get(p)
                if r is not None and (rank is None or r < rank):
                    rank, pair = r, p
            if pair is None or rank is None:
                break
            ids = self._replace(ids, pair, rank)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size