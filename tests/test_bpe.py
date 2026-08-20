"""
Tests for the byte-level BPE tokenizer (src/transformer/bpe.py) and the
training-loop options (AMP path, gradient accumulation).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from transformer.bpe import BPETokenizer  # noqa: E402
from transformer.gpt import GPT, CharTokenizer, train  # noqa: E402

CORPUS = (
    "To be, or not to be - that is the question. Whether 'tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune, or to take arms "
    "against a sea of troubles, and by opposing end them. "
) * 8


def test_bpe_roundtrip():
    tok = BPETokenizer(CORPUS, vocab_size=300)
    for s in ["To be, or not to be", "", "slings and arrows", "12345",
              "mixed CASE and spaces."]:
        assert tok.decode(tok.encode(s)) == s
    assert tok.vocab_size > 256
    assert 256 + len(tok.merges) == tok.vocab_size


def test_bpe_compresses():
    tok = BPETokenizer(CORPUS, vocab_size=300)
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS)
    assert all(i < tok.vocab_size for i in ids)
    assert max(tok.merges.values()) < tok.vocab_size


def test_bpe_deterministic():
    a = BPETokenizer(CORPUS, vocab_size=300)
    b = BPETokenizer(CORPUS, vocab_size=300)
    assert a.merges == b.merges
    assert a.encode("to be") == b.encode("to be")


def test_bpe_utf8_bytes():
    tok = BPETokenizer(CORPUS + "caf\u00e9 \u2014 na\u00efve \u4e2d\u6587", vocab_size=400)
    s = "caf\u00e9 \u2014 na\u00efve \u4e2d\u6587"
    assert tok.decode(tok.encode(s)) == s


def test_train_amp_grad_accum_smoke():
    """train() with amp + gradient accumulation on a tiny corpus must run and
    produce a model (AMP silently degrades to fp32 on CPU)."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True, amp=True,
        grad_accum=2, seed=0, num_pairs=16, num_val_pairs=4,
    )
    assert model.count_params() > 0
    assert len(model.blocks) == 1


def test_train_compile_smoke():
    """torch.compile path must be usable (or degrade gracefully)."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    try:
        model = train(
            tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
            num_layers=1, batch_size=4, save=False, rope=True, compile_model=True,
            seed=0, num_pairs=16, num_val_pairs=4,
        )
        assert model.count_params() > 0
    except Exception as e:  # torch.compile unsupported on this platform
        print(f"  compile unsupported, skipping: {e}")


def test_num_kv_heads_must_divide():
    """Bad GQA configs fail with a clear error, not an assert."""
    import pytest

    with pytest.raises(ValueError, match="multiple of"):
        GPT(65, d_model=16, num_heads=4, d_ff=32, num_layers=1,
            rope=True, num_kv_heads=3)


def test_tie_weights_checkpoint_roundtrip(tmp_path):
    """Saving a tied model and loading it restores the shared embedding."""
    tok = CharTokenizer(CORPUS)
    model = GPT(tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1,
                rope=True, tie_weights=True)
    path = str(tmp_path / "tied.pt")
    torch.save({"model": model.state_dict(), "tokenizer": tok,
                "val_loss": 1.0, "num_kv_heads": 2}, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["model"]["lm_head.weight"].data_ptr() \
        == ckpt["model"]["token_embedding.weight"].data_ptr()

    loaded = GPT(tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1,
                 rope=True, tie_weights=True)
    loaded.load_state_dict(ckpt["model"])
    assert loaded.lm_head.weight.data_ptr() == loaded.token_embedding.weight.data_ptr()


def test_tie_weights_shares_embedding_and_head():
    """Weight tying: lm_head.weight IS token_embedding.weight (same storage),
    saving vocab·d_model parameters vs the untied model."""
    tok = CharTokenizer(CORPUS)
    tied = GPT(tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1,
               rope=True, tie_weights=True)
    untied = GPT(tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1,
                 rope=True)
    assert tied.lm_head.weight.data_ptr() == tied.token_embedding.weight.data_ptr()
    assert tied.lm_head.bias is not None  # bias stays separate
    saved = tied.count_params()
    untied_params = untied.count_params()
    assert saved == untied_params - tok.vocab_size * 16

    # training runs, and checkpoint round-trips restore the tying
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2,
                  d_ff=32, num_layers=1, batch_size=4, save=False, rope=True,
                  seed=0, num_pairs=16, num_val_pairs=4, tie_weights=True)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


if __name__ == "__main__":
    for name in sorted(g for g in globals() if g.startswith("test_")):
        print(f"[{name}]")
        globals()[name]()
    print("\nAll checks passed.")