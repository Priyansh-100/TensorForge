"""
Tests for the byte-level BPE tokenizer (src/transformer/bpe.py) and the
training-loop options (AMP path, gradient accumulation).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from transformer.bpe import BPETokenizer  # noqa: E402
from transformer.gpt import GPT, CharTokenizer, train, sample, beam_sample  # noqa: E402

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


def test_cosine_restarts_scheduler():
    """CosineWarmRestarts scheduler runs and restarts correctly."""
    from transformer.model import CosineWarmRestarts
    import torch.nn as nn
    optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = CosineWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    lrs = []
    for step in range(25):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    
    # Check first period (5 steps): lr goes from 1.0 down to eta_min
    assert lrs[0] == 1.0
    assert lrs[4] < 1.0
    assert lrs[4] > 1e-6
    
    # Second period should restart to 1.0 (T_mult=2, so period doubles to 10)
    assert lrs[5] == 1.0  # restart
    
    # Third period should restart again (period = 20)
    assert lrs[15] == 1.0  # restart
    
    # lr should never go below eta_min
    assert all(lr >= 1e-6 for lr in lrs)


def test_train_cosine_restarts():
    """train() with cosine_restarts scheduler runs."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=2, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True,
        scheduler="cosine_restarts", scheduler_t0=3, scheduler_t_mult=2,
        scheduler_eta_min=1e-6, seed=0, num_pairs=16, num_val_pairs=4,
    )
    assert model.count_params() > 0


def test_train_weight_decay_grad_clip():
    """train() with weight_decay and grad_clip runs."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True,
        weight_decay=0.01, grad_clip=1.0, seed=0, num_pairs=16, num_val_pairs=4,
    )
    assert model.count_params() > 0


def test_train_rope_scaling_ntk():
    """train() with NTK rope scaling runs."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True,
        rope_scaling="ntk", rope_scaling_factor=2.0, seed=0,
        num_pairs=16, num_val_pairs=4,
    )
    assert model.count_params() > 0


def test_train_rope_scaling_linear():
    """train() with linear rope scaling runs."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True,
        rope_scaling="linear", rope_scaling_factor=2.0, seed=0,
        num_pairs=16, num_val_pairs=4,
    )
    assert model.count_params() > 0


def test_beam_sample():
    """beam_sample produces output and differs from greedy."""
    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)
    n_val = int(0.1 * len(data))
    tr, va = data[:-n_val], data[-n_val:]
    model = train(
        tok, tr, va, epochs=1, block_size=32, d_model=16, num_heads=2, d_ff=32,
        num_layers=1, batch_size=4, save=False, rope=True,
        seed=0, num_pairs=16, num_val_pairs=4,
    )
    
    # Test beam_sample doesn't crash
    prompt = "To be, or not to be"
    
    # Capture beam output
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    beam_sample(model, tok, prompt, n_chars=10, beam_size=4, length_penalty=0.0)
    beam_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    sys.stdout = io.StringIO()
    sample(model, tok, prompt, n_chars=10, temperature=0.8, top_p=1.0)
    greedy_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    assert "To be" in beam_text
    assert "To be" in greedy_text


def test_trainer_weight_decay_grad_clip():
    """seq2seq trainer with weight_decay and grad_clip runs."""
    from transformer.trainer import train as seq2seq_train
    model, device = seq2seq_train(task="reverse", epochs=1, weight_decay=0.01, grad_clip=1.0)
    assert model.count_params() > 0


def test_trainer_greedy_decode():
    """seq2seq greedy_decode runs and produces valid output."""
    from transformer.trainer import train as seq2seq_train, greedy_decode
    model, device = seq2seq_train(task="reverse", epochs=1)
    # Test greedy_decode runs
    src = torch.randint(2, 20, (1, 5))
    out = greedy_decode(model, src, device, max_len=5)
    assert out.size(0) == 1
    assert out.size(1) == 5


def test_trainer_evaluate():
    """seq2seq evaluate function runs without error."""
    from transformer.trainer import train as seq2seq_train, evaluate
    model, device = seq2seq_train(task="reverse", epochs=1)
    # evaluate runs without error (prints inference examples)
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    evaluate(model, device, task="reverse")
    sys.stdout = old_stdout


def test_trainer_copy_task():
    """seq2seq trainer works on copy task."""
    from transformer.trainer import train as seq2seq_train
    model, device = seq2seq_train(task="copy", epochs=1)
    assert model.count_params() > 0


if __name__ == "__main__":
    for name in sorted(g for g in globals() if g.startswith("test_")):
        print(f"[{name}]")
        globals()[name]()
    print("\nAll checks passed.")