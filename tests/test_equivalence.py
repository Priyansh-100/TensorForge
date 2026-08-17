"""
Fast equivalence checks — the verification suite as a test harness.

Run with pytest:        python -m pytest tests -q
Or plain python:        python tests/test_equivalence.py

The GPT-based checks load trained checkpoints from checkpoints/; they are
skipped (with a note) when a checkpoint is missing so the suite stays green
on a fresh clone without retraining.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from transformer.model import (  # noqa: E402
    precompute_rope,
    apply_rope,
    create_look_ahead_mask,
)
from transformer.attention import attention_forward, attention_backward  # noqa: E402
from transformer.gpt import GPT, CharTokenizer  # noqa: E402

# Checkpoints pickle the tokenizer as __main__.CharTokenizer; in pytest the
# test module is not __main__, so register the class there — the same
# namespace contract the CLI scripts get for free by running as __main__.
import __main__ as _main  # noqa: E402

_main.CharTokenizer = CharTokenizer


def _load_gpt(name):
    """Load a GPT checkpoint; returns (model, tokenizer) or None."""
    path = os.path.join(CKPT, name)
    if not os.path.exists(path):
        print(f"  skip: {name} not found — retrain with scripts/gpt.py")
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = ckpt["tokenizer"]
    model = GPT(vocab_size=tokenizer.vocab_size, max_len=128, rope=True,
                num_kv_heads=ckpt.get("num_kv_heads"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def test_attention_gradients():
    """Manual attention backward must match torch.autograd (~1e-15) and
    finite differences (~1e-8)."""
    torch.manual_seed(42)
    B, H, T, D = 2, 3, 5, 8
    Q = torch.randn(B, H, T, D, dtype=torch.double)
    K = torch.randn(B, H, T, D, dtype=torch.double)
    V = torch.randn(B, H, T, D, dtype=torch.double)
    upstream = torch.randn_like(Q)

    Qg, Kg, Vg = [x.clone().requires_grad_(True) for x in (Q, K, V)]
    F.scaled_dot_product_attention(Qg, Kg, Vg).backward(upstream)

    Y, P, _ = attention_forward(Q, K, V)
    dQ, dK, dV = attention_backward(upstream, Q, K, V, P)

    def rel_err(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

    assert rel_err(dQ, Qg.grad) < 1e-8
    assert rel_err(dK, Kg.grad) < 1e-8
    assert rel_err(dV, Vg.grad) < 1e-8

    # finite-difference spot check on dQ
    def loss_fn(q, k, v):
        return (attention_forward(q, k, v)[0] * upstream).sum()

    eps = 1e-6
    idxs = torch.randint(0, Q.view(-1).numel(), (10,))
    for i in idxs:
        Qp, Qm = Q.clone(), Q.clone()
        Qp.view(-1)[i] += eps
        Qm.view(-1)[i] -= eps
        fd = (loss_fn(Qp, K, V) - loss_fn(Qm, K, V)) / (2 * eps)
        assert abs(fd - dQ.view(-1)[i]).item() / (abs(fd) + 1e-12) < 1e-5
    print("  attention gradients vs autograd & finite differences: OK")


def test_rope_toeplitz():
    """RoPE scores depend only on the relative offset: score[m+1,n+1] == score[m,n]."""
    torch.manual_seed(0)
    d_model, max_len = 32, 16
    cos, sin = precompute_rope(d_model, max_len)
    q = torch.randn(1, d_model)
    k = torch.randn(1, d_model)

    scores = torch.zeros(max_len, max_len)
    for m in range(max_len):
        q_m = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        for n in range(max_len):
            scores[m, n] = (q_m * apply_rope(k, cos[n : n + 1], sin[n : n + 1])).sum()
    assert (scores[1:, 1:] - scores[:-1, :-1]).abs().max().item() < 1e-4
    print("  RoPE Toeplitz property (relative positions only): OK")


def test_kv_cache_prefill_bit_identical():
    """Cached prefill must be bit-identical to the plain forward pass."""
    loaded = _load_gpt("gpt_rope.pt")
    if loaded is None:
        return
    model, tokenizer = loaded
    idx = torch.tensor([tokenizer.encode("To be, or not to be")], dtype=torch.long)
    with torch.no_grad():
        mask = create_look_ahead_mask(idx.size(1))
        naive = model(idx, mask)
        caches = [None] * len(model.blocks)
        x, caches = model._cached_forward(idx, caches, 0, mask)
        cached = model.lm_head(model.ln_f(x))
    assert (naive - cached).abs().max().item() == 0.0, "prefill cache mismatch"
    print("  KV-cache prefill bit-identical: OK")


def test_gqa_step_and_grouping():
    """GQA/MQA: cached one-token step within tolerance; vectorized grouping
    matches a hand-rolled per-group loop."""
    for ckpt_name, kv_heads in (("gpt_rope_gqa.pt", 2), ("gpt_rope_mqa.pt", 1)):
        loaded = _load_gpt(ckpt_name)
        if loaded is None:
            continue
        model, tokenizer = loaded
        idx = torch.randint(0, tokenizer.vocab_size, (1, 12), dtype=torch.long)
        with torch.no_grad():
            mask = create_look_ahead_mask(idx.size(1))
            naive = model(idx, mask)
            caches = [None] * len(model.blocks)
            x, caches = model._cached_forward(idx, caches, 0, mask)
            nxt = naive[:, -1:].argmax(-1)
            x, caches = model._cached_forward(nxt, caches, 12)
            step = model.lm_head(model.ln_f(x))
            seq = torch.cat([idx, nxt], dim=1)
            naive_step = model(seq, create_look_ahead_mask(13))[:, -1:]
        assert (step - naive_step).abs().max().item() < 1e-4, f"{ckpt_name}: cached step mismatch"

        attn = model.blocks[0].self_attn
        assert attn.num_kv_heads == kv_heads and model.num_heads == 4

        # group-loop reference: head i uses KV head i // group_size
        torch.manual_seed(0)
        xr = torch.randn(1, 12, attn.d_model)
        fast = attn(xr, xr, xr, mask, rope_cos=model.cos[:12], rope_sin=model.sin[:12],
                    rope_cos_k=model.cos_k[:12], rope_sin_k=model.sin_k[:12])
        B, T, _ = xr.shape
        q = apply_rope(attn.W_q(xr), model.cos[:12].unsqueeze(0), model.sin[:12].unsqueeze(0))
        k = apply_rope(attn.W_k(xr), model.cos_k[:12].unsqueeze(0), model.sin_k[:12].unsqueeze(0))
        v = attn.W_v(xr)
        q = q.view(B, T, attn.num_heads, attn.d_k).transpose(1, 2)
        k = k.view(B, T, attn.num_kv_heads, attn.d_kv).transpose(1, 2)
        v = v.view(B, T, attn.num_kv_heads, attn.d_kv).transpose(1, 2)
        heads = []
        for i in range(attn.num_heads):
            j = i // attn.group_size
            scores = q[:, i] @ k[:, j].transpose(-2, -1) / math.sqrt(attn.d_k)
            scores = scores.masked_fill(~mask.squeeze(1), float("-inf"))
            heads.append(F.softmax(scores, dim=-1) @ v[:, j])
        ref = attn.W_o(torch.stack(heads, dim=1).transpose(1, 2).contiguous().view(B, T, attn.d_model))
        assert (fast - ref).abs().max().item() < 1e-4, f"{ckpt_name}: grouping mismatch"
    print("  GQA/MQA cached step + group-loop equivalence: OK")


def test_length_extrapolation():
    """RoPE generates past the training window; learned positions refuse."""
    loaded = _load_gpt("gpt_rope.pt")
    if loaded is None:
        return
    model, tokenizer = loaded
    idx = torch.tensor([tokenizer.encode("To be, or not to be")], dtype=torch.long)
    torch.manual_seed(7)
    out = model.generate_cached(idx, 170 - idx.size(1), temperature=0.9)
    assert out.size(1) == 170, "RoPE should not be capped at the training window"
    print("  length extrapolation past 128-token window: OK")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except AssertionError:
            traceback.print_exc()
            sys.exit(1)
    print("\nAll checks passed.")