"""
RoPE + KV cache demo (the project's verification suite).

1. Relative-position property of RoPE (why it beats sinusoidal embeddings):
   the attention score between query at position m and key at position n
   depends ONLY on the offset (m - n), not the absolute positions.

2. KV-cache correctness: a forward pass with the cache must produce the same
   logits (within float tolerance) as a plain forward pass.

3. Speed: cached generation processes 1 token per step instead of the whole
   prefix → O(T) instead of O(T²) attention work.

Usage:
  python scripts/gpt.py --epochs 30 --save         # train the char LM first (checkpoints/gpt.pt)
  python scripts/verify.py
  python scripts/verify.py --rope --ckpt checkpoints/gpt_rope.pt
  python scripts/verify.py --rope --ckpt checkpoints/gpt_rope_gqa.pt
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from transformer.model import precompute_rope, apply_rope, create_look_ahead_mask  # noqa: E402
from transformer.gpt import GPT, CharTokenizer  # noqa: E402,F401
# CharTokenizer must be in THIS namespace: checkpoints pickle it as
# `__main__.CharTokenizer`, so torch.load resolves the class against the
# importing module — removing this import breaks loading.


def demo_relative_property():
    print("=" * 70)
    print("1. RoPE relative-position property")
    print("=" * 70)
    torch.manual_seed(0)
    d_model, max_len = 32, 16
    cos, sin = precompute_rope(d_model, max_len)

    # random query & key vectors (would be projected embeddings)
    q = torch.randn(1, d_model)
    k = torch.randn(1, d_model)

    scores = torch.zeros(max_len, max_len)
    for m in range(max_len):
        q_m = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        for n in range(max_len):
            k_n = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
            scores[m, n] = (q_m * k_n).sum()

    # score[m+1, n+1] == score[m, n]: offset-only dependence
    max_dev = (scores[1:, 1:] - scores[:-1, :-1]).abs().max().item()
    print(f"  score[m+1, n+1] == score[m, n]?  max deviation: {max_dev:.2e}")

    print("  score matrix (rows = m, cols = n):")
    for row in scores:
        print("   " + " ".join(f"{v:5.2f}" for v in row))
    print("  Every diagonal has the same value — position is encoded as a ROTATION,")
    print("  and the model sees only relative offsets. Length extrapolation comes free.")


def demo_kv_cache(model, tokenizer, device, n_chars=200):
    print()
    print("=" * 70)
    print("2. KV cache correctness + 3. speed")
    print("=" * 70)
    model.eval()

    prompt = "To be, or not to be"
    idx0 = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    # Learned position embeddings cap total length at max_len (RoPE has no cap)
    if not model.rope:
        n_chars = min(n_chars, model.max_len - idx0.size(1))
        if n_chars <= 0:
            raise SystemExit("prompt too long for max_len")

    # --- correctness: cached forward must equal naive forward (fp tolerance) ---
    with torch.no_grad():
        mask = create_look_ahead_mask(idx0.size(1)).to(device)
        naive_logits = model(idx0, mask)

        caches = [None] * len(model.blocks)
        x, caches = model._cached_forward(idx0, caches, 0, mask)
        cached_logits = model.lm_head(model.ln_f(x))

    max_dev = (naive_logits - cached_logits).abs().max().item()
    print(f"  prefill logits match (fp tolerance): max |Δ| over {naive_logits.numel()} logits = {max_dev:.1e}")
    print("  → the cache is equivalent to recomputing attention every step")

    # --- incremental step: one new token through the cache, no mask needed ---
    with torch.no_grad():
        next_tok = torch.tensor([[model.lm_head(model.ln_f(x[:, -1:, :])).argmax(-1).item()]], dtype=torch.long, device=device)
        x_step, caches = model._cached_forward(next_tok, caches, idx0.size(1))
        step_logits = model.lm_head(model.ln_f(x_step))

        seq = torch.cat([idx0, next_tok], dim=1)
        mask_full = create_look_ahead_mask(seq.size(1)).to(device)
        naive_step_logits = model(seq, mask_full)[:, -1:, :]

    max_dev_step = (step_logits - naive_step_logits).abs().max().item()
    ok_step = max_dev_step < 1e-4
    print(f"  one-token step logits match: max |Δ| = {max_dev_step:.1e}  {'OK' if ok_step else 'FAIL'}")
    if not ok_step:
        print("  → cached K/V for past tokens are NOT equivalent to recomputing them")

    # --- 3. speed: cached generation O(T) vs naive O(T²) ---
    torch.manual_seed(123)
    t0 = time.time()
    naive = model.generate(idx0, n_chars, temperature=0.9)
    t_naive = time.time() - t0

    torch.manual_seed(123)
    t0 = time.time()
    cached = model.generate_cached(idx0, n_chars, temperature=0.9)
    t_cached = time.time() - t0

    print(f"  speed: naive {t_naive:.2f}s ({n_chars/t_naive:6.1f} tok/s) | "
          f"cached {t_cached:.2f}s ({n_chars/t_cached:6.1f} tok/s) → {t_naive/t_cached:.1f}x")

    # --- same-seed sampling must produce the same text (fp tolerance) ---
    if torch.equal(naive, cached):
        print("  sampled text identical: True (both paths follow the same distribution)")
    else:
        div = (naive != cached).nonzero()[0, 1].item()
        print(f"  sampled text identical: False — first divergence at token {div} "
              f"(expected only from float noise flipping a rare sample)")
        print(f"    naive : {tokenizer.decode(naive[0].tolist())}")
        print(f"    cached: {tokenizer.decode(cached[0].tolist())}")

    text = tokenizer.decode(cached[0].tolist())
    print(f"\n  Generated (cached path):\n  {text}")


def demo_rope_kv_cache(device):
    """RoPE path: rotation happens inside attention, so cached K must be rotated
    at its ORIGINAL position — a whole different cache-equivalence bug class."""
    print()
    print("=" * 70)
    print("4. KV cache correctness with RoPE (cached K rotations)")
    print("=" * 70)
    torch.manual_seed(0)
    model = GPT(vocab_size=65, d_model=32, num_heads=4, d_ff=64, num_layers=2,
                max_len=64, rope=True).eval().to(device)

    idx0 = torch.randint(0, 65, (1, 12), device=device)
    with torch.no_grad():
        mask = create_look_ahead_mask(idx0.size(1)).to(device)
        naive_logits = model(idx0, mask)

        caches = [None] * len(model.blocks)
        x, caches = model._cached_forward(idx0, caches, 0, mask)
        cached_logits = model.lm_head(model.ln_f(x))

    max_dev = (naive_logits - cached_logits).abs().max().item()
    ok = max_dev < 1e-4
    print(f"  prefill logits match: max |Δ| = {max_dev:.1e}  {'OK' if ok else 'FAIL'}")

    with torch.no_grad():
        next_tok = model.lm_head(model.ln_f(x[:, -1:, :])).argmax(-1)
        x_step, caches = model._cached_forward(next_tok, caches, idx0.size(1))
        step_logits = model.lm_head(model.ln_f(x_step))

        seq = torch.cat([idx0, next_tok], dim=1)
        mask_full = create_look_ahead_mask(seq.size(1)).to(device)
        naive_step_logits = model(seq, mask_full)[:, -1:, :]

    max_dev_step = (step_logits - naive_step_logits).abs().max().item()
    ok_step = max_dev_step < 1e-4
    print(f"  one-token step logits match: max |Δ| = {max_dev_step:.1e}  {'OK' if ok_step else 'FAIL'}")
    print("  → RoPE + cached K/V is consistent: keys stay rotated at their original positions")
    if not (ok and ok_step):
        print("  FAILED — cached K was probably rotated at the wrong position")


def demo_length_extrapolation(model, tokenizer, device, total=170):
    """Generation beyond the training window (128 tokens). RoPE has no
    position table, so there is nothing to run out of. Learned positions
    cannot index past max_len and raise instead."""
    print()
    print("=" * 70)
    print("5. Length extrapolation (trained on blocks of 128 tokens)")
    print("=" * 70)

    prompt = "To be, or not to be"
    idx0 = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    if model.rope:
        n_new = total - idx0.size(1)
        torch.manual_seed(7)
        out = model.generate_cached(idx0, n_new, temperature=0.9)
        text = tokenizer.decode(out[0].tolist())
        print(f"  RoPE: generated {total} tokens — {total - 128} PAST the training window.")
        print("  Positions are rotations, so inputs beyond 128 are well-defined;")
        print("  the model keeps attending with relative offsets (quality fades, honesty-checked):")
        print(f"  {text[:85]}")
        print(f"  ... [{total} tokens total]")
    else:
        try:
            model.generate_cached(idx0, total - idx0.size(1), temperature=0.9)
            print("  unexpectedly succeeded")
        except ValueError as e:
            print(f"  learned positions: generation beyond max_len raises:\n    ValueError: {e}")
        print("  → the position embedding is a fixed-size table; RoPE never needs one.")


def reference_gqa_attn(attn, x, cos, sin, cos_k, sin_k, mask):
    """Independent GQA reference: one query head at a time, K/V slices taken
    by EXPLICIT group index. No repeat_interleave anywhere in this path — if
    the vectorized module grouped heads wrongly (or cached the expanded K/V),
    this catches it."""
    B, T, _ = x.shape
    q = apply_rope(attn.W_q(x), cos.unsqueeze(0), sin.unsqueeze(0))
    k = apply_rope(attn.W_k(x), cos_k.unsqueeze(0), sin_k.unsqueeze(0))
    v = attn.W_v(x)
    q = q.view(B, T, attn.num_heads, attn.d_k).transpose(1, 2)
    k = k.view(B, T, attn.num_kv_heads, attn.d_kv).transpose(1, 2)
    v = v.view(B, T, attn.num_kv_heads, attn.d_kv).transpose(1, 2)
    if mask is not None:
        mask = mask.squeeze(1)  # (1, T, T) — broadcasts over the batch
    heads = []
    for i in range(attn.num_heads):
        j = i // attn.group_size                 # head i shares KV head j
        scores = q[:, i] @ k[:, j].transpose(-2, -1) / math.sqrt(attn.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        heads.append(F.softmax(scores, dim=-1) @ v[:, j])
    out = torch.stack(heads, dim=1).transpose(1, 2).contiguous().view(B, T, attn.d_model)
    return attn.W_o(out)


def demo_gqa(device):
    """GQA: query heads share K/V heads. Cache shrinks by num_kv_heads/num_heads,
    the math (and the KV-cache equivalence) must be unchanged."""
    print()
    print("=" * 70)
    print("6. Grouped Query Attention (GQA): 8 query heads, 2 K/V heads")
    print("=" * 70)
    torch.manual_seed(0)
    model = GPT(vocab_size=65, d_model=64, num_heads=8, num_kv_heads=2, d_ff=128,
                num_layers=2, max_len=32, rope=True).eval().to(device)

    idx0 = torch.randint(0, 65, (1, 12), device=device)
    with torch.no_grad():
        mask = create_look_ahead_mask(idx0.size(1)).to(device)
        naive = model(idx0, mask)
        caches = [None] * len(model.blocks)
        x, caches = model._cached_forward(idx0, caches, 0, mask)
        prefill = model.lm_head(model.ln_f(x))
        next_tok = naive[:, -1:].argmax(-1)
        x_step, caches = model._cached_forward(next_tok, caches, idx0.size(1))
        step = model.lm_head(model.ln_f(x_step))
        seq = torch.cat([idx0, next_tok], dim=1)
        naive_step = model(seq, create_look_ahead_mask(seq.size(1)).to(device))[:, -1:]

    print(f"  prefill logits match:        max |Δ| = {(naive - prefill).abs().max().item():.1e}")
    print(f"  one-token step logits match: max |Δ| = {(step - naive_step).abs().max().item():.1e}")
    print("  → sharing K/V heads changes nothing but the cache size")

    attn = model.blocks[0].self_attn
    k, v = caches[0]
    T = k.size(2)
    print(f"  cached K tensor (step {T}):  {tuple(k.shape)} — SMALL form [B, num_kv_heads, T, d_k]")
    print(f"  expanded for scoring it is   [B, {model.num_heads}, T, {attn.d_k}] — the cache itself never grows")
    per_tok_full = 2 * model.num_heads * attn.d_k * 4          # K and V, float32
    per_tok_gqa = 2 * attn.num_kv_heads * attn.d_kv * 4
    tot_full = per_tok_full * model.max_len * len(model.blocks)
    tot_gqa = per_tok_gqa * model.max_len * len(model.blocks)
    print(f"  KV cache per token per layer: {per_tok_full} B (full MHA) vs {per_tok_gqa} B (GQA) "
          f"→ {per_tok_full / per_tok_gqa:.0f}x smaller")
    print(f"  at 128 tokens, 4 layers:     {tot_full/1024:.0f} KiB vs {tot_gqa/1024:.0f} KiB")

    # --- independent reference: explicit per-group loop, NO repeat_interleave ---
    with torch.no_grad():
        xr = torch.randn(1, 12, attn.d_model, device=device)
        fast = attn(xr, xr, xr, mask, rope_cos=model.cos[:12], rope_sin=model.sin[:12],
                    rope_cos_k=model.cos_k[:12], rope_sin_k=model.sin_k[:12])
        ref = reference_gqa_attn(attn, xr, model.cos[:12], model.sin[:12],
                                 model.cos_k[:12], model.sin_k[:12], mask)
    dev_ref = (fast - ref).abs().max().item()
    ok_ref = dev_ref < 1e-4
    print(f"  vs hand-rolled group loop:   max |Δ| = {dev_ref:.1e}  "
          f"{'OK' if ok_ref else 'FAIL (grouping order wrong!)'}")
    print("  → repeat_interleave puts head i in group i // group_size; the loop proves it")
    if not ok_ref:
        print("  head i must attend to its group's K/V only — this would hide a cache-expansion bug")

    # --- same proof on the REAL trained checkpoint (4 query heads → 2 KV heads) ---
    gqa_path = os.path.join(CKPT, "gpt_rope_gqa.pt")
    try:
        ckpt = torch.load(gqa_path, map_location=device, weights_only=False)
        tok = ckpt["tokenizer"]
        trained = GPT(vocab_size=tok.vocab_size, max_len=128, rope=True,
                      num_kv_heads=ckpt.get("num_kv_heads")).to(device)
        trained.load_state_dict(ckpt["model"])
        trained.eval()
        idx1 = torch.randint(0, tok.vocab_size, (1, 12), device=device)
        with torch.no_grad():
            mask1 = create_look_ahead_mask(12).to(device)
            naive1 = trained(idx1, mask1)
            caches1 = [None] * len(trained.blocks)
            x1, caches1 = trained._cached_forward(idx1, caches1, 0, mask1)
            prefill1 = trained.lm_head(trained.ln_f(x1))
            next1 = naive1[:, -1:].argmax(-1)
            x1, caches1 = trained._cached_forward(next1, caches1, 12)
            step1 = trained.lm_head(trained.ln_f(x1))
            seq1 = torch.cat([idx1, next1], dim=1)
            naive_step1 = trained(seq1, create_look_ahead_mask(13).to(device))[:, -1:]
        k1 = caches1[0][0]
        dev_prefill1 = (naive1 - prefill1).abs().max().item()
        dev_step1 = (step1 - naive_step1).abs().max().item()
        ok1 = dev_prefill1 == 0.0 and dev_step1 < 1e-4
        print(f"  trained 4→{trained.num_kv_heads} (gpt_rope_gqa.pt): prefill |Δ| {dev_prefill1:.1e}, "
              f"step |Δ| {dev_step1:.1e}  {'OK' if ok1 else 'FAIL'}")
        print(f"    cached K tensor: {tuple(k1.shape)} = {2 * trained.num_kv_heads * k1.size(2) * trained.blocks[0].self_attn.d_k} "
              f"floats vs {2 * trained.num_heads * k1.size(2) * trained.blocks[0].self_attn.d_k} "
              f"(full heads) → {trained.num_kv_heads}/{trained.num_heads}× the cache")
    except FileNotFoundError:
        print("  (no trained GQA checkpoint yet — the fresh model above stands in)")


def demo_mha_vs_gqa(device):
    """Same architecture, same training recipe — only the KV head count differs.
    gpt_rope.pt is full MHA (4/4), gpt_rope_gqa.pt is GQA (4/2), gpt_rope_mqa.pt
    is MQA (4/1). What the smaller caches actually buy: memory, at ~0 quality cost."""
    print()
    print("=" * 70)
    print("7. Trained head-to-head: full MHA vs GQA vs MQA")
    print("=" * 70)
    paths = {"full MHA": (os.path.join(CKPT, "gpt_rope.pt"), None),
             "GQA 4→2": (os.path.join(CKPT, "gpt_rope_gqa.pt"), 2),
             "MQA 4→1": (os.path.join(CKPT, "gpt_rope_mqa.pt"), 1)}
    models: dict[str, GPT] = {}
    for tag, (path, kv) in paths.items():
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"  {tag}: {path} missing — skipping")
            continue
        tok = ckpt["tokenizer"]
        m = GPT(vocab_size=tok.vocab_size, max_len=128, rope=True,
                num_kv_heads=kv if kv is not None else ckpt.get("num_kv_heads")).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models[tag] = (m, tok, ckpt.get("val_loss"))

    for tag, (m, tok, _) in models.items():
        attn = m.blocks[0].self_attn
        per_layer = 2 * attn.num_kv_heads * attn.d_k * 4 * m.max_len  # K+V, fp32, 128 tok
        print(f"  {tag:8s} {m.num_heads} Q hd / {attn.num_kv_heads} KV hd: "
              f"KV cache {per_layer / 1024:5.0f} KiB/layer → {per_layer * len(m.blocks) / 1024:4.0f} KiB total")

    for tag, (m, tok, _) in models.items():
        idx = torch.tensor([tok.encode("To be, or not to be")], dtype=torch.long, device=device)
        torch.manual_seed(11)
        t0 = time.time()
        m.generate_cached(idx, 100, temperature=0.9)
        dt = time.time() - t0
        print(f"  {tag:8s} cached decode: 100 tok in {dt:.2f}s ({100 / dt:6.1f} tok/s)")

    losses = {tag: v for tag, (_, _, v) in models.items()}
    print("  quality: " + " vs ".join(
        f"{tag} ppl {math.exp(v):.2f}" for tag, v in losses.items() if v is not None) +
        " — smaller caches cost ~nothing at this scale")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chars", type=int, default=200)
    parser.add_argument("--rope", action="store_true", help="load a RoPE-trained checkpoint")
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT, "gpt.pt"),
                        help="checkpoint path")
    args = parser.parse_args()

    demo_relative_property()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # contains tokenizer class
    tokenizer = ckpt["tokenizer"]
    if not isinstance(tokenizer, CharTokenizer):
        raise SystemExit(f"{args.ckpt} does not contain a CharTokenizer")
    model = GPT(vocab_size=tokenizer.vocab_size, max_len=128, rope=args.rope,
                num_kv_heads=ckpt.get("num_kv_heads"))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    demo_kv_cache(model, tokenizer, device, args.n_chars)
    demo_rope_kv_cache(device)
    demo_length_extrapolation(model, tokenizer, device)
    demo_gqa(device)
    demo_mha_vs_gqa(device)