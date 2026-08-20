# Transformer From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20MPS%20%7C%20Linux%20CUDA%20%7C%20CPU-lightgrey.svg)]()
[![CI](https://github.com/Priyansh-100/TensorForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Priyansh-100/TensorForge/actions/workflows/ci.yml)
[![Docs](https://github.com/Priyansh-100/TensorForge/actions/workflows/docs.yml/badge.svg)](https://priyansh-100.github.io/TensorForge/)
[![Coverage](https://codecov.io/gh/Priyansh-100/TensorForge/branch/main/graph/badge.svg)](https://codecov.io/gh/Priyansh-100/TensorForge)
[![from scratch](https://img.shields.io/badge/from_scratch-no%20nn.Transformer-success.svg)](src/transformer/model.py)
[![Ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/typed-mypy-2a6db2.svg)](https://mypy-lang.org)

**A complete transformer written line-by-line, verified numerically at every
step.** No `nn.Transformer`, no HuggingFace, no black boxes: attention,
positional encodings, RoPE, a KV cache, Grouped Query Attention, and a
character-level GPT are implemented, trained, and *proved correct* — the
equivalence checks are executable code, not claims.

> The core idea (2017, "Attention Is All You Need"): a token should decide
> *how much to listen to every other token*. That is all attention is — a
> weighted average. This repo builds that idea up from a single matrix
> multiply to a working language model, and pays for it with an unusual
> currency: *verification*. Every optimization added here (KV cache, RoPE
> rotation at arbitrary length, GQA) carries an independent numerical proof
> that it changes nothing about what the model computes.

---

## Table of Contents

- [Part 0 — The big picture (start here if you are new)](#part-0--the-big-picture-start-here-if-you-are-new)
- [Part 1 — Setup & prerequisites](#part-1--setup--prerequisites)
- [Part 2 — Project layout](#part-2--project-layout)
- [Part 3 — Step 1: the math of attention (`src/transformer/attention.py`)](#part-3--step-1-the-math-of-attention-srctransformerattentionpy)
- [Part 4 — Step 2: the architecture (`src/transformer/model.py`)](#part-4--step-2-the-architecture-srctransformermodelpy)
- [Part 5 — Step 3: train a seq2seq model (`scripts/train_seq2seq.py`)](#part-5--step-3-train-a-seq2seq-model-scriptstrain_seq2seqpy)
- [Part 6 — Step 4: decoding strategies (`scripts/decode.py`)](#part-6--step-4-decoding-strategies-scriptsdecodepy)
- [Part 7 — Step 5: look inside attention (`scripts/visualize.py` / `scripts/attention_dump.py`)](#part-7--step-5-look-inside-attention-scriptsvisualizepy--scriptsattention_dumppy)
- [Part 8 — Step 6: multi-GPU training (`scripts/train_dist.py`)](#part-8--step-6-multi-gpu-training-scriptstrain_distpy)
- [Part 9 — Step 7: the mini-GPT (`scripts/gpt.py`)](#part-9--step-7-the-mini-gpt-scriptsgptpy)
- [Part 10 — Step 8: Grouped Query Attention](#part-10--step-8-grouped-query-attention)
- [Part 11 — Step 9: the verification suite (`scripts/verify.py`)](#part-11--step-9-the-verification-suite-scriptsverifypy)
- [Part 12 — Step 10: BPE tokenizer & training knobs](#part-12--step-10-bpe-tokenizer--training-knobs)
- [Part 13 — Results at a glance](#part-13--results-at-a-glance)
- [Part 14 — Notebooks](#part-14--notebooks)
- [Part 15 — Troubleshooting](#part-15--troubleshooting)
- [Part 16 — FAQ](#part-16--faq)
- [Part 17 — Roadmap: improvements & advancement](#part-17--roadmap-improvements--advancement)
- [Appendix — The verification philosophy](#appendix--the-verification-philosophy)
- [References](#references)

---

## Part 0 — The big picture (start here if you are new)

### What is a transformer?

A transformer is a neural network for sequences: text, code, time series,
anything ordered. Two capabilities matter:

1. **It reads the whole input at once** (unlike RNNs, which read left to
   right and gradually forget). At every position, it decides how much to
   attend to every *other* position — this is **self-attention**.
2. **Its core math is order-agnostic** — a bag of tokens has no notion of
   "before". Order information has to be *injected*. That is the job of the
   **positional encodings**, and this repo implements all three historical
   flavors:
   - **sinusoidal** (the original paper, `src/transformer/model.py`),
   - **learned embeddings** (`scripts/gpt.py` default),
   - **RoPE / rotary** (what Llama and modern models use, `--rope`).

### A tiny glossary (all terms used in this repo)

| term | meaning |
|---|---|
| **token** | one atomic unit (here: one character, or one integer) |
| **embedding** | a learned vector (`d_model` floats) representing a token |
| **d_model** | width of every token's vector (64 here; 512 in the paper) |
| **head** | an independent attention "view"; heads attend differently in parallel and are merged at the end |
| **d_k** | width of each head's query/key vectors (`d_model / num_heads`) |
| **Q / K / V** | three views of the input: Q asks "what am I looking for?", K says "what do I contain?", V is the content passed on when Q matches K |
| **score** | dot product Q·K: how much this query wants this key |
| **attention weights** | softmax over scores: a probability distribution over keys |
| **causal mask** | forbids attending to the *future* (essential for decoder-only models) |
| **logits** | raw model scores over the vocabulary, before softmax |
| **teacher forcing** | training with the *true* previous token as input, never the model's own guesses |
| **autoregressive** | generating one token at a time, feeding outputs back as inputs |

### Training vs inference — the same model, two lives

- **Training** (next-token prediction): show `"To be"`, ask for `" be"`.
  A single forward pass predicts the next character at *every* position at
  once (teacher forcing). Loss = how wrong each prediction was.
- **Inference** (the "Continue" button): take only the *last* position's
  logits, sample one token, append it, repeat. This loop is **generation /
  decoding**.

> This asymmetry matters enormously: inference repeats work. The **KV
> cache** (Part 9) is the fix, and *proving the cache changes nothing* is
> the heart of this repo.

---

## Part 1 — Setup & prerequisites

You need Python 3.10+ and a rough idea of what a matrix multiply is. That
is genuinely about it. (Familiarity with `nn.Linear`, `softmax`, and
backprop helps, but every concept is re-derived in `src/transformer/attention.py`.)

```bash
cd Transformer
python3 -m venv venv            # isolated environment
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt # torch, numpy, matplotlib (+ notebook/test extras)
```

The project **auto-detects your device** (CUDA → MPS → CPU), so the same
code runs on NVIDIA GPUs, Apple Silicon, or plain CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

Sanity check before continuing:

```bash
python -c "import torch, matplotlib; print(torch.__version__)"
python src/transformer/attention.py   # ~2 s; see Part 3
```

---

## Part 2 — Project layout

```
src/transformer/        the library (importable package)
  model.py              core architecture (attention, RoPE, positions, encoder/decoder, masks)
  attention.py          manual attention forward/backward vs torch.autograd (the math proof)
  gpt.py                mini-GPT: model, training (AMP / grad-accum / compile), GQA, KV-cache inference
  bpe.py                byte-level BPE tokenizer from scratch (GPT-2 style)
  trainer.py            seq2seq datasets (reverse/copy) + training loop
scripts/                runnable entry points
  train_seq2seq.py      seq2seq training (reverse / copy tasks)     → checkpoints/model.pt
  train_dist.py         distributed training (DDP) variant of train_seq2seq.py
  decode.py             greedy vs beam-search decoding on checkpoints/model.pt
  visualize.py          attention heatmaps                          → plots/*.png
  attention_dump.py     same attention weights as readable text
  gpt.py                mini-GPT CLI: train / sample, RoPE, GQA, BPE, AMP, KV cache
  verify.py             the verification harness — 8 proof sections, all runnable
  benchmark.py          training throughput: fp32 vs AMP vs torch.compile  → plots/benchmark.png
  scale.py              scaling-curve experiment (loss vs model size)      → plots/scaling_curve.png
  export_onnx.py        export GPT to ONNX, benchmark vs ONNX Runtime       → checkpoints/*.onnx
tests/
  test_equivalence.py   the proofs as pytest tests (python -m pytest tests)
  test_bpe.py           BPE round-trip / compression / determinism + training-knob tests
checkpoints/            trained models
  model.pt              trained seq2seq (reverse task)
  gpt.pt                trained GPT, learned positions
  gpt_rope.pt           trained GPT + RoPE, full attention
  gpt_rope_gqa.pt       trained GPT + RoPE + GQA (4 heads → 2 KV heads)
  gpt_rope_bpe.pt       trained GPT + RoPE with a byte-level BPE tokenizer
data/shakespeare.txt    character corpus for the GPT
notebooks/              interactive versions of the demos
plots/                  output of visualize.py / benchmark.py / scale.py
docs/                   source of the GitHub Pages site (mkdocs, README mirror)
.github/                CI, docs deploy, dependabot, issue/PR templates
venv/                   local Python environment (not part of the project)
```

**Suggested order** (mirrored by this README): math → architecture → train
→ decode → inspect → GPT → artifacts → verification → notebooks.

---

## Part 3 — Step 1: the math of attention (`src/transformer/attention.py`)

Everything in this repo is a wrapper around one formula:

```
         Q · Kᵀ
S  =  ───────────     scores: how much each query wants each key
           √d_k

P  =  softmax(S)      probabilities over keys (rows sum to 1)

Y  =  P · V           output: weighted average of the values
```

`src/transformer/attention.py` does three things, *numerically*:

**3.1 Forward and backward, by hand.** It implements `∂L/∂Q, ∂L/∂K, ∂L/∂V`
with the chain rule and the **softmax Jacobian trick**: the full Jacobian
of softmax is an n×n matrix per row, but the derivative simplifies to
`∂L/∂Sᵢ = Pᵢ(∂L/∂Pᵢ − Σⱼ Pⱼ·∂L/∂Pⱼ)` — O(n), not O(n²). The manual
gradients are checked against `torch.autograd` (relative error **~1e-15**)
and against brute-force finite differences (relative error **~1e-8**):

```
tensor |     autograd       manual    rel_err
    dQ |     1.609757     1.609757   4.14e-16
    dK |     1.596158     1.596158   7.65e-16
    dV |     2.368320     2.368320   2.81e-16
```

**3.2 Why divide by √d_k?** Dot products of length-`d_k` random vectors
have variance `d_k`. Without scaling, scores grow with `d_k`, softmax
saturates to one-hot, and its derivative approaches 0 — **vanishing
gradients**. The demo measures it: with 50 keys, the winning key's
probability grows **0.47 → 0.98** as `d_k` goes 8 → 4096 when unscaled,
while the scaled version stays flat at **0.125**.

> Study this file first. If you can prove *one* primitive by hand, the rest
> of the repo ("the cache is correct", "GQA is correct") is the same proof
> pattern applied higher up.

---

## Part 4 — Step 2: the architecture (`src/transformer/model.py`)

Each component is ~20–50 self-contained lines. The model is the full
encoder–decoder from *Attention Is All You Need*:

```
input ─▶ token_embedding ─▶ + positional_encoding ─▶ [encoder × N]
                                                       │ self-attention
                                                       ▼
output ◀─ Linear ◀─ output_proj ◀─ [decoder × N] ◀─── encoder memory
                                   │
                                   └ masked self-attention + cross-attention
```

### The components, one by one

| component | what it is | why it matters |
|---|---|---|
| `scaled_dot_product_attention` | the Part 3 formula, batched | the only "engine"; everything else is plumbing |
| `precompute_rope / rotate_half / apply_rope` | **RoPE**: rotate Q/K dim pairs `(2i, 2i+1)` by `pos·ωᵢ` | turns absolute positions into *relative* offsets inside attention; enables length extrapolation |
| `MultiHeadAttention` | `W_q/W_k/W_v/W_o` projections + head split + merge | learns where to look; supports GQA via `num_kv_heads` (Part 10) |
| `FeedForward` | linear → ReLU → linear (`d_model → d_ff → d_model`) | the "memory" that processes what attention gathered |
| `LayerNorm` | normalizes each token's vector | training stability |
| `PositionalEncoding` | sinusoidal `sin/cos` table | the paper's original order injection |
| `EncoderLayer / DecoderLayer` | attention + FFN + residual + norm, in canonical order | the blueprint the paper prescribes |
| `NoamSchedule` | `lr = d_model⁻⁰·⁵ · min(step⁻⁰·⁵, step·warmup⁻¹·⁵)` | warmup-then-decay learning rate |
| `create_pad_mask / create_look_ahead_mask / create_masks` | the three masking helpers | padding ignored; decoder never sees the future |

**Residuals + norms everywhere**: `x = norm(x + attention(x))` — the trick
that lets gradients flow around attention entirely. The barrier to very
deep transformers, solved in 2015, absorbed here for free.

### Why this file is trustworthy

No `nn.MultiheadAttention` — every reshape, transpose, and multiply is
explicit, so the *shapes are the documentation*. If a shape comment is
wrong, `src/transformer/attention.py` or `scripts/verify.py` will refuse to
pass silently.

---

## Part 5 — Step 3: train a seq2seq model (`scripts/train_seq2seq.py`)

Two toy tasks that are trivial to verify and impossible to memorize:

```bash
python scripts/train_seq2seq.py --task reverse --epochs 100     # [2,5,1,3] → [3,1,5,2]
python scripts/train_seq2seq.py --task copy    --epochs 100     # [2,5,1,3] → [2,5,1,3]
```

What happens, step by step:

1. **Data**: 2,000 random length-5 sequences, vocab 20 (SOS=1, PAD=0, real
   tokens 2–19). `ReverseDataset` / `CopyDataset` differ only in `transform`.
2. **Teacher forcing**: each sample becomes `input = <SOS> + target[:-1]`,
   `output = target`. The decoder predicts the next token at every position
   from the *true* prefix — never its own guesses.
3. **Masking**: `create_masks` = pad mask AND causal mask; the decoder
   attention matrix must be lower-triangular.
4. **Training**: Adam (base lr 1.0 — the Noam scheduler controls the real
   rate) + cross-entropy on non-pad tokens. Best checkpoint saved as
   `checkpoints/model.pt`.

Expected behavior (MPS ≈ these numbers):

```
Epoch 10 | Loss: 1.2013 | LR: 4.98e-03     ← warmup, learning fast
Epoch 30 | Loss: 0.4423 | LR: 2.87e-03
Epoch 60 | Loss: 0.1249 | LR: 2.03e-03     ← decay phase
Epoch 100| Loss: 0.0387 | LR: 1.57e-03
```

then 5 fresh examples with ✓/✗ verdicts — typically all correct. ~171k
params; seconds per 10 epochs on MPS, ~1 min on CPU.

> **Pause-worthy insight**: with 20⁵ possible inputs and only 2,000 training
> sequences, the model *cannot* memorize — it must learn the abstract rule
> "output = reversed input". Accuracy on unseen inputs is real
> generalization, the same mechanism that lets a 100× bigger model "reverse"
> sentences in English.

---

## Part 6 — Step 4: decoding strategies (`scripts/decode.py`)

**Greedy**: argmax at every step. Fast, but one wrong choice is
unrecoverable — attention never backtracks.

**Beam search**: keep the `beam_size` best partial sequences by joint
log-probability at each step, expanding only those. The demo runs the whole
beam in **one batched forward pass** and unpicks the flattened `(beam ×
vocab)` scores; scores live in log space and can be length-normalized.

```bash
python scripts/decode.py --task reverse --beam-size 4
```

```
Accuracy on 8 examples
    greedy: 8/8
     beam4: 8/8
```

On this toy task greedy already wins; the point is *why* beam exists (greedy
can't recover from errors) and how it is implemented (batched, log-space,
length-normalized).

---

## Part 7 — Step 5: look inside attention (`scripts/visualize.py` / `scripts/attention_dump.py`)

```bash
python scripts/visualize.py --task reverse     # → plots/*.png heatmaps
python scripts/attention_dump.py               # same data as text matrices
```

Six PNGs: encoder self-attention, decoder *masked* self-attention, and
decoder cross-attention — two layers each, all heads.

**What you will see, and what it means:**

- **Decoder self-attention** must be lower-triangular: position `i` attends
  only to positions `≤ i`. Heat above the diagonal = broken mask.
- **Cross-attention** (queries = target, keys = source): a reverse model
  attends to the source token it needs for the current output slot — the
  model "pointing" at its memory.
- Convention used everywhere: **rows = queries, columns = keys** — matching
  the canonical `attn_weights[batch, head]` orientation.
  (`scripts/attention_dump.py` labels both axes explicitly.)

---

## Part 8 — Step 6: multi-GPU training (`scripts/train_dist.py`)

A production concern, scaffolded honestly: what if one GPU is not enough?

**DistributedDataParallel (DDP)** mirrors the model on each rank, feeds
each rank a different shard of the batch, and all-reduces the gradients
before every optimizer step. The effective batch becomes
`batch_size × world_size`; each step trains like a single, bigger machine.

```bash
# single process (macOS / CPU smoke test):
python scripts/train_dist.py --task reverse --epochs 20

# multi-GPU box:
torchrun --nproc_per_node=2 scripts/train_dist.py --task reverse --epochs 20
```

Details worth knowing:

- ranks are seeded identically (`torch.manual_seed(42)`) so every rank
  starts from the **same weights**,
- `DistributedSampler` shards the dataset; `set_epoch(epoch)` reshuffles
  differently on each rank,
- a `dist.barrier()` before the loop avoids startup deadlock,
- losses are all-reduced and averaged so every rank prints the same number.

---

## Part 9 — Step 7: the mini-GPT (`scripts/gpt.py`)

The decoder-only half of the transformer — no encoder, no cross-attention —
trained as a **character-level language model** on Shakespeare.

```bash
python scripts/gpt.py --epochs 30 --save                                           # learned positions → checkpoints/gpt.pt
python scripts/gpt.py --epochs 30 --rope --save --save-path checkpoints/gpt_rope.pt
python scripts/gpt.py --epochs 30 --rope --save --save-path checkpoints/gpt_rope_gqa.pt --kv-heads 2
```

30 epochs over 20,000 random 128-character slices (a few minutes on MPS),
keeping the best-`val_loss` checkpoint, which stores:

```python
{"model": state_dict, "tokenizer": tokenizer, "val_loss": best_val, "num_kv_heads": n}
```

### CLI reference

| flag | default | what it does |
|---|---|---|
| `--epochs` | 30 | training epochs |
| `--block-size` | 128 | context window (training + generation limit) |
| `--d-model / --heads / --d-ff / --layers` | 128 / 4 / 512 / 4 | architecture |
| `--batch-size` | 64 | batch size |
| `--rope` | off | rotary position embeddings instead of learned |
| `--kv-heads` | None | GQA: KV heads per layer (must divide `--heads`) |
| `--save / --load` | off | train-and-save / load-and-sample |
| `--save-path / --load-path` | checkpoints/gpt.pt | checkpoint file |
| `--prompt / --n-chars / --temperature` | "To be, or not to be" / 400 / 0.8 | sampling |
| `--top-p` | 1.0 | nucleus sampling mass (1.0 = plain temperature) |
| `--seed` | None | bit-for-bit reproducible training |
| `--amp` | off | fp16 autocast + GradScaler (CUDA/MPS); measures speed vs fp32 |
| `--grad-accum` | 1 | accumulate N batches per optimizer step (bigger effective batch) |
| `--compile` | off | wrap the model with `torch.compile` |
| `--tb-log` | None | log loss/ppl/lr to a TensorBoard dir (`pip install tensorboard`) |
| `--bpe` | off | byte-level BPE tokenizer instead of characters |
| `--bpe-vocab-size` | 512 | BPE vocab (256 base bytes + merges) |
| `--tie-weights` | off | share the token embedding and output head (GPT-2 style) |
| `--num-pairs` | 20000 | training slices per epoch |
| `--num-val-pairs` | 2000 | validation slices per epoch |

### The two inference paths

| path | cost | how it works |
|---|---|---|
| `generate()` | O(T²) | re-runs the whole prefix through all layers at every step — the naive baseline |
| `generate_cached()` | O(T) | prefill once, then one forward pass per new token, reusing past K/V — what llama.cpp / vLLM / HF serve |

The KV cache works because a past token's K and V never change once
computed — caching them is a pure win. But there is a famous bug waiting
here, and this repo's history includes it: **the prefill must run with the
causal mask**. Without it, early positions attend to the future, their
hidden states are wrong, and every later cache entry is silently poisoned.
`scripts/verify.py` exists precisely to catch that class of bug.

### RoPE in `scripts/gpt.py`

- Q rotates with the full-width table; K's table has width
  `num_kv_heads · d_k` under GQA (falls back to Q's table in full MHA).
- Tables are `persistent=False` buffers: derived state that is **not**
  stored in checkpoints — old checkpoints keep loading (a real upgrade-path
  concern).
- **Extrapolation for free**: `_grow_rope_tables` lazily regenerates
  geometrically larger tables when generation passes `max_len`. RoPE
  frequencies are deterministic, so position 5000 is just another rotation.
  Learned embeddings have no such escape and raise a clear `ValueError`.

---

## Part 10 — Step 8: Grouped Query Attention

**The problem.** Inference memory is often dominated by the KV cache:

```
2 × num_heads × d_k × context_tokens × layers × 4 bytes
```

At production scale (70B models, 100k-token contexts) that is *tens of
gigabytes* per request.

**The observation.** Heads don't all need their own K/V — they can *share*.
Grouped Query Attention (GQA; Ainslie et al., 2023) gives `num_kv_heads <
num_heads` heads their own K/V, and groups of `num_heads ÷ num_kv_heads`
query heads share each one. (`num_kv_heads = 1` is the extreme: Multi-Query
Attention.) LLaMA-2/3 and Mistral all ship this.

**The invariant used here** (differs from some open-source
implementations): sharing shrinks the head *count*, never the head *width*
— `d_kv = d_k = d_model / num_heads`. So `W_k` is a
`(num_kv_heads · d_k) × d_model` matrix, and `assert num_heads %
num_kv_heads == 0` guards the grouping.

**The cache discipline.** MHA always appends K/V in the *small*
`num_kv_heads` form and only calls `repeat_interleave(group_size, dim=1)`
to expand for scoring. The cache the model returns is the small form —
`(B, kv_heads, T, d_k)`, e.g. `(1, 2, 13, 32)` — never the expanded one.
Expanding at store time would throw the whole saving away.

**Measured payoff** (Part 12): 512 KiB → 256 KiB (GQA) → 128 KiB (MQA,
`--kv-heads 1`) of KV cache at 128-token context, at the cost of +0.03 val
loss (≈0.15 ppl) for GQA — MQA lands *between* them (ppl 4.55) — with decode
throughput unchanged at this scale. The win is *memory*; at production
scale that memory becomes long-context capability.

**Two independent proofs that GQA computes the same function:**

- cached vs naive under GQA: prefill **|Δ| = 0.0**, one-token step ≤ 1e-5;
- a **hand-rolled group loop** (query head `i` uses KV head
  `i // group_size`, sliced explicitly, no `repeat_interleave` anywhere)
  matches the vectorized path bit-for-bit — the grouping order itself
  cannot be wrong.

---

## Part 11 — Step 9: the verification suite (`scripts/verify.py`)

The project's soul. Eight sections; every number printed is *actually
measured* (the whole suite reruns in under a minute):

```bash
python scripts/verify.py                                        # learned positions
python scripts/verify.py --rope --ckpt checkpoints/gpt_rope.pt  # RoPE, full attention
python scripts/verify.py --rope --ckpt checkpoints/gpt_rope_gqa.pt  # RoPE + GQA
```

| # | what is proven | method | example result |
|---|---|---|---|
| 1 | RoPE encodes *relative* positions | `score[m+1,n+1] == score[m,n]` across the whole matrix (Toeplitz property) | max deviation 2.86e-06 |
| 2 | KV cache ≡ recomputation | prefill logits through the cache vs plain forward pass | max \|Δ\| = **0.0** (bit-identical) |
| 3 | the cache is faster | naive vs cached generation, same seed & temperature | ~2–3× (O(T²) vs O(T)) |
| 4 | RoPE path is cache-safe | cached K stays rotated at its *original* position | step \|Δ\| = 3.6e-07 |
| 5 | length extrapolation | RoPE generates past the 128-token window; learned positions refuse with a clear `ValueError` | 170 tokens (42 past the window) |
| 6 | GQA correctness + cache math | trained checkpoint: prefill/step equality; cache `(1,2,13,32)` small-form vs `(1,4,13,32)` expanded; byte math; hand-rolled group loop | prefill 0.0, step 4.1e-06, loop bit-identical |
| 7 | trained head-to-head | full MHA vs GQA vs MQA checkpoints: cache bytes at full context, cached-decode tok/s, perplexity | 512→256→128 KiB; ~equal tok/s; ppl 4.49 / 4.64 / 4.55 |
| 8 | BPE tokenizer (from scratch) | exact encode→decode round-trips; corpus compression; trained BPE GPT vs char GPT at equal *per-character* cost | round-trip OK; 0.51 tok/char; BPE char-ppl ≈ char-ppl |

The *honest* part: thresholds are explicit (prefill `== 0.0`, step `< 1e-4`,
same-seed text equality) and the numbers printed are whatever they are — if
a run degrades, it says so in plain sight.

---

## Part 12 — Step 10: BPE tokenizer & training knobs

**Byte-level BPE** (`src/transformer/bpe.py`) is the tokenizer modern models
actually use (GPT-2, Llama, tiktoken). It treats the corpus as raw UTF-8
bytes — 256 base tokens — then repeatedly merges the most frequent adjacent
pair into a new token until a vocabulary budget is reached. The merge list
*is* the tokenizer: no unknown tokens, exact decode, real word fragments
("th", "ing") learned from data.

```bash
# train a GPT on BPE tokens instead of characters:
python scripts/gpt.py --epochs 30 --rope --save --bpe --save-path checkpoints/gpt_rope_bpe.pt
```

Measured on Shakespeare with a 512-token BPE vocab: the corpus drops from
1.0 to **0.51 tokens/char**, and a BPE GPT reaches roughly the same
**per-character** perplexity as the character model — with a 4× smaller
context window for the same information. (§8 of `scripts/verify.py` proves
round-trips and does the fair per-character comparison.)

**Training knobs** (all verified in `tests/test_bpe.py`):

- `--amp`: fp16 autocast + GradScaler on CUDA/MPS. Measure it:
  `python scripts/benchmark.py`.
- `--grad-accum N`: accumulate N batches before each optimizer step — bigger
  effective batches on small machines (Noam lr stepping stays aligned).
- `--compile`: `torch.compile(model)` — one line, Inductor kernels.
- `--tb-log DIR`: TensorBoard scalars (loss/ppl/lr per epoch).
- `--bpe / --bpe-vocab-size`: tokenizer swap.
- `--tie-weights`: share the embedding and the output head (GPT-2 style) —
  `vocab · d_model` fewer parameters, verified in `tests/test_bpe.py`.

**Benchmarks & experiments:**

```bash
python scripts/benchmark.py    # fp32 vs AMP vs compile vs amp+compile  → plots/benchmark.png
python scripts/scale.py        # loss vs model size (4 tiny GPTs)       → plots/scaling_curve.png
python scripts/export_onnx.py  # GPT → ONNX (dynamic batch/seq) + ORT speed test
```

`scripts/scale.py` is the Chinchilla-style story at toy scale: four GPTs
(32 → 256 `d_model`) trained on the same data with the same recipe, showing
validation loss falling as parameters grow. `scripts/export_onnx.py` exports
`checkpoints/gpt_rope.pt` to ONNX with dynamic batch/sequence axes and
checks ONNX Runtime logits against PyTorch (`pip install onnx onnxruntime`).

---

## Part 13 — Results at a glance

### Seq2seq (`checkpoints/model.pt`, reverse task, 100 epochs)

| metric | value |
|---|---|
| parameters | 171,284 |
| final train loss / LR | 0.039 / 1.57e-3 |
| greedy / beam-4 accuracy (8 unseen examples) | 8/8 |

### Character GPT (30 epochs, Shakespeare, block 128)

| checkpoint | positions | tokenizer | heads → KV | params | val loss (best, stored) | perplexity |
|---|---|---|---|---|---|---|
| `checkpoints/gpt.pt` | learned | chars | 4 → 4 | 826,433 | 1.498 † | 4.47 |
| `checkpoints/gpt_rope.pt` | RoPE | chars | 4 → 4 | 810,049 | 1.5023 | 4.49 |
| `checkpoints/gpt_rope_gqa.pt` | RoPE | chars | 4 → 2 | 744,001 | 1.5339 | 4.64 |
| `checkpoints/gpt_rope_mqa.pt` | RoPE | chars | 4 → 1 | 710,977 | 1.5143 | 4.55 |
| `checkpoints/gpt_rope_bpe.pt` | RoPE | BPE-512 | 4 → 4 | 924,928 | 3.0439 ‡ | ~20.7 (per BPE token) |

† `checkpoints/gpt.pt` predates checkpoint metadata; its val loss was
measured independently (same eval recipe as training). Reproduce any row
with `--seed N` for bit-reproducibility (e.g. `scripts/gpt.py --epochs 30
--rope --seed 0 --save`).

‡ BPE checkpoints are not directly comparable with the char rows — their
tokens carry ~2× the information. `scripts/verify.py` §8 compares both
models at equal *per-character* cost.

---

## Part 14 — Notebooks

`notebooks/01_transformer_walkthrough.ipynb` is the README's story in
interactive form: attention weights, the trained seq2seq's cross-attention
heatmap, GPT sampling, KV-cache equivalence, RoPE's Toeplitz property,
170-token extrapolation, and the GQA cache math — one cell per claim.

```bash
source venv/bin/activate
pip install -r requirements.txt      # includes jupyter/ipykernel/nbformat
python -m ipykernel install --user   # register the kernel
jupyter notebook notebooks/
```

The notebook is generated from `build_notebook.py` (keeps the JSON
canonical; `python3 build_notebook.py --execute` also runs every cell) and
has been executed end-to-end with zero errors on this repo's checkpoints.

---

## Part 15 — Troubleshooting

| symptom | cause & fix |
|---|---|
| `module '__main__' has no attribute 'CharTokenizer'` | checkpoints pickle the tokenizer as `__main__.CharTokenizer`; any loader must import (or define) `CharTokenizer` *in its own namespace* before `torch.load`. `scripts/verify.py` and the notebook both do this — copy the pattern for new loaders. |
| `gpt_rope.pt was trained with RoPE — pass --rope to load it` | positional-style mismatch between CLI flag and checkpoint; `scripts/gpt.py` detects both directions and says so instead of dying inside `load_state_dict`. |
| `Missing key(s) in state_dict: pos_embedding.weight` | loaded a learned-positions checkpoint as RoPE (or vice versa) outside `scripts/gpt.py`. Match `--rope` to the checkpoint. |
| always falls back to CPU | `torch.backends.mps.is_available()` is false on that machine; CPU is fine at these sizes. |
| `torchrun` not found / DDP errors on macOS | `scripts/train_dist.py` falls back to single-process without CUDA; on a multi-GPU box install the CUDA torch build and use `torchrun --nproc_per_node=N`. |
| numbers drift between runs | sampling demos seed before *both* paths (`torch.manual_seed(n)`); for training use `--seed` for bit-reproducibility. |
| blank plots | `checkpoints/model.pt` must exist — run `python scripts/train_seq2seq.py --task reverse --epochs 100` first. |

---

## Part 16 — FAQ

**Why is prefill bit-identical (|Δ| = 0.0) but the one-token step only
< 1e-4?** Prefill literally runs the same computation as the naive forward
(cached K = concatenation, no reordering), while the step compares a
one-token *append* against a full-*batch recomputation* — same math,
different fp32 operation order → tiny differences. The tolerance is loose
enough for fp32, tight enough to catch real bugs.

**Temperature vs top-p?** Temperature reshapes the softmax via
`logits/τ` — lower τ = sharper = greedier. Top-p (nucleus) keeps only the
smallest set of tokens whose cumulative probability exceeds `p`, then
re-normalizes — it trims the long tail without flattening the head.
`scripts/gpt.py` implements both; τ=0.8, p=0.9 is a sane starting point.

**RoPE or learned embeddings?** Learned: slightly better exactly at the
training window (`checkpoints/gpt.pt` is the best val here). RoPE: infinitely
extensible and what modern models use — length extrapolation is its whole
point. At production scale there's no contest; RoPE or a descendant wins.

**Why post-norm (`norm(x + attn(x))`) instead of pre-norm?** This is the
2017 paper's literal arrangement. Modern practice moved to pre-norm
(`x + attn(norm(x))`) for 100+ layer stability; both work fine at the 2–4
layers used here, which is why the original is kept for fidelity.

**Why 4 heads?** Toy scale — nothing constrains it (`--heads 8 --d-model
128` trains fine; attention parameters roughly double).

**What does the KV cache actually store under GQA?** The small form
`(B, num_kv_heads, T, d_k)` — e.g. `(1, 2, 13, 32)` — expanded per-head
only inside the scoring op. This single discipline is why GQA's memory win
is real (Part 10).

---

## Part 17 — Roadmap: improvements & advancement

Ordered beginner-safe → research-grade. Every item fits this repo's spirit:
make the change, then *prove* it (equivalence check) or *measure* the delta
(val loss, speed, cache bytes).

**Easy (a weekend each)**

1. ✅ **Weight tying** — done (`--tie-weights`): embedding and output head
   share one matrix, `vocab · d_model` fewer parameters; round-trip tested.
2. ✅ **Gradient accumulation** — done (`--grad-accum N`), including
   alignment with the Noam lr schedule.
3. **Longer training** — 60 epochs on the same data; val loss keeps falling
   (the model underfits at 30). Watch for the plateau; log it.
4. **Bigger architecture** — `--heads 8 --layers 6 --d-model 256` is
   already supported; bigger model + more epochs is the single best quality
   lever here.
5. ✅ **`torch.compile(model)`** — done (`--compile`); benchmark it with
   `scripts/benchmark.py`.
6. **LR curve plot** — graph `NoamSchedule` and confirm the warmup peak
   matches the paper formula.

**Medium**

7. ✅ **FP16/BF16 autocast** — done (`--amp`); `scripts/benchmark.py` prints
   the fp32-vs-AMP throughput delta, and the tests cover the GradScaler path.
8. **NTK-aware / linear scaling for long context** — rescale RoPE
   frequencies to reach 512-token windows without retraining from scratch;
   extend §5 of `scripts/verify.py` to compare 128 vs 512 behavior.
9. ✅ **BPE tokenizer** — done (`--bpe`, `src/transformer/bpe.py`); GPT-2
   style byte-level merges, verified in `scripts/verify.py` §8 and
   `tests/test_bpe.py`.
10. **Eval harness** — a fixed validation split and a small
    `bench.py` printing val/ppl/sample for every checkpoint: the Part 13
    table, automated.
11. **Beam search for `scripts/gpt.py`** — port `scripts/decode.py`'s
    batched beam as `--beam`; greedy vs beam on characters with a best-path
    comparison.

**Advanced (research-shaped)**

12. **Scaling-curve study** — `scripts/scale.py` trains 4 tiny GPTs and plots
    loss vs parameters; the next step is 8+ points and a fitted power law.
13. **Long-context benchmark** — 512-token generation with `--rope`: does
    quality fade gracefully? Plot ppl vs position for honest extrapolation
    curves (the §5 demo, quantified).
14. **Speculative decoding** — a tiny draft model proposing K tokens, the
    main model verifying them in one batched pass; measure wall-clock vs
    acceptance rate. (The KV-cache machinery here is the prerequisite.)
15. **Alignment: SFT → RM → PPO** — supervised fine-tuning on curated
    Shakespeare, a small reward model, then PPO with the cached generator
    as the actor. The reward signal: "is this line in Shakespeare's style?"
    Everything (reward model, PPO loop) fits the from-scratch philosophy.
16. **Serving shim** — wrap `generate_cached` in a tiny HTTP/SSE handler;
    the KV-cache path is already the production-shaped one. (An ONNX export
    is already available via `scripts/export_onnx.py`.)

---

## Appendix — The verification philosophy

Nothing in this repo is *believed*; everything is *checked*:

| claim | proof |
|---|---|
| attention math | manual backward ≡ `torch.autograd` (≈1e-15), finite differences (≈1e-8) |
| KV cache (learned + RoPE + GQA) | cached ≡ naive logits; prefill bit-identical, step < 1e-4 |
| generation paths | same seed → identical sampled text |
| GQA grouping | vectorized path ≡ explicit per-group loop, bit-for-bit |
| length extrapolation | 42 tokens past the training window, generated; learned positions fail loudly |
| sampling compatibility | `--top-p`/`--temperature` share one `_sample_token` used by both paths |

Thresholds are explicit and logged with the numbers — upgrade the model,
rerun the suite, and the table tells you the truth.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — `1706.03762`; the
  architecture, Noam schedule, positional encodings.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*
  (2021) — RoPE; what makes §1 and §5 of `scripts/verify.py` true.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints* (2023) — Grouped Query Attention, Part 10.
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2019) —
  nucleus (top-p) sampling, implemented in `scripts/gpt.py`.
- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with
  Subword Units* (2016) — BPE for tokenization, `src/transformer/bpe.py`.
- Radford et al., *Language Models are Unsupervised Multitask Learners*
  (2019) — GPT-2's byte-level BPE and weight tying, Part 12.
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020) — the
  loss-vs-parameters story behind `scripts/scale.py`.
- Karpathy, *nanoGPT* — the pedagogical ancestor of `scripts/gpt.py`'s
  training setup (data, block sizes, checkpointing ideas).
- The PyTorch documentation on `MultiHeadAttention`, `DistributedDataParallel`,
  `torch.compile`, and `torch.backends.mps` — the only external APIs this
  project touches.