# Roadmap: improvements & advancement

Ordered beginner-safe → research-grade. Every item fits this repo's spirit:
make the change, then *prove* it (equivalence check) or *measure* the delta
(val loss, speed, cache bytes).

## Easy (a weekend each)

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

## Medium

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
11. ✅ **Beam search for `scripts/gpt.py`** — port `scripts/decode.py`'s
    batched beam as `--beam`; greedy vs beam on characters with a best-path
    comparison.

## Advanced (research-shaped)

12. **Scaling-curve study** — `scripts/scale.py` trains 4 tiny GPTs and plots
    loss vs parameters; the next step is 8+ points and a fitted power law.
13. ✅ **Long-context benchmark** — 512-token generation with `--rope`: does
    quality fade gracefully? Plot ppl vs position for honest extrapolation
    curves (the §5 demo, quantified). — `scripts/long_context_benchmark.py`
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

*Status legend: ✅ = implemented, blank = open.*