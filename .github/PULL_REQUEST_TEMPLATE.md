## Summary

<!-- One or two sentences: what and why. Link the issue: "Fixes #123". -->

## Verification

<!-- The repo's rule: prove or measure. Paste the relevant output, e.g. -->
<!-- - scripts/verify.py numbers (prefill |Δ|, step |Δ|, ppl, tok/s) -->
<!-- - pytest / ruff / mypy output -->
<!-- - benchmark numbers for a perf change -->

- [ ] `ruff check .` passes
- [ ] `mypy src/transformer` passes
- [ ] `python -m pytest tests -q` passes
- [ ] `python build_notebook.py` passes (if relevant)
- [ ] `python scripts/verify.py --rope --ckpt checkpoints/gpt_rope.pt --n-chars 20` passes

## Checklist

- [ ] No new dependencies (or justified in pyproject extras)
- [ ] README updated if user-facing behavior changed
- [ ] New behavior is opt-in via a CLI flag that defaults to off (if applicable)