---
name: Bug report
about: Something is wrong, or a verification check fails
title: ""
labels: bug
assignees: ""
---

**What did you run?**

Command, script, and any flags (e.g. `python scripts/verify.py --rope --ckpt checkpoints/gpt_rope.pt`).

**Expected behavior**

**Actual behavior**

Include the output — especially the numbers from `scripts/verify.py`
(max |Δ|, ppl, tok/s). This repo's whole point is that numbers are checked,
so paste them.

**Environment**

- OS / device (CPU, MPS, CUDA):
- `python -c "import torch; print(torch.__version__)"`:
- Repo commit (`git log -1 --oneline`):

**Checklist**

- [ ] I ran `ruff check .` and `mypy src/transformer`
- [ ] I can reproduce with the checkpoints in `checkpoints/` (or I say which
      training step is missing)