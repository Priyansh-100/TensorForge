# Contributing to TensorForge

First off: thank you for considering a contribution. This project's core
value is **verifiability** — every claim about the models is backed by an
executable check. Please keep that spirit in any change you make.

## Code of conduct

Be kind and constructive. That's the whole policy.

## Ground rules

1. **Nothing is believed, everything is checked.** If you change behavior,
   add or extend a proof (a section in `scripts/verify.py`, a test in
   `tests/`, or both).
2. **Measure, don't assert.** New optimizations (caching, precision,
   quantization, ...) must come with a before/after number: val loss,
   tokens/sec, cache bytes.
3. **No new dependencies without a discussion.** The repo runs on torch +
   numpy + matplotlib; optional tooling goes in `pyproject.toml` extras
   (`[project.optional-dependencies]`).
4. **Keep the public API honest.** New CLI flags default to off and must
   not change existing behavior.

## Development setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Checks to pass before opening a PR

```bash
ruff check .                    # lint
mypy src/transformer            # type check
python -m pytest tests -q       # tests (train checkpoints first if missing)
python scripts/verify.py --rope --ckpt checkpoints/gpt_rope.pt --n-chars 20  # proof suite
python build_notebook.py        # notebook JSON stays canonical
```

The CI workflow (`.github/workflows/ci.yml`) runs all of these; a ruleset
on `main` requires them to pass on the PR.

## What needs work

See **Part 17 — Roadmap** in the README. The `Easy` items are a great
starting point. If you pick one up, say so in the issue thread first.

## Suggested workflow

1. Open an issue or comment on one (feature / bug / question).
2. Fork the repo, branch from `main`.
3. Make the change **with its proof/measurement**.
4. Run the checks above.
5. Open a PR referencing the issue. CI + the `main` ruleset will verify it.

Questions are welcome in issues — nothing here is too simple to ask about.