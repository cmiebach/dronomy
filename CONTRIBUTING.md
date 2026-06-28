# Contributing

## Language policy

**Everything in this repository is written in English** — source code, comments,
docstrings, commit messages, pull-request descriptions, and all documentation
(`README.md`, `STRUCTURE.md`, `STATUS.md`, etc.). The team's working language is English; please keep all committed content
English-only. Anything printed at runtime (console output, log messages) must be
plain ASCII — em-dashes and other non-ASCII characters mangle on Windows consoles.

## Data never goes through git

Drone videos, extracted frames, satellite tiles, GPS tracks, model outputs,
`.env` files, and any NDA material are git-ignored and must never be committed
(see `.gitignore`). GitHub is the source of truth for **code**; data is moved
out-of-band. The localization model stays **telemetry-free** — GPS is used only
as ground truth in evaluation, never as a model input.

## Tests

Tests are offline, deterministic (seeded), and require no network, GPU, or real
video. Run the suite before pushing:

```bash
python -m pytest -q
```

A fresh checkout works without an install step: `[tool.pytest.ini_options]
pythonpath=["src"]` puts the package on the path, and `scripts/_bootstrap.py`
does the same for the CLI scripts. For the full dev setup, `pip install -e ".[dev]"`.
